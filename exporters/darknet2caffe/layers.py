# --------------------------------------------------------------------------
# layers.py
# --------------------------------------------------------------------------
# This file is part of:
# SushiConverter
# https://github.com/SushiSystems/SushiConverter
# https://sushisystems.io
# --------------------------------------------------------------------------
# Copyright (c) 2026-present  Mustafa Garip & Sushi Systems
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# --------------------------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
import core.caffe_pb2 as pb
from core.logger import log_info, log_error, log_warning
from .utils import numpy_to_caffe_blob, get_nearest_weights

class LayerMapper:
    """
    Registry for Darknet-to-Caffe operator mapping functions.
    """
    _registry = {}

    @classmethod
    def register(cls, op_type):
        def decorator(func):
            cls._registry[op_type] = func
            return func
        return decorator

    @classmethod
    def map_block(cls, module, block, module_idx, bottom, net_msg, top_names, net):
        """
        Dispatches darknet block to the appropriate mapper.
        """
        b_type = block['type']
        if b_type in cls._registry:
            return cls._registry[b_type](module, block, module_idx, bottom, net_msg, top_names, net)
        else:
            log_error(f"Unsupported Darknet block type: {b_type}. Exporting a broken Caffe graph is prohibited.")
            raise NotImplementedError(f"Darknet to Caffe mapping for block type '{b_type}' is currently not implemented.")

@LayerMapper.register('convolutional')
def map_convolutional(module, block, idx, bottom, net_msg, top_names, net):
    conv_layer = module[0]
    batch_normalize = int(block.get('batch_normalize', 0))
    activation = block.get('activation', 'linear')

    conv_name = f"conv_{idx}"
    layer = net_msg.layer.add()
    layer.name = conv_name
    layer.type = "Convolution"
    layer.bottom.append(bottom)
    layer.top.append(conv_name)
    
    conv_param = layer.convolution_param
    conv_param.num_output = conv_layer.out_channels
    
    k = conv_layer.kernel_size[0] if isinstance(conv_layer.kernel_size, tuple) else conv_layer.kernel_size
    s = conv_layer.stride[0] if isinstance(conv_layer.stride, tuple) else conv_layer.stride
    p = conv_layer.padding[0] if isinstance(conv_layer.padding, tuple) else conv_layer.padding
    
    conv_param.kernel_size.append(k)
    conv_param.stride.append(s)
    conv_param.pad.append(p)
    conv_param.bias_term = not batch_normalize
    if conv_layer.groups > 1:
        conv_param.group = conv_layer.groups

    layer.blobs.extend([numpy_to_caffe_blob(conv_layer.weight.data.cpu().numpy())])

    if not batch_normalize:
        layer.blobs.extend([numpy_to_caffe_blob(conv_layer.bias.data.cpu().numpy())])

    current_top = conv_name

    if batch_normalize:
        bn_layer = module[1]
        bn_name = f"bn_{idx}"
        layer_bn = net_msg.layer.add()
        layer_bn.name = bn_name
        layer_bn.type = "BatchNorm"
        layer_bn.bottom.append(current_top)
        layer_bn.top.append(bn_name)
        current_top = bn_name
        
        bn_param = layer_bn.batch_norm_param
        bn_param.use_global_stats = True
        bn_param.eps = 1e-5

        mean_blob = numpy_to_caffe_blob(bn_layer.running_mean.cpu().numpy())
        var_blob = numpy_to_caffe_blob(bn_layer.running_var.cpu().numpy())
        
        scale_factor = pb.BlobProto()
        scale_factor.data.append(1.0)
        scale_factor.shape.dim.append(1)
        
        layer_bn.blobs.extend([mean_blob, var_blob, scale_factor])

        scale_name = f"scale_{idx}"
        layer_scale = net_msg.layer.add()
        layer_scale.name = scale_name
        layer_scale.type = "Scale"
        layer_scale.bottom.append(current_top)
        layer_scale.top.append(scale_name)
        current_top = scale_name
        layer_scale.scale_param.bias_term = True
        
        layer_scale.blobs.extend([
            numpy_to_caffe_blob(bn_layer.weight.data.cpu().numpy()),
            numpy_to_caffe_blob(bn_layer.bias.data.cpu().numpy())
        ])

    if activation == 'leaky':
        act_name = f"leaky_{idx}"
        layer_act = net_msg.layer.add()
        layer_act.name = act_name
        layer_act.type = "ReLU"
        layer_act.bottom.append(current_top)
        layer_act.top.append(act_name)
        current_top = act_name
        layer_act.relu_param.negative_slope = 0.1
    elif activation == 'mish':
        raise NotImplementedError("Mish activation is not standard Caffe. Please use an NPU-compatible activation like Leaky ReLU.")
    elif activation == 'relu':
        act_name = f"relu_{idx}"
        layer_act = net_msg.layer.add()
        layer_act.name = act_name
        layer_act.type = "ReLU"
        layer_act.bottom.append(current_top)
        layer_act.top.append(act_name)
        current_top = act_name

    top_names[idx] = current_top
    return current_top

@LayerMapper.register('maxpool')
def map_maxpool(module, block, idx, bottom, net_msg, top_names, net):
    pool_name = f"pool_{idx}"
    layer = net_msg.layer.add()
    layer.name = pool_name
    layer.type = "Pooling"
    layer.bottom.append(bottom)
    layer.top.append(pool_name)
    
    pool_param = layer.pooling_param
    pool_param.pool = pb.PoolingParameter.MAX
    
    size = int(block['size'])
    stride = int(block['stride'])
    
    pool_param.kernel_size = size
    pool_param.stride = stride
    
    if stride == 1 and size % 2:
        pool_param.pad = size // 2
        top_names[idx] = pool_name
    elif stride == 1 and size == 2:
        # Darknet asymmetric padding workaround
        pool_param.pad = 1
        crop_name = f"crop_{idx}"
        layer_crop = net_msg.layer.add()
        layer_crop.name = crop_name
        layer_crop.type = "Crop"
        layer_crop.bottom.append(pool_name)
        layer_crop.bottom.append(bottom)
        layer_crop.top.append(crop_name)
        layer_crop.crop_param.axis = 2
        layer_crop.crop_param.offset.extend([1, 1])
        top_names[idx] = crop_name
    else:
        top_names[idx] = pool_name
    return top_names[idx]

@LayerMapper.register('upsample')
def map_upsample(module, block, idx, bottom, net_msg, top_names, net):
    up_name = f"upsample_{idx}"
    layer = net_msg.layer.add()
    layer.name = up_name
    layer.type = "Deconvolution"
    layer.bottom.append(bottom)
    layer.top.append(up_name)
    
    stride = int(block['stride'])
    if idx > 0 and hasattr(net, 'out_filters'):
        prev_channels = net.out_filters[idx - 1]
    else:
        prev_channels = int(net.net_info.get('channels', 3))
    
    if prev_channels == 0: prev_channels = 1
    
    conv_param = layer.convolution_param
    conv_param.num_output = prev_channels
    conv_param.kernel_size.append(stride)
    conv_param.stride.append(stride)
    conv_param.pad.append(0)
    conv_param.bias_term = False
    conv_param.group = prev_channels
    
    weight = get_nearest_weights(prev_channels, stride)
    layer.blobs.extend([numpy_to_caffe_blob(weight)])
    
    top_names[idx] = up_name
    return up_name

@LayerMapper.register('route')
def map_route(module, block, idx, bottom, net_msg, top_names, net):
    layers = block['layers'].split(',')
    layers = [int(i) if int(i) >= 0 else int(i) + idx for i in layers]
    
    if len(layers) == 1 and 'groups' in block:
        groups = int(block['groups'])
        group_id = int(block['group_id'])
        
        conv_name = f"route_extract_{idx}"
        layer = net_msg.layer.add()
        layer.name = conv_name
        layer.type = "Convolution"
        layer.bottom.append(top_names[layers[0]])
        layer.top.append(conv_name)
        
        total_channels = net.out_filters[layers[0]]
        group_channels = total_channels // groups
        
        conv_param = layer.convolution_param
        conv_param.num_output = group_channels
        conv_param.kernel_size.append(1)
        conv_param.stride.append(1)
        conv_param.pad.append(0)
        conv_param.bias_term = False
        
        weight = np.zeros((group_channels, total_channels, 1, 1), dtype=np.float32)
        start_idx = group_channels * group_id
        for j in range(group_channels):
            weight[j, start_idx + j, 0, 0] = 1.0
        layer.blobs.extend([numpy_to_caffe_blob(weight)])
        
        top_names[idx] = conv_name
    else:
        route_name = f"route_{idx}"
        layer = net_msg.layer.add()
        layer.name = route_name
        layer.type = "Concat"
        for l in layers:
            layer.bottom.append(top_names[l])
        layer.top.append(route_name)
        top_names[idx] = route_name
    return top_names[idx]

@LayerMapper.register('shortcut')
def map_shortcut(module, block, idx, bottom, net_msg, top_names, net):
    short_name = f"shortcut_{idx}"
    layer = net_msg.layer.add()
    layer.name = short_name
    layer.type = "Eltwise"
    from_layer = int(block['from'])
    from_layer = from_layer if from_layer >= 0 else from_layer + idx
    layer.bottom.append(top_names[from_layer])
    layer.bottom.append(bottom)
    layer.top.append(short_name)
    layer.eltwise_param.operation = pb.EltwiseParameter.SUM
    top_names[idx] = short_name
    return short_name

@LayerMapper.register('yolo')
def map_yolo(module, block, idx, bottom, net_msg, top_names, net):
    if hasattr(net, 'no_yolo_layer') and net.no_yolo_layer:
        top_names[idx] = bottom
        return bottom
    
    raise NotImplementedError("YOLO layer conversion for Caffe is currently not supported. "
                              "Please do not use --yolo-layer to export raw tensors.")
