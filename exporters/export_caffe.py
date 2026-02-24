# --------------------------------------------------------------------------
# export_caffe.py
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

import os
import torch
import numpy as np
import torch.nn as nn
from core.logger import log_info, log_error, log_warning, log_success

try:
    import core.caffe_pb2 as pb
except ImportError:
    log_warning("caffe_pb2 missing. Generate it from caffe.proto.")
    pb = None

class CaffeExporter:
    """
    Exports Darknet PyTorch networks to Caffe models.
    """
    def __init__(self, net, shape):
        """
        Setup exporter state.
        @param net The model instance.
        @param shape Input image dimensions.
        """
        self.net = net
        self.shape = shape
        self.net_msg = pb.NetParameter()
        self.net_msg.name = "Darknet2Caffe"
        self.blobs = []
        self.layer_id = 0
        self.top_names = dict()

    def add_input(self):
        """
        Specifies the network input layer.
        """
        layer = self.net_msg.layer.add()
        layer.name = "data"
        layer.type = "Input"
        layer.top.append("data")
        shape = layer.input_param.shape.add()
        shape.dim.extend(self.shape)
        self.top_names[-1] = "data"

    def add_conv_bn_act(self, module, block, module_idx, bottom):
        """
        Maps a convolutional block.
        """
        conv_layer = module[0]
        batch_normalize = int(block.get('batch_normalize', 0))
        activation = block.get('activation', 'linear')

        conv_name = f"conv_{module_idx}"
        layer = self.net_msg.layer.add()
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

        weight_blob = pb.BlobProto()
        weight_blob.data.extend(conv_layer.weight.data.cpu().numpy().flatten())
        weight_blob.shape.dim.extend(list(conv_layer.weight.shape))
        layer.blobs.extend([weight_blob])

        if not batch_normalize:
            bias_blob = pb.BlobProto()
            bias_blob.data.extend(conv_layer.bias.data.cpu().numpy().flatten())
            bias_blob.shape.dim.extend(list(conv_layer.bias.shape))
            layer.blobs.extend([bias_blob])

        current_top = conv_name

        if batch_normalize:
            bn_layer = module[1]
            bn_name = f"bn_{module_idx}"
            layer_bn = self.net_msg.layer.add()
            layer_bn.name = bn_name
            layer_bn.type = "BatchNorm"
            layer_bn.bottom.append(current_top)
            layer_bn.top.append(current_top)
            
            bn_param = layer_bn.batch_norm_param
            bn_param.use_global_stats = True
            bn_param.eps = 1e-5

            mean_blob = pb.BlobProto()
            mean_blob.data.extend(bn_layer.running_mean.cpu().numpy().flatten())
            mean_blob.shape.dim.extend(list(bn_layer.running_mean.shape))
            
            var_blob = pb.BlobProto()
            var_blob.data.extend(bn_layer.running_var.cpu().numpy().flatten())
            var_blob.shape.dim.extend(list(bn_layer.running_var.shape))
            
            scale_factor = pb.BlobProto()
            scale_factor.data.append(1.0)
            scale_factor.shape.dim.append(1)
            
            layer_bn.blobs.extend([mean_blob, var_blob, scale_factor])

            scale_name = f"scale_{module_idx}"
            layer_scale = self.net_msg.layer.add()
            layer_scale.name = scale_name
            layer_scale.type = "Scale"
            layer_scale.bottom.append(current_top)
            layer_scale.top.append(current_top)
            
            layer_scale.scale_param.bias_term = True
            
            scale_w_blob = pb.BlobProto()
            scale_w_blob.data.extend(bn_layer.weight.data.cpu().numpy().flatten())
            scale_w_blob.shape.dim.extend(list(bn_layer.weight.shape))
            
            scale_b_blob = pb.BlobProto()
            scale_b_blob.data.extend(bn_layer.bias.data.cpu().numpy().flatten())
            scale_b_blob.shape.dim.extend(list(bn_layer.bias.shape))
            
            layer_scale.blobs.extend([scale_w_blob, scale_b_blob])

        if activation == 'leaky':
            act_name = f"leaky_{module_idx}"
            layer_act = self.net_msg.layer.add()
            layer_act.name = act_name
            layer_act.type = "ReLU"
            layer_act.bottom.append(current_top)
            layer_act.top.append(current_top)
            layer_act.relu_param.negative_slope = 0.1
        elif activation == 'mish':
            log_warning("Mish is not standard Caffe. Replacing with Linear.")
            pass
            
        self.top_names[module_idx] = current_top

    def add_maxpool(self, module, block, module_idx, bottom):
        """
        Maps a maxpooling layer.
        """
        pool_name = f"pool_{module_idx}"
        layer = self.net_msg.layer.add()
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
           
        self.top_names[module_idx] = pool_name

    def nearest_weight(self, channels, s):
        """
        Generates nearest neighbor weights.
        """
        weight = np.ones((channels, 1, s, s), dtype=np.float32)
        return weight

    def add_upsample(self, module, block, module_idx, bottom):
        """
        Maps upsampling using Deconvolution.
        """
        up_name = f"upsample_{module_idx}"
        layer = self.net_msg.layer.add()
        layer.name = up_name
        layer.type = "Deconvolution"
        layer.bottom.append(bottom)
        layer.top.append(up_name)
        
        stride = int(block['stride'])
        
        if module_idx > 0 and hasattr(self.net, 'out_filters'):
            prev_channels = self.net.out_filters[module_idx - 1]
        else:
            prev_channels = int(self.net.net_info.get('channels', 3))
            
        if prev_channels == 0:
            log_warning(f"Could not determine channels for Upsample. Defaulting to 1.")
            prev_channels = 1
            
        kernel_size = stride
        
        conv_param = layer.convolution_param
        conv_param.num_output = prev_channels
        conv_param.kernel_size.append(kernel_size)
        conv_param.stride.append(stride)
        conv_param.pad.append(0)
        conv_param.bias_term = False
        conv_param.group = prev_channels
        
        weight = self.nearest_weight(prev_channels, kernel_size)
        weight_blob = pb.BlobProto()
        weight_blob.data.extend(weight.flatten())
        weight_blob.shape.dim.extend(list(weight.shape))
        layer.blobs.extend([weight_blob])
        
        self.top_names[module_idx] = up_name

    def add_route(self, module, block, module_idx, bottom):
        """
        Maps routing using Concat.
        """
        route_name = f"route_{module_idx}"
        layer = self.net_msg.layer.add()
        layer.name = route_name
        layer.type = "Concat"
        
        layers = block['layers'].split(',')
        layers = [int(i) if int(i) >= 0 else int(i) + module_idx for i in layers]
        
        if 'groups' in block:
            log_warning("Group routing might fail in standard Caffe.")
            
        for l in layers:
            layer.bottom.append(self.top_names[l])
            
        layer.top.append(route_name)
        self.top_names[module_idx] = route_name

    def add_shortcut(self, module, block, module_idx, bottom):
        """
        Maps residual connections.
        """
        short_name = f"shortcut_{module_idx}"
        layer = self.net_msg.layer.add()
        layer.name = short_name
        layer.type = "Eltwise"
        
        from_layer = int(block['from'])
        from_layer = from_layer if from_layer >= 0 else from_layer + module_idx
        
        layer.bottom.append(self.top_names[from_layer])
        layer.bottom.append(bottom)
        layer.top.append(short_name)
        
        layer.eltwise_param.operation = pb.EltwiseParameter.SUM
        
        self.top_names[module_idx] = short_name

    def add_yolo(self, module, block, module_idx, bottom):
        """
        Maps a custom YOLO layer.
        """
        yolo_name = f"yolo_{module_idx}"
        layer = self.net_msg.layer.add()
        layer.name = yolo_name
        layer.type = "Yolo"
        layer.bottom.append(bottom)
        layer.top.append(yolo_name)
        
        self.top_names[module_idx] = yolo_name

    def build(self):
        """
        Sequentially builds the graph.
        """
        log_info("Converting architecture to Caffe Protobuf...")
        self.add_input()

        for i, (module, block) in enumerate(zip(self.net.models, self.net.blocks[1:])):
            b_type = block['type']
            bottom = self.top_names[i - 1]
            
            if b_type == 'convolutional':
                self.add_conv_bn_act(module, block, i, bottom)
            elif b_type == 'maxpool':
                self.add_maxpool(module, block, i, bottom)
            elif b_type == 'upsample':
                self.add_upsample(module, block, i, bottom)
            elif b_type == 'route':
                self.add_route(module, block, i, bottom)
            elif b_type == 'shortcut':
                self.add_shortcut(module, block, i, bottom)
            elif b_type == 'yolo':
                 if getattr(self.net, 'no_yolo_layer', False):
                     self.top_names[i] = bottom
                 else:
                     self.add_yolo(module, block, i, bottom)
            else:
                 self.top_names[i] = bottom
                 
    def save(self, prototxt_path, caffemodel_path):
        """
        Serialized to disk.
        @param prototxt_path .prototxt path.
        @param caffemodel_path .caffemodel path.
        """
        with open(prototxt_path, 'w') as f:
            from google.protobuf import text_format
            
            net_proto = pb.NetParameter()
            net_proto.CopyFrom(self.net_msg)
            for layer in net_proto.layer:
                del layer.blobs[:]
                
            f.write(text_format.MessageToString(net_proto))
            
        with open(caffemodel_path, 'wb') as f:
            f.write(self.net_msg.SerializeToString())
            
        log_success(f"Caffe files: {prototxt_path}, {caffemodel_path}")

def export_pytorch_to_caffe(model, input_shape, output_path):
    """
    Export wrapper.
    @param model Model instance.
    @param input_shape Data shape.
    @param output_path Resulting path.
    @return True if success.
    """
    log_info("Starting Caffe export...")
    
    if not hasattr(model, 'blocks') or not hasattr(model, 'models'):
        log_error("Only DarknetParser is supported for Caffe export.")
        raise TypeError("DarknetParser instance required.")
        
    if pb is None:
        log_error("caffe_pb2 module missing.")
        raise ImportError("caffe_pb2 not found")
        
    exporter = CaffeExporter(model, input_shape)
    exporter.build()
    
    base_name = output_path
    if base_name.endswith('.caffemodel'):
        base_name = base_name[:-11]
    
    prototxt = base_name + ".prototxt"
    caffemodel = base_name + ".caffemodel"
    
    exporter.save(prototxt, caffemodel)
    return True
