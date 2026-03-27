# --------------------------------------------------------------------------
# layers.py
# --------------------------------------------------------------------------
# This file is part of:
# SushiConverter
# https://github.com/SushiSystems/SushiConverter
# https://sushisystems.io
# --------------------------------------------------------------------------

import core.caffe_pb2 as pb
from core.logger import log_warning, log_info, log_error
from exporters.onnx2caffe.utils import extract_weight, numpy_to_caffe_blob, get_node_attributes, determine_padding

class LayerMapper:
    """
    Registry for ONNX to Caffe node maps.
    """
    _registry = {}

    @classmethod
    def register(cls, op_type):
        def decorator(func):
            cls._registry[op_type] = func
            return func
        return decorator

    @classmethod
    def map_node(cls, node, graph, top_names):
        """
        Dispatches node to the appropriate mapper.
        """
        if node.op_type in cls._registry:
            return cls._registry[node.op_type](node, graph, top_names)
        else:
            log_error(f"Unsupported ONNX operation: {node.op_type}. Ignoring node {node.name}.")
            return []

# --- Core Mappers ---

@LayerMapper.register('Conv')
def map_conv(node, graph, top_names):
    initializers = graph.initializer
    attrs = get_node_attributes(node)
    
    layer = pb.LayerParameter()
    layer.name = node.name or node.output[0]
    layer.type = "Convolution"
    layer.bottom.extend([top_names.get(node.input[0], node.input[0])])
    layer.top.extend([node.output[0]])
    
    conv_param = layer.convolution_param
    
    # Process Weights
    weight_np = extract_weight(node.input[1], initializers)
    if weight_np is not None:
        conv_param.num_output = weight_np.shape[0]
        layer.blobs.extend([numpy_to_caffe_blob(weight_np)])
    
    bias_term = len(node.input) > 2
    conv_param.bias_term = bias_term
    if bias_term:
        bias_np = extract_weight(node.input[2], initializers)
        if bias_np is not None:
            layer.blobs.extend([numpy_to_caffe_blob(bias_np)])

    if 'group' in attrs and attrs['group'] > 1:
        conv_param.group = attrs['group']

    if 'kernel_shape' in attrs:
        conv_param.kernel_size.extend(attrs['kernel_shape'])
    
    if 'strides' in attrs:
        conv_param.stride.extend(attrs['strides'])
        
    if 'dilations' in attrs:
        conv_param.dilation.extend(attrs['dilations'])

    if 'pads' in attrs:
        pad_y, pad_x = determine_padding(attrs['pads'])
        if pad_y == pad_x:
            conv_param.pad.append(pad_y)
        else:
            conv_param.pad_h = pad_y
            conv_param.pad_w = pad_x

    return [layer]


@LayerMapper.register('BatchNormalization')
def map_batch_norm(node, graph, top_names):
    initializers = graph.initializer
    attrs = get_node_attributes(node)
    layers = []
    
    name = node.name or node.output[0]
    bottom_name = top_names.get(node.input[0], node.input[0])
    
    # 1. Caffe BatchNorm
    bn_layer = pb.LayerParameter()
    bn_layer.name = name + "_bn"
    bn_layer.type = "BatchNorm"
    bn_layer.bottom.extend([bottom_name])
    bn_layer.top.extend([bn_layer.name])
    
    bn_layer.batch_norm_param.use_global_stats = True
    bn_layer.batch_norm_param.eps = attrs.get('epsilon', 1e-5)
    
    mean = extract_weight(node.input[3], initializers)
    var = extract_weight(node.input[4], initializers)
    
    if mean is not None and var is not None:
        bn_layer.blobs.extend([
            numpy_to_caffe_blob(mean),
            numpy_to_caffe_blob(var),
            numpy_to_caffe_blob(np.array([1.0], dtype=np.float32)) # scaling factor
        ])
    layers.append(bn_layer)
    
    # 2. Caffe Scale
    scale_layer = pb.LayerParameter()
    scale_layer.name = name + "_scale"
    scale_layer.type = "Scale"
    scale_layer.bottom.extend([bn_layer.name])
    scale_layer.top.extend([node.output[0]])
    scale_layer.scale_param.bias_term = True
    
    gamma = extract_weight(node.input[1], initializers)
    beta = extract_weight(node.input[2], initializers)
    
    if gamma is not None and beta is not None:
        scale_layer.blobs.extend([
            numpy_to_caffe_blob(gamma),
            numpy_to_caffe_blob(beta)
        ])
    layers.append(scale_layer)
    
    return layers


@LayerMapper.register('Relu')
def map_relu(node, graph, top_names):
    layer = pb.LayerParameter()
    layer.name = node.name or node.output[0]
    layer.type = "ReLU"
    layer.bottom.extend([top_names.get(node.input[0], node.input[0])])
    layer.top.extend([node.output[0]])
    return [layer]


@LayerMapper.register('LeakyRelu')
def map_leaky_relu(node, graph, top_names):
    attrs = get_node_attributes(node)
    layer = pb.LayerParameter()
    layer.name = node.name or node.output[0]
    layer.type = "ReLU"
    layer.bottom.extend([top_names.get(node.input[0], node.input[0])])
    layer.top.extend([node.output[0]])
    layer.relu_param.negative_slope = attrs.get('alpha', 0.1)
    return [layer]


@LayerMapper.register('MaxPool')
def map_max_pool(node, graph, top_names):
    attrs = get_node_attributes(node)
    layer = pb.LayerParameter()
    layer.name = node.name or node.output[0]
    layer.type = "Pooling"
    layer.bottom.extend([top_names.get(node.input[0], node.input[0])])
    layer.top.extend([node.output[0]])
    
    layer.pooling_param.pool = pb.PoolingParameter.MAX
    if 'kernel_shape' in attrs:
        layer.pooling_param.kernel_size = attrs['kernel_shape'][0]
    if 'strides' in attrs:
        layer.pooling_param.stride = attrs['strides'][0]
    if 'pads' in attrs:
        layer.pooling_param.pad = attrs['pads'][0]
        
    return [layer]


@LayerMapper.register('AveragePool')
@LayerMapper.register('GlobalAveragePool')
def map_average_pool(node, graph, top_names):
    attrs = get_node_attributes(node)
    layer = pb.LayerParameter()
    layer.name = node.name or node.output[0]
    layer.type = "Pooling"
    layer.bottom.extend([top_names.get(node.input[0], node.input[0])])
    layer.top.extend([node.output[0]])
    
    layer.pooling_param.pool = pb.PoolingParameter.AVE
    
    if node.op_type == 'GlobalAveragePool':
        layer.pooling_param.global_pooling = True
    else:
        if 'kernel_shape' in attrs:
            layer.pooling_param.kernel_size = attrs['kernel_shape'][0]
        if 'strides' in attrs:
            layer.pooling_param.stride = attrs['strides'][0]
        if 'pads' in attrs:
            layer.pooling_param.pad = attrs['pads'][0]
            
    return [layer]


@LayerMapper.register('Concat')
def map_concat(node, graph, top_names):
    attrs = get_node_attributes(node)
    layer = pb.LayerParameter()
    layer.name = node.name or node.output[0]
    layer.type = "Concat"
    
    for inp in node.input:
        layer.bottom.extend([top_names.get(inp, inp)])
        
    layer.top.extend([node.output[0]])
    
    if 'axis' in attrs:
        # ONNX uses 1 for channel, Caffe default is 1
        layer.concat_param.axis = attrs['axis']
        
    return [layer]


@LayerMapper.register('Add')
def map_add(node, graph, top_names):
    layer = pb.LayerParameter()
    layer.name = node.name or node.output[0]
    layer.type = "Eltwise"
    layer.eltwise_param.operation = pb.EltwiseParameter.SUM
    
    for inp in node.input:
        layer.bottom.extend([top_names.get(inp, inp)])
        
    layer.top.extend([node.output[0]])
    return [layer]

@LayerMapper.register('Sub')
def map_sub(node, graph, top_names):
    # Caffe Eltwise does not natively do SUB, we use SUM with a -1 coeff or it's NPU specific
    layer = pb.LayerParameter()
    layer.name = node.name or node.output[0]
    layer.type = "Eltwise"
    layer.eltwise_param.operation = pb.EltwiseParameter.SUM
    layer.eltwise_param.coeff.extend([1.0, -1.0])
    for inp in node.input:
        layer.bottom.extend([top_names.get(inp, inp)])
    layer.top.extend([node.output[0]])
    return [layer]

@LayerMapper.register('Mul')
def map_mul(node, graph, top_names):
    layer = pb.LayerParameter()
    layer.name = node.name or node.output[0]
    layer.type = "Eltwise"
    layer.eltwise_param.operation = pb.EltwiseParameter.PROD
    for inp in node.input:
        layer.bottom.extend([top_names.get(inp, inp)])
    layer.top.extend([node.output[0]])
    return [layer]

@LayerMapper.register('Sigmoid')
def map_sigmoid(node, graph, top_names):
    layer = pb.LayerParameter()
    layer.name = node.name or node.output[0]
    layer.type = "Sigmoid"
    layer.bottom.extend([top_names.get(node.input[0], node.input[0])])
    layer.top.extend([node.output[0]])
    return [layer]

@LayerMapper.register('Resize')
@LayerMapper.register('Upsample')
def map_resize(node, graph, top_names):
    layer = pb.LayerParameter()
    layer.name = node.name or node.output[0]
    layer.type = "Deconvolution"
    layer.bottom.extend([top_names.get(node.input[0], node.input[0])])
    layer.top.extend([node.output[0]])
    
    scale_factor = 2
    if len(node.input) > 2:
        scales = extract_weight(node.input[2], graph.initializer)
        if scales is not None and len(scales) >= 3:
            scale_factor = int(scales[2])
            
    # Extract channels from ValueInfo using shape inference results
    channels = 1
    input_name = node.input[0]
    for info in graph.value_info:
        if info.name == input_name:
            if len(info.type.tensor_type.shape.dim) >= 2:
                channels = info.type.tensor_type.shape.dim[1].dim_value
                if channels <= 0: channels = 1
            break
            
    conv_param = layer.convolution_param
    conv_param.num_output = channels
    conv_param.kernel_size.append(scale_factor)
    conv_param.stride.append(scale_factor)
    conv_param.pad.append(0)
    conv_param.bias_term = False
    conv_param.group = channels
    
    weight = np.ones((channels, 1, scale_factor, scale_factor), dtype=np.float32)
    layer.blobs.extend([numpy_to_caffe_blob(weight)])
    
    return [layer]

@LayerMapper.register('Slice')
def map_slice(node, graph, top_names):
    layer = pb.LayerParameter()
    layer.name = node.name or node.output[0]
    layer.type = "Slice"
    layer.bottom.extend([top_names.get(node.input[0], node.input[0])])
    layer.top.extend([node.output[0]])
    return [layer]

@LayerMapper.register('Split')
def map_split(node, graph, top_names):
    layer = pb.LayerParameter()
    layer.name = node.name or node.output[0]
    layer.type = "Slice"
    layer.bottom.extend([top_names.get(node.input[0], node.input[0])])
    for out in node.output:
        layer.top.extend([out])
    return [layer]

import numpy as np
