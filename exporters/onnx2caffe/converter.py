# --------------------------------------------------------------------------
# converter.py
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
import onnx
import core.caffe_pb2 as pb
from core.logger import log_info, log_error, log_success
from exporters.onnx2caffe.layers import LayerMapper

class OnnxToCaffeConverter:
    """
    Orchestrates the conversion from ONNX to Caffe.
    """
    def __init__(self, onnx_model_path, shape=None):
        self.onnx_model_path = onnx_model_path
        self.shape = shape
        model_raw = onnx.load(self.onnx_model_path)
        self.model = onnx.shape_inference.infer_shapes(model_raw)
        self.graph = self.model.graph
        
        self.net_msg = pb.NetParameter()
        self.net_msg.name = self.graph.name or "ONNX2Caffe"
        self.top_names = {}
        self.constants = {}

    def _add_input(self):
        """
        Creates Caffe input layer from ONNX graph input.
        """
        if len(self.graph.input) == 0:
            log_error("No input found in ONNX graph.")
            raise ValueError("Invalid ONNX graph.")
            
        graph_input = self.graph.input[0]
        layer = self.net_msg.layer.add()
        layer.name = graph_input.name
        layer.type = "Input"
        layer.top.append(graph_input.name)
        
        # Use provided shape or extract from graph
        shape_dim = self.shape
        if shape_dim is None:
            shape_dim = []
            for d in graph_input.type.tensor_type.shape.dim:
                val = d.dim_value
                if val <= 0: val = 1
                shape_dim.append(val)
                
        shape = layer.input_param.shape.add()
        shape.dim.extend(shape_dim)
        
        self.top_names[graph_input.name] = graph_input.name
        log_info(f"Input '{graph_input.name}' created with shape {shape_dim}")

    def build(self):
        """
        Iterates over the ONNX graph and converts it to Caffe.
        """
        log_info("Starting ONNX to Caffe conversion...")
        self._add_input()
        
        # 1. Pre-scan for Constant nodes to populate constants map
        from onnx import numpy_helper
        for node in self.graph.node:
            if node.op_type == 'Constant':
                for attr in node.attribute:
                    if attr.name == 'value':
                        self.constants[node.output[0]] = numpy_helper.to_array(attr.t)
        
        # 2. Traverse nodes for conversion
        for node in self.graph.node:
            caffe_layers = LayerMapper.map_node(node, self.graph, self.top_names, self.constants)
            
            for layer in caffe_layers:
                self.net_msg.layer.append(layer)
                
            # ONNX node outputs -> Caffe layer tops
            if len(caffe_layers) > 0:
                last_layer = caffe_layers[-1]
                all_tops = set()
                for cl in caffe_layers:
                    all_tops.update(cl.top)
                    
                for out in node.output:
                    if out in all_tops:
                        # Direct match: the mapper already used the ONNX output name
                        self.top_names[out] = out
                    elif len(node.output) == 1 and len(last_layer.top) >= 1:
                        # Single-output node: map to first top of last layer
                        self.top_names[out] = last_layer.top[0]
                    else:
                        # Multi-output: positional fallback
                        idx = list(node.output).index(out)
                        if idx < len(last_layer.top):
                            self.top_names[out] = last_layer.top[idx]
                        else:
                            self.top_names[out] = last_layer.top[0]

        log_success("Graph converted in memory.")

    def save(self, prototxt_path, caffemodel_path):
        """
        Serializes the NetParameter to disk.
        """
        with open(prototxt_path, 'w') as f:
            from google.protobuf import text_format
            # Make a copy without blobs for prototxt
            net_proto = pb.NetParameter()
            net_proto.CopyFrom(self.net_msg)
            for layer in net_proto.layer:
                del layer.blobs[:]
            f.write(text_format.MessageToString(net_proto))
            
        with open(caffemodel_path, 'wb') as f:
            f.write(self.net_msg.SerializeToString())
            
        log_success(f"Caffe files saved: {prototxt_path}, {caffemodel_path}")

def convert_onnx_to_caffe(onnx_model_path, output_path, shape=None):
    """
    Main entry point for ONNX to Caffe export.
    """
    converter = OnnxToCaffeConverter(onnx_model_path, shape)
    converter.build()
    
    base_name = output_path
    if base_name.endswith('.caffemodel'):
        base_name = base_name[:-11]
    
    prototxt = base_name + ".prototxt"
    caffemodel = base_name + ".caffemodel"
    
    converter.save(prototxt, caffemodel)
    return True
