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
from google.protobuf import text_format
import core.caffe_pb2 as pb
from core.logger import log_info, log_error, log_success
from .layers import LayerMapper

class DarknetToCaffeConverter:
    """
    Exports Darknet PyTorch networks to Caffe models.
    """
    def __init__(self, net, shape):
        self.net = net
        self.shape = shape
        self.net_msg = pb.NetParameter()
        self.net_msg.name = "Darknet2Caffe"
        self.top_names = dict()

    def add_input(self):
        layer = self.net_msg.layer.add()
        layer.name = "data"
        layer.type = "Input"
        layer.top.append("data")
        shape = layer.input_param.shape.add()
        shape.dim.extend(self.shape)
        self.top_names[-1] = "data"

    def build(self):
        log_info("Converting Darknet architecture to Caffe Protobuf...")
        self.add_input()

        for i, (module, block) in enumerate(zip(self.net.models, self.net.blocks[1:])):
            bottom = self.top_names[i - 1]
            LayerMapper.map_block(module, block, i, bottom, self.net_msg, self.top_names, self.net)

    def save(self, output_path):
        base_name = output_path
        if base_name.endswith('.caffemodel'):
            base_name = base_name[:-11]
        
        prototxt_path = base_name + ".prototxt"
        caffemodel_path = base_name + ".caffemodel"

        # Save Prototxt (without weights)
        with open(prototxt_path, 'w') as f:
            net_proto = pb.NetParameter()
            net_proto.CopyFrom(self.net_msg)
            for layer in net_proto.layer:
                del layer.blobs[:]
            f.write(text_format.MessageToString(net_proto))
            
        # Save Caffemodel (with weights)
        with open(caffemodel_path, 'wb') as f:
            f.write(self.net_msg.SerializeToString())
            
        log_success(f"Caffe files saved: {prototxt_path}, {caffemodel_path}")

def export_darknet_to_caffe(model, input_shape, output_path):
    """
    Export wrapper for ExportDispatcher.
    """
    log_info("Starting Darknet to Caffe export...")
    converter = DarknetToCaffeConverter(model, input_shape)
    converter.build()
    converter.save(output_path)
    return True
