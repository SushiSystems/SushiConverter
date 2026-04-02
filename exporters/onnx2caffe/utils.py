# --------------------------------------------------------------------------
# utils.py
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
from onnx import numpy_helper
import core.caffe_pb2 as pb
from core.logger import log_warning

def extract_weight(node_input_name, initializers, constants=None):
    """
    Extracts a numpy weight array from the ONNX initializers list or constants map.
    Returns None if the weight is not found.
    """
    if constants and node_input_name in constants:
        return constants[node_input_name]
        
    for init in initializers:
        if init.name == node_input_name:
            return numpy_helper.to_array(init)
    return None

def numpy_to_caffe_blob(np_array):
    """
    Converts a numpy array into a Caffe BlobProto.
    """
    blob = pb.BlobProto()
    blob.shape.dim.extend(list(np_array.shape))
    blob.data.extend(np_array.astype(np.float32).flatten())
    return blob

def get_node_attributes(node):
    """
    Parses ONNX node attributes into a Python dictionary.
    """
    kwargs = {}
    for attr in node.attribute:
        if attr.type == 1:
            kwargs[attr.name] = attr.f
        elif attr.type == 2:
            kwargs[attr.name] = attr.i
        elif attr.type == 3:
            kwargs[attr.name] = attr.s.decode('utf-8')
        elif attr.type == 4:
            kwargs[attr.name] = attr.t
        elif attr.type == 5:
            kwargs[attr.name] = attr.g
        elif attr.type == 6:
            kwargs[attr.name] = list(attr.floats)
        elif attr.type == 7:
            kwargs[attr.name] = list(attr.ints)
        elif attr.type == 8:
            kwargs[attr.name] = [s.decode('utf-8') for s in attr.strings]
        elif attr.type == 9:
            kwargs[attr.name] = list(attr.tensors)
        elif attr.type == 10:
            kwargs[attr.name] = list(attr.graphs)
    return kwargs

def determine_padding(pads):
    """
    Converts ONNX pads [y1, x1, y2, x2] to Caffe pad/pad_h/pad_w.
    Supports asymmetric padding warnings.
    Returns: (pad_h, pad_w)
    """
    if not pads or len(pads) == 0:
        return 0, 0
    
    if len(pads) == 4:
        # ONNX format: top, left, bottom, right [y1, x1, y2, x2]
        if pads[0] != pads[2] or pads[1] != pads[3]:
            log_warning(f"Asymmetric padding detected: {pads}. Caffe layers support symmetric padding by default. Converting to max padding...")
        
        pad_y = max(pads[0], pads[2])
        pad_x = max(pads[1], pads[3])
        return pad_y, pad_x
    
    elif len(pads) == 2:
        return pads[0], pads[1]
    
    return pads[0], pads[0]

class OnnxModelWrapper:
    """
    Wraps an ONNX model to provide a PyTorch-like __call__ interface.
    Used for validation comparison.
    """
    def __init__(self, onnx_path):
        import onnxruntime as ort
        self.onnx_path = onnx_path
        self.session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name

    def __call__(self, x):
        """
        Runs inference.
        @param x torch.Tensor or numpy array
        @return list of torch Tensors
        """
        import torch
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        
        onnx_out = self.session.run(None, {self.input_name: x})
        return [torch.from_numpy(out) for out in onnx_out]

    def eval(self):
        pass

    @property
    def device(self):
        import torch
        return torch.device('cpu')

