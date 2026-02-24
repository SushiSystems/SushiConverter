# --------------------------------------------------------------------------
# export_ultralytics.py
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
import torch
import torch.nn as nn
from core.logger import log_info, log_warning, log_error, log_success

class RawNPUHead(nn.Module):
    """
    NPU optimized detection head for Ultralytics models.
    """
    def __init__(self, original_layer):
        super().__init__()
        self.nl = original_layer.nl
        self.cv2 = original_layer.cv2

    def forward(self, x):
        res = []
        for i in range(self.nl):
            feat = self.cv2[i](x[i])
            res.append(feat)
        return res

def is_ultralytics_model(model):
    """
    Identifies if model is from Ultralytics.
    @param model model object.
    @return tuple (is_ultra, layers).
    """
    model_layers = None
    if hasattr(model, 'model') and isinstance(model.model, nn.Sequential):
        model_layers = model.model
    elif isinstance(model, nn.Sequential):
        model_layers = model
    
    if model_layers is None:
        return False, None

    try:
        last_layer = model_layers[-1]
        if not (hasattr(last_layer, 'cv2') and hasattr(last_layer, 'nl')):
            return False, None
    except:
        return False, None
    
    return True, model_layers

def export_ultralytics_to_onnx(model, input_shape, output_path):
    """
    Optimized export for YOLOv5/v8.
    @param model Ultralytics model.
    @param input_shape input dimensions.
    @param output_path output location.
    @return True if success.
    """
    OPSET_VERSION = 11

    is_ultra, model_layers = is_ultralytics_model(model)
    if not is_ultra:
        return False

    log_info("Applying NPU optimization for Ultralytics head...")

    try:
        last_layer = model_layers[-1]
        new_layer = RawNPUHead(last_layer)
        for attr in ['i', 'f', 'type']:
            if hasattr(last_layer, attr):
                setattr(new_layer, attr, getattr(last_layer, attr))
        
        model_layers[-1] = new_layer
        log_info("Detect layer replaced.")
        
    except Exception as e:
        log_warning(f"NPU transformation failed: {e}")
        return False

    try:
        device = next(model.parameters()).device
        dummy_input = torch.randn(*input_shape).to(device)
        model.eval()

        log_info(f"Exporting to ONNX (v{OPSET_VERSION})...")
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            opset_version=OPSET_VERSION,
            input_names=['images'],
            output_names=['output0'],
            do_constant_folding=True
        )

        onnx_model = onnx.load(output_path)
        onnx.save_model(onnx_model, output_path, save_as_external_data=False)
        
        data_file = output_path + ".data"
        if os.path.exists(data_file):
            os.remove(data_file)

        log_success("Ultralytics export PASSED.")
        return True

    except Exception as e:
        log_error(f"Export error: {e}")
        raise e
