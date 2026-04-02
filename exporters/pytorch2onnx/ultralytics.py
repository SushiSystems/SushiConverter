# --------------------------------------------------------------------------
# ultralytics.py
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
        if x is None:
            from core.logger import log_error
            log_error("RawNPUHead: Received x=None as input!")
            return []
        
        # If x is not a list/tuple (e.g. YOLOv10 might behave differently)
        if not isinstance(x, (list, tuple)):
            # Log it but try to wrap it
            from core.logger import log_warning
            log_warning(f"RawNPUHead: Expected list input, got {type(x)}")
            x = [x]
        
        res = []
        for i in range(self.nl):
            if i < len(x):
                feat = self.cv2[i](x[i])
                res.append(feat)
            else:
                # Padding or different head structure?
                pass
        return res

def is_ultralytics_model(model):
    """
    Identifies if model is from Ultralytics.
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
    Optimized export for Ultralytics models (YOLOv5/v8/v10/v11/v26+).
    Attempts NPU-optimized head replacement first; falls back to
    standard export if the model architecture is incompatible.
    """
    OPSET_VERSION = 11

    is_ultra, model_layers = is_ultralytics_model(model)
    if not is_ultra:
        return False

    device = next(model.parameters()).device
    dummy_input = torch.randn(*input_shape).to(device)
    model.eval()

    # --- Attempt 1: NPU-optimized export (raw detection head) ---
    original_layer = model_layers[-1]
    npu_success = False

    try:
        log_info("Applying NPU optimization for Ultralytics head...")
        new_layer = RawNPUHead(original_layer)
        for attr in ['i', 'f', 'type']:
            if hasattr(original_layer, attr):
                setattr(new_layer, attr, getattr(original_layer, attr))

        model_layers[-1] = new_layer
        log_info("Detect layer replaced.")

        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            opset_version=OPSET_VERSION,
            input_names=['images'],
            output_names=['output0'],
            do_constant_folding=True
        )
        npu_success = True
        log_success("NPU-optimized Ultralytics export PASSED.")

    except Exception as e:
        log_warning(f"NPU head export failed ({e}). Restoring original head...")
        # Restore original detect layer
        model_layers[-1] = original_layer

    # --- Attempt 2: Standard export (full detect head) ---
    if not npu_success:
        try:
            log_info("Falling back to standard Ultralytics ONNX export...")
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                opset_version=OPSET_VERSION,
                input_names=['images'],
                output_names=['output0'],
                do_constant_folding=True
            )
            log_success("Standard Ultralytics export PASSED.")

        except Exception as e:
            log_error(f"Standard export also failed: {e}")
            raise e

    # Clean up external data artifacts
    try:
        onnx_model = onnx.load(output_path)
        onnx.save_model(onnx_model, output_path, save_as_external_data=False)

        data_file = output_path + ".data"
        if os.path.exists(data_file):
            os.remove(data_file)
    except Exception:
        pass

    return True
