# --------------------------------------------------------------------------
# source_validator.py
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
import sys
import torch
import numpy as np
import importlib.util
from core.logger import log_info, log_error, log_success

def validate_standalone_source(pt_model, py_path, input_shape, tolerance=1e-5):
    """
    Validates dynamic source module vs original.
    @param pt_model root model.
    @param py_path standalone source path.
    @param input_shape data shape.
    @param tolerance error limit.
    @return True if MAE < tolerance.
    """
    log_info(f"Checking Standalone -> {os.path.basename(py_path)}")
    
    pth_path = os.path.splitext(py_path)[0] + '.pth'
    if not os.path.exists(py_path) or not os.path.exists(pth_path):
        log_error("Source or weight files missing.")
        return False

    try:
        module_name = "standalone_validator_mod"
        spec = importlib.util.spec_from_file_location(module_name, py_path)
        standalone_mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = standalone_mod
        spec.loader.exec_module(standalone_mod)
        
        no_yolo_layer = getattr(pt_model, 'no_yolo_layer', True)
        standalone_model = standalone_mod.load_model(pth_path, no_yolo_layer=no_yolo_layer)
        standalone_model.eval()
        
    except Exception as e:
        log_error(f"Module load failed: {e}")
        return False

    device = next(pt_model.parameters()).device if hasattr(pt_model, 'parameters') else torch.device('cpu')
    dummy_input = torch.randn(*input_shape).float().to(device)
    
    pt_model.eval()
    with torch.no_grad():
        original_out = pt_model(dummy_input)
        standalone_out = standalone_model(dummy_input.to('cpu'))

    def to_numpy(x):
        if isinstance(x, (list, tuple)):
            return [v.detach().cpu().numpy() for v in x]
        return [x.detach().cpu().numpy()]

    orig_np = to_numpy(original_out)
    std_np = to_numpy(standalone_out)

    if len(orig_np) != len(std_np):
        log_error("Tensor count mismatch.")
        return False

    all_pass = True
    for i, (o, s) in enumerate(zip(orig_np, std_np)):
        if o.shape != s.shape:
            log_error(f"Shape mismatch at output {i}.")
            all_pass = False
            continue
            
        mae = np.mean(np.abs(o - s))
        if mae > tolerance:
            log_error(f"Deviation found: {mae:.8f}")
            all_pass = False
        else:
            log_success(f"Output {i} (Shape: {tuple(s.shape)}) matches (MAE: {mae:.8f})")

    return all_pass
