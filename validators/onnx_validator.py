# --------------------------------------------------------------------------
# onnx_validator.py
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

import torch
import numpy as np
import onnxruntime as ort
from core.logger import log_info, log_warning, log_error, log_success

def validate_onnx(pt_model, onnx_path, input_shape, tolerance=1e-4):
    """
    Compares PyTorch vs ONNXRuntime.
    @param pt_model source model.
    @param onnx_path exported path.
    @param input_shape data shape.
    @param tolerance threshold.
    @return True if MAE < tolerance.
    """
    log_info("Starting ONNX numerical validation...")
    
    device = next(pt_model.parameters()).device if hasattr(pt_model, 'parameters') else torch.device('cpu')
    dummy_input = torch.randn(*input_shape).float().to(device)
    
    pt_model.eval()
    with torch.no_grad():
        pt_out = pt_model(dummy_input)
    
    if isinstance(pt_out, (list, tuple)):
        pt_out = [x.cpu().numpy() for x in pt_out]
    else:
        pt_out = [pt_out.cpu().numpy()]
        
    try:
        session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    except Exception as e:
        log_error(f"Inference session failed: {e}")
        return False

    input_name = session.get_inputs()[0].name
    try:
        onnx_out = session.run(None, {input_name: dummy_input.cpu().numpy()})
    except Exception as e:
        log_error(f"Execution failed: {e}")
        return False
    
    log_info(f"Comparing {len(pt_out)} predicted vs {len(onnx_out)} exported.")
    
    all_pass = True
    for i, (p, o) in enumerate(zip(pt_out, onnx_out)):
        if p.shape != o.shape:
            log_error(f"Output {i} shape mismatch.")
            all_pass = False
            continue
            
        mae = np.mean(np.abs(p - o))
        if np.isnan(mae) or mae > tolerance:
            log_warning(f"Output {i} difference: {mae:.6f}")
            all_pass = False
        else:
            log_success(f"Output {i} validated (MAE: {mae:.6f}).")
            
    return all_pass
