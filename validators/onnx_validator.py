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

def _flatten_to_numpy(obj):
    """Recursively flatten nested dict/list/tuple of tensors to list of numpy arrays."""
    results = []
    if isinstance(obj, dict):
        for v in obj.values():
            results.extend(_flatten_to_numpy(v))
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            results.extend(_flatten_to_numpy(item))
    elif hasattr(obj, 'cpu'):
        results.append(obj.cpu().numpy())
    return results

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
    
    if hasattr(pt_model, 'parameters') and any(True for _ in pt_model.parameters()):
        device = next(pt_model.parameters()).device
    elif hasattr(pt_model, 'device'):
        device = pt_model.device
    else:
        device = torch.device('cpu')
        
    dummy_input = torch.randn(*input_shape).float().to(device)
    
    pt_model.eval()
    with torch.no_grad():
        pt_out = pt_model(dummy_input)
    
    if isinstance(pt_out, dict):
        # Newer Ultralytics models (v10/v11/v26) return dicts
        pt_out = _flatten_to_numpy(pt_out)
    elif isinstance(pt_out, (list, tuple)):
        pt_out = _flatten_to_numpy(pt_out)
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
    
    log_info(f"Comparing {len(pt_out)} PyTorch vs {len(onnx_out)} ONNX outputs.")
    
    # When output counts differ (e.g. PyTorch returns post-processed + raw),
    # match by shape to find corresponding outputs
    if len(pt_out) != len(onnx_out):
        log_warning(f"Output count mismatch ({len(pt_out)} vs {len(onnx_out)}). Matching by shape...")
    
    matched = 0
    failed = 0
    
    for oi, o in enumerate(onnx_out):
        best_mae = float('inf')
        best_idx = -1
        matched_by_shape = False

        # Pass 1: Exact shape match
        for pi, p in enumerate(pt_out):
            if p.shape == o.shape:
                mae = np.mean(np.abs(p - o))
                if mae < best_mae:
                    best_mae = mae
                    best_idx = pi
                    matched_by_shape = True

        # Pass 2: Fallback to size (reshape) if no exact shape was good enough
        if best_idx == -1 or best_mae > tolerance:
            for pi, p in enumerate(pt_out):
                if p.size == o.size and p.shape != o.shape:
                    candidate = o.reshape(p.shape)
                    mae = np.mean(np.abs(p - candidate))
                    if mae < best_mae:
                        best_mae = mae
                        best_idx = pi
        
        if best_idx >= 0:
            if np.isnan(best_mae) or best_mae > tolerance:
                log_warning(f"ONNX output {oi} best match MAE: {best_mae:.6f} (above tolerance)")
                failed += 1
            else:
                log_success(f"ONNX output {oi} validated (MAE: {best_mae:.6f}).")
                matched += 1
        else:
            log_warning(f"ONNX output {oi} has no shape-matching PyTorch output. Skipped.")
    
    if matched == 0:
        log_error("No outputs matched between PyTorch and ONNX.")
        return False
    
    log_info(f"Matched {matched}/{len(onnx_out)} outputs, {failed} above tolerance.")
    return failed == 0