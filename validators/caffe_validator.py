# --------------------------------------------------------------------------
# caffe_validator.py
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
import cv2
import torch
import numpy as np
from core.logger import log_info, log_warning, log_error, log_success

def validate_caffe(pt_model, prototxt_path, caffemodel_path, input_shape, tolerance=1e-2):
    """
    Compares PyTorch vs Caffe numerical outputs.
    @param pt_model PyTorch source.
    @param prototxt_path Caffe graph.
    @param caffemodel_path Caffe weights.
    @param input_shape Data shape.
    @param tolerance Error threshold.
    @return True if MAE < tolerance.
    """
    log_info("Starting Caffe validation...")
    
    device = next(pt_model.parameters()).device if hasattr(pt_model, 'parameters') else torch.device('cpu')
    dummy_input = torch.randn(*input_shape).float().to(device)
    
    pt_model.eval()
    with torch.no_grad():
        pt_out = pt_model(dummy_input)
    
    if isinstance(pt_out, (list, tuple)):
        pt_out = [x.cpu().numpy() for x in pt_out]
    else:
        pt_out = [pt_out.cpu().numpy()]

    if not os.path.exists(prototxt_path) or not os.path.exists(caffemodel_path):
        log_error("Files missing.")
        return False

    try:
        net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        blob = dummy_input.cpu().numpy()
        net.setInput(blob)
        
        out_names = net.getUnconnectedOutLayersNames()
        caffe_out = net.forward(out_names)
        
        if isinstance(caffe_out, tuple):
            caffe_out = list(caffe_out)
        elif not isinstance(caffe_out, list):
            caffe_out = [caffe_out]

    except Exception as e:
        log_error(f"Inference Engine failed: {e}")
        return False

    log_info(f"Comparing {len(pt_out)} predicted vs {len(caffe_out)} extracted.")
    
    all_pass = True
    min_len = min(len(pt_out), len(caffe_out))
    
    for i in range(min_len):
        p, c = pt_out[i], caffe_out[i]
        
        if p.shape != c.shape:
             if p.size == c.size:
                 c = c.reshape(p.shape)
             else:
                 log_error(f"Output {i} size mismatch.")
                 all_pass = False
                 continue
            
        mae = np.mean(np.abs(p - c))
        if np.isnan(mae) or mae > tolerance:
            log_warning(f"Output {i} difference: {mae:.6f}")
            all_pass = False
        else:
            log_success(f"Output {i} matches (MAE: {mae:.6f}).")
            
    return all_pass
