# --------------------------------------------------------------------------
# onnx_engine.py
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
import numpy as np
from core.logger import log_info, log_warning, log_error, log_success

def run_onnx_inference(onnx_path, input_shape, source_mode='darknet'):
    """
    Simulates production inference using OpenCV DNN.
    @param onnx_path path to .onnx.
    @param input_shape B-C-H-W.
    @param source_mode input origin.
    @return True if execution was valid.
    """
    log_info("Initializing OpenCV DNN Engine...")
    
    batch, channels, height, width = input_shape
    
    if not os.path.exists(onnx_path):
        log_error(f"Missing file: {onnx_path}")
        return False

    try:
        import onnx
        onnx_model = onnx.load(onnx_path)
        unsupported_ops = set()
        for node in onnx_model.graph.node:
            if node.op_type in ['ScatterND', 'Gather', 'GatherElements']:
                unsupported_ops.add(node.op_type)
        
        if unsupported_ops:
            ops_str = ", ".join(sorted(list(unsupported_ops)))
            log_error(f"Unsupported ops found: {ops_str}")
            return False

        net = cv2.dnn.readNetFromONNX(onnx_path)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        log_warning("Simulating data for inference test...")
        img_blob = np.random.randint(0, 255, (height, width, channels), dtype=np.uint8)
        blob = cv2.dnn.blobFromImage(img_blob, 1/255.0, (width, height), swapRB=True, crop=False)
        
        net.setInput(blob)
        out_names = net.getUnconnectedOutLayersNames()
        outputs = net.forward(out_names)
        
        log_info(f"Engine produced {len(outputs)} tensors.")
        
        all_ok = True
        for i, out in enumerate(outputs):
            max_val = np.max(out)
            log_info(f"Layer {i} -> Shape: {out.shape}, Max: {max_val:.6f}")
            if np.isnan(max_val) or np.isinf(max_val):
                log_error(f"Invalid values in output {i}.")
                all_ok = False
        
        return all_ok

    except Exception as e:
        log_error(f"Inference failed: {e}")
        return False
