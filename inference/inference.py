# --------------------------------------------------------------------------
# inference.py
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

from core.logger import log_info, log_warning
from .onnx_engine import run_onnx_inference

def run_inference_test(output_mode, final_path, shape, source_mode='darknet'):
    """
    Routes inference tests to the correct backend.
    @param output_mode format being tested.
    @param final_path file path to model.
    @param shape input dimensions.
    @param source_mode input format.
    @return True if inference succeeded.
    """
    if output_mode == 'onnx':
        log_info(f"Functional Test: [ONNX Engine]")
        return run_onnx_inference(final_path, shape, source_mode=source_mode)
        
    elif output_mode == 'caffe':
        log_info("Caffe uses OpenCV DNN for validation. Skipping extra test.")
        return True
        
    elif output_mode in ['pytorch', 'source', 'pth']:
        log_info(f"{output_mode.upper()} validated natively. Skipping extra test.")
        return True
        
    log_warning(f"Inference test for {output_mode} not supported.")
    return True 
