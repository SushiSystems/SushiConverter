# --------------------------------------------------------------------------
# onnx_optimizer.py
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
from core.logger import log_info, log_error

def optimize_onnx(onnx_path, simplify=True):
    """
    Optimizes ONNX graph for hardware deployment.
    """
    if not simplify:
        return onnx_path

    try:
        import onnx
        from onnxsim import simplify
        
        log_info(f"Simplifying ONNX graph: {onnx_path}")
        model = onnx.load(onnx_path)
        model_simp, check = simplify(model)
        
        if not check:
            log_error("Simulated graph match failed. Results might be invalid.")
        
        base, ext = os.path.splitext(onnx_path)
        optimized_path = f"{base}_optimized{ext}"
        onnx.save(model_simp, optimized_path)
        
        return optimized_path
    
    except ImportError:
        log_error("onnxsim not found. Skipping simplification.")
        return onnx_path
    except Exception as e:
        log_error(f"Optimization failed: {e}")
        return onnx_path
