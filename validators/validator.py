# --------------------------------------------------------------------------
# validator.py
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

from .onnx_validator import validate_onnx
from .caffe_validator import validate_caffe
from .pytorch_validator import validate_pytorch
from .source_validator import validate_standalone_source
from core.logger import log_info, log_warning, log_error

def run_validation(output_mode, model, final_path, shape, source_mode='darknet'):
    """
    Dispatcher for numerical accuracy checks.
    @param output_mode format to validate.
    @param model source model instance.
    @param final_path exported file path.
    @param shape input dimensions.
    @param source_mode input format.
    @return True if validation passed.
    """
    if model is None:
        log_warning(f"Skipping validation for {output_mode}: model missing.")
        return False

    if output_mode == 'onnx':
        log_info(f"Numerical Validation: [{source_mode.upper()} -> ONNX]")
        return validate_onnx(model, final_path, shape)
        
    elif output_mode == 'caffe':
        log_info(f"Numerical Validation: [{source_mode.upper()} -> Caffe]")
        base_name = final_path
        if base_name.endswith('.caffemodel'): 
            base_name = base_name[:-11]
        prototxt = base_name + ".prototxt"
        caffemodel = base_name + ".caffemodel"
        return validate_caffe(model, prototxt, caffemodel, shape)
        
    elif output_mode == 'pytorch':
        log_info(f"Validation: [{source_mode.upper()} -> PyTorch]")
        return validate_pytorch(model, final_path)
        
    elif output_mode == 'source':
        log_info(f"Validation: [{source_mode.upper()} -> Source]")
        return validate_standalone_source(model, final_path, shape)
        
    log_warning(f"Validation for {output_mode} not supported.")
    return None
