# --------------------------------------------------------------------------
# export_pytorch.py
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
from core.logger import log_error, log_info, log_success

def export_pytorch_to_onnx(model, input_shape, output_path):
    """
    Exports PyTorch model via ONNX tracer.
    @param model PyTorch model.
    @param input_shape shape for dummy input.
    @param output_path output location.
    @return True if export succeeded.
    """
    OPSET_VERSION = 11
    
    dummy_input = torch.randn(*input_shape, requires_grad=False)
    device = next(model.parameters()).device
    dummy_input = dummy_input.to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(dummy_input)
    
    if isinstance(outputs, (list, tuple)):
        output_names = [f'output{i}' for i in range(len(outputs))]
    else:
        output_names = ['output']

    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=OPSET_VERSION,
            do_constant_folding=True,
            input_names=['input'],
            output_names=output_names,
            dynamic_axes=None,
            verbose=False
        )
        return True
    except Exception as e:
        log_error(f"Native export failed: {e}")
        raise e
