# --------------------------------------------------------------------------
# converter.py
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
import json
import torch
import torch.nn as nn
from core.logger import log_info, log_error

def export_to_pth(model, output_path):
    """
    Saves model state_dict.
    """
    if not output_path.lower().endswith('.pth'):
        output_path = os.path.splitext(output_path)[0] + '.pth'
    
    state_dict = model.state_dict()
    torch.save(state_dict, output_path)
    return output_path

def export_darknet_to_source(model, output_path):
    """
    Exports architecture to .py and weights to .pth.
    """
    if not output_path.lower().endswith('.py'):
        output_path = os.path.splitext(output_path)[0] + '.py'
    
    pth_path = os.path.splitext(output_path)[0] + '.pth'
    export_to_pth(model, pth_path)
    
    from core.darknet_parser import DarknetParser
    if isinstance(model, DarknetParser):
        source_code = _generate_darknet_source(model, os.path.basename(pth_path), os.path.basename(output_path))
    else:
        log_error("Source export only supports DarknetParser.")
        raise NotImplementedError("Manual source export not available for arbitrary models.")

    with open(output_path, 'w') as f:
        f.write(source_code)
        
    log_info(f"Source saved: {output_path}, {pth_path}")
    return output_path, pth_path

def _generate_darknet_source(model, pth_filename, py_filename):
    """
    Builds the standalone Python source code.
    """
    blocks_json = json.dumps(model.blocks)
    no_yolo_layer_flag = "True" if model.no_yolo_layer else "False"
    
    # Template contents extracted from export_pytorch_source.py
    # (Abbreviated here for brevity in the tool call, but would contain the full class structures)
    # ... [Full Template from export_pytorch_source.py] ...
    
    # Actually I should include the full template to be correct.
    # Re-using the logic from the viewed file.
    
    from .template import SOURCE_TEMPLATE # I'll create a template.py for the massive string
    
    source = SOURCE_TEMPLATE.replace('BLOCKS_PLACEHOLDER', blocks_json)
    source = source.replace('PTH_FILENAME', pth_filename)
    source = source.replace('DEFAULT_YOLO_FLAG', no_yolo_layer_flag)
    source = source.replace('PY_FILENAME', py_filename)
    
    return source
