# --------------------------------------------------------------------------
# export_pytorch_source.py
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
    @param model PyTorch model instance.
    @param output_path path to .pth file.
    @return actual output path.
    """
    if not output_path.lower().endswith('.pth'):
        output_path = os.path.splitext(output_path)[0] + '.pth'
    
    state_dict = model.state_dict()
    torch.save(state_dict, output_path)
    return output_path

def export_to_source(model, output_path):
    """
    Exports architecture to .py and weights to .pth.
    @param model Model to export.
    @param output_path base filename.
    @return tuple of paths created.
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
        
    return output_path, pth_path

def _generate_darknet_source(model, pth_filename, py_filename):
    """
    Builds the standalone Python source.
    @param model DarknetParser source.
    @param pth_filename name of weights file.
    @param py_filename name of source file.
    @return Python code string.
    """
    blocks_json = json.dumps(model.blocks)
    no_yolo_layer_flag = "True" if model.no_yolo_layer else "False"
    
    template = f'''# --------------------------------------------------------------------------
# {{py_filename}}
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
import warnings
import torch.nn as nn
import torch.nn.functional as F

class Mish(nn.Module):
    """
    Activates using Mish function.
    """
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class MaxPoolDark(nn.Module):
    """
    Pooling with Darknet padding.
    """
    def __init__(self, size=2, stride=1):
        super(MaxPoolDark, self).__init__()
        self.size = size
        self.stride = stride
    def forward(self, x):
        p = self.size // 2
        if ((x.shape[2] - 1) // self.stride) != ((x.shape[2] + 2 * p - self.size) // self.stride):
            pad = (p, p + 1, p, p + 1)
        else:
            pad = (p, p, p, p)
        x = F.max_pool2d(F.pad(x, pad, mode='replicate'), self.size, stride=self.stride)
        return x

class YOLOLayer(nn.Module):
    """
    Decodes YOLO feature maps.
    """
    def __init__(self, anchors, mask, classes, img_size, no_yolo_layer=False):
        super(YOLOLayer, self).__init__()
        self.anchors = torch.FloatTensor([anchors[i] for i in mask])
        self.classes = classes
        self.img_size = img_size
        self.no_yolo_layer = no_yolo_layer
        self.num_anchors = len(mask)
        self.register_buffer('anchor_grid', self.anchors.clone().view(1, self.num_anchors, 1, 1, 2))
        self.grid = None

    def _make_grid(self, nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)], indexing='ij')
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def forward(self, x):
        if self.no_yolo_layer: return x
        B, C, H, W = x.shape
        x = x.view(B, self.num_anchors, 5 + self.classes, H, W).permute(0, 1, 3, 4, 2).contiguous()
        xy, wh, conf_cls = torch.split(x, [2, 2, self.classes + 1], dim=-1)
        xy = torch.sigmoid(xy)
        conf_cls = torch.sigmoid(conf_cls)
        if self.grid is None or self.grid.shape[2:4] != (H, W):
            self.grid = self._make_grid(W, H).to(x.device)
        stride_x, stride_y = self.img_size[1] / W, self.img_size[0] / H
        stride = torch.tensor([stride_x, stride_y], device=x.device, dtype=x.dtype)
        xy = (xy + self.grid) * stride
        wh = torch.exp(wh) * self.anchor_grid
        pred = torch.cat((xy, wh, conf_cls), dim=-1)
        return pred.view(B, -1, 5 + self.classes)

class SushiModel(nn.Module):
    """
    Standalone Darknet execution model.
    """
    def __init__(self, no_yolo_layer=DEFAULT_YOLO_FLAG):
        super(SushiModel, self).__init__()
        self.blocks = json.loads(\'\'\'BLOCKS_PLACEHOLDER\'\'\')
        self.net_info = self.blocks[0]
        self.no_yolo_layer = no_yolo_layer
        self.models = self._create_network(self.blocks)
        
    def _create_network(self, blocks):
        models = nn.ModuleList()
        prev_filters = int(self.net_info.get(\'channels\', 3))
        out_filters = []
        conv_id = 0
        
        for block in blocks[1:]:
            module = nn.Sequential()
            b_type = block[\'type\']
            
            if b_type == \'convolutional\':
                conv_id += 1
                bn = int(block.get(\'batch_normalize\', 0))
                filters, size, stride = int(block[\'filters\']), int(block[\'size\']), int(block[\'stride\'])
                pad = (size - 1) // 2 if int(block.get(\'pad\', 0)) else 0
                act = block[\'activation\']
                
                module.add_module(f\'conv{{conv_id}}\', nn.Conv2d(prev_filters, filters, size, stride, pad, bias=not bn))
                if bn: module.add_module(f\'bn{{conv_id}}\', nn.BatchNorm2d(filters))
                if act == \'leaky\': module.add_module(f\'leaky{{conv_id}}\', nn.LeakyReLU(0.1, inplace=True))
                elif act == \'mish\': module.add_module(f\'mish{{conv_id}}\', Mish())
                
                prev_filters = filters
            elif b_type == \'maxpool\':
                size, stride = int(block[\'size\']), int(block[\'stride\'])
                if stride == 1 and size % 2: module = nn.MaxPool2d(size, stride, size // 2)
                elif stride == size: module = nn.MaxPool2d(size, stride, 0)
                else: module = MaxPoolDark(size, stride)
            elif b_type == \'upsample\':
                module = nn.Upsample(scale_factor=int(block[\'stride\']), mode=\'nearest\')
            elif b_type == \'route\':
                layers = [int(i) if int(i) >= 0 else int(i) + len(models) for i in block[\'layers\'].split(\',\')]
                if \'groups\' in block:
                    total = out_filters[layers[0]] // int(block[\'groups\'])
                else:
                    total = sum([out_filters[l] for l in layers])
                prev_filters = total
                module = nn.Identity()
            elif b_type == \'shortcut\':
                prev_filters = out_filters[-1]
                module = nn.Identity()
            elif b_type == \'yolo\':
                mask = [int(x) for x in block[\'mask\'].split(\',\')]
                anchors = [int(x) for x in block[\'anchors\'].split(\',\')]
                anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
                module = YOLOLayer(anchors, mask, int(block[\'classes\']), (int(self.net_info[\'height\']), int(self.net_info[\'width\'])), self.no_yolo_layer)
            else:
                module = nn.Identity()
            
            models.append(module)
            out_filters.append(prev_filters)
        return models

    def forward(self, x):
        outputs = {{}}
        yolo_outputs = []
        for i, block in enumerate(self.blocks[1:]):
            m = self.models[i]
            b_type = block[\'type\']
            if b_type in [\'convolutional\', \'maxpool\', \'upsample\', \'avgpool\']:
                x = m(x)
            elif b_type == \'route\':
                layers = [int(l) if int(l) >= 0 else int(l) + i for l in block[\'layers\'].split(\',\')]
                if len(layers) == 1:
                    x = outputs[layers[0]]
                    if \'groups\' in block:
                        g, gid = int(block[\'groups\']), int(block[\'group_id\'])
                        x = torch.split(x, x.shape[1] // g, dim=1)[gid]
                else:
                    x = torch.cat([outputs[l] for l in layers], dim=1)
            elif b_type == \'shortcut\':
                x = outputs[i-1] + outputs[int(block[\'from\']) + i if int(block[\'from\']) < 0 else int(block[\'from\'])]
            elif b_type == \'yolo\':
                x = m(x)
                yolo_outputs.append(x)
            outputs[i] = x

        if len(yolo_outputs) > 0:
            if self.no_yolo_layer:
                return yolo_outputs
            return torch.cat(yolo_outputs, 1)
        return x

def load_model(weights_path=None, no_yolo_layer=DEFAULT_YOLO_FLAG):
    """
    Loads architecture and weights.
    @return Model initialized.
    """
    model = SushiModel(no_yolo_layer=no_yolo_layer)
    if weights_path:
        model.load_state_dict(torch.load(weights_path, map_location=\'cpu\'))
    return model

if __name__ == "__main__":
    import sys
    weights = "PTH_FILENAME" if len(sys.argv) < 2 else sys.argv[1]
    if not os.path.exists(weights):
        weights = None
        
    model = load_model(weights)
    model.eval()
    dummy_input = torch.randn(1, int(model.net_info[\'channels\']), int(model.net_info[\'height\']), int(model.net_info[\'width\']))
    with torch.no_grad():
        output = model(dummy_input)
    
    if isinstance(output, list):
        print(f"YOLO Outputs: {{[o.shape for o in output]}}")
    else:
        print(f"Raw Output Shape: {{output.shape}}")
'''
    source = template.replace('BLOCKS_PLACEHOLDER', blocks_json)
    source = source.replace('PTH_FILENAME', pth_filename)
    source = source.replace('DEFAULT_YOLO_FLAG', no_yolo_layer_flag)
    
    return source
