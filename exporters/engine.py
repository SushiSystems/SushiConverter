# --------------------------------------------------------------------------
# engine.py
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
import torch
from core.darknet_parser import DarknetParser
from exporters.post_process import optimize_onnx
from exporters.export_onnx import load_onnx_to_pytorch
from exporters.export_pytorch import export_pytorch_to_onnx
from exporters.export_caffe import export_pytorch_to_caffe
from exporters.export_pytorch_source import export_to_source, export_to_pth
from core.logger import log_info, log_error, log_success, log_warning
from exporters.export_ultralytics import is_ultralytics_model, export_ultralytics_to_onnx

class ExportDispatcher:
    """
    Main controller for exporting models between different formats.
    """
    def __init__(self, input_mode, output_mode, weights_path=None, graph_path=None, 
                 shape=None, output_path="model", no_yolo_layer=True, simplify=True):
        """
        Setup dispatcher state.
        @param input_mode format of input (darknet/pytorch/onnx).
        @param output_mode format of output.
        @param weights_path file path for weights.
        @param graph_path file path for graph definition.
        @param shape target input dimensions.
        @param output_path base output filename.
        @param no_yolo_layer skip YOLO post-processing.
        @param simplify optimize ONNX graph.
        """
        self.input_mode = input_mode
        self.output_mode = output_mode
        self.weights_path = weights_path
        self.graph_path = graph_path
        self.shape = shape or [1, 3, 416, 416]
        self.output_path = output_path
        self.no_yolo_layer = no_yolo_layer
        self.simplify = simplify

        if self.output_mode == 'onnx' and not self.output_path.lower().endswith('.onnx'):
            self.output_path += '.onnx'
        elif self.output_mode == 'pytorch' and not self.output_path.lower().endswith('.pt'):
            self.output_path += '.pt'
        elif self.output_mode == 'caffe' and '.' not in os.path.basename(self.output_path):
            self.output_path += '.caffemodel'
        elif self.output_mode == 'source' and not self.output_path.lower().endswith('.py'):
            self.output_path += '.py'
        elif self.output_mode == 'pth' and not self.output_path.lower().endswith('.pth'):
            self.output_path += '.pth'

    def _load_darknet(self):
        """
        Loads a Darknet network.
        @return Initialized DarknetParser model.
        """
        log_info("Loading DARKNET model...")
        if not self.graph_path:
            raise ValueError("Darknet requires a graph file (.cfg).")
        model = DarknetParser(self.graph_path, no_yolo_layer=self.no_yolo_layer)
        if self.weights_path:
            model.load_weights(self.weights_path)
        else:
            log_info("Missing weights. Random initialization used.")
        
        if hasattr(model, 'width') and hasattr(model, 'height'):
            self.shape = [self.shape[0], self.shape[1], model.height, model.width]
            
        return model

    def _load_pytorch(self):
        """
        Loads a PyTorch model file.
        @return loaded PyTorch model instance.
        """
        log_info(f"Loading PYTORCH model from {self.weights_path}...")
        if not self.weights_path:
            raise ValueError("PyTorch requires a weights file (.pt).")
            
        try:
            try:
                loaded = torch.load(self.weights_path, map_location='cpu', weights_only=False)
            except TypeError:
                loaded = torch.load(self.weights_path, map_location='cpu')

            if isinstance(loaded, dict):
                if 'model' in loaded:
                    model = loaded['model']
                    if hasattr(model, 'float'): 
                        model.float()
                elif 'state_dict' in loaded:
                    raise ValueError("State-dict alone is not sufficient. Architecture missing.")
                else:
                    model = loaded
            else:
                model = loaded
        except ModuleNotFoundError as e:
            if 'models' in str(e) or 'utils' in str(e):
                log_info("YOLOv5 detected. Using torch.hub fallback.")
                try:
                    model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.weights_path, force_reload=False, trust_repo=True)
                except Exception as ex:
                    raise RuntimeError(f"Fallback Hub load failed: {ex}")
            else:
                raise e

        if hasattr(model, 'fuse'):
            try: 
                model.fuse()
            except: 
                pass
            
        return model

    def run(self):
        """
        Launches the conversion process.
        @return tuple including resultant file path and model.
        """
        log_info(f"Pipeline started: [{self.input_mode}] -> [{self.output_mode}]")
        
        if self.input_mode == 'onnx':
            if not self.weights_path:
                raise ValueError("ONNX input expects a file path.")
                
            if self.output_mode == 'pytorch':
                model = load_onnx_to_pytorch(self.weights_path)
                torch.save(model, self.output_path)
                log_success(f"Final PyTorch saved to {self.output_path}")
                return self.output_path, model
                
            elif self.output_mode == 'onnx':
                log_warning("Input is already ONNX. Optimizing graph.")
                final_path = optimize_onnx(self.weights_path, simplify=self.simplify)
                log_success(f"Optimized model at {final_path}")
                return final_path, None
                
        model = None
        if self.input_mode == 'darknet':
            model = self._load_darknet()
            log_success("Darknet loaded.")
        elif self.input_mode == 'pytorch':
            model = self._load_pytorch()
            log_success("PyTorch loaded.")
            
        if model is not None and hasattr(model, 'eval'):
            model.eval()
            
        if self.output_mode == 'pytorch':
            torch.save(model, self.output_path)
            log_success(f"PyTorch exported to {self.output_path}")
            return self.output_path, model
            
        elif self.output_mode == 'caffe':
            export_pytorch_to_caffe(model, self.shape, self.output_path)
            log_success(f"Caffe exported to {self.output_path}")
            return self.output_path, model
            
        elif self.output_mode == 'source':
            py_path, pth_path = export_to_source(model, self.output_path)
            log_success(f"Standalone code at {py_path} and weights at {pth_path}")
            return py_path, model

        elif self.output_mode == 'pth':
            pth_path = export_to_pth(model, self.output_path)
            log_success(f"Weights saved at {pth_path}")
            return pth_path, model
            
        elif self.output_mode == 'onnx':
            is_ultra, _ = is_ultralytics_model(model)
            if is_ultra and self.input_mode == 'pytorch':
                success = export_ultralytics_to_onnx(model, self.shape, self.output_path)
            else:
                success = export_pytorch_to_onnx(model, self.shape, self.output_path)
                
            if not success:
                raise RuntimeError("ONNX export failed.")
                
            log_info("Applying NPU patches...")
            final_path = optimize_onnx(self.output_path, simplify=self.simplify)
            log_success(f"Final model at {final_path}")
            return final_path, model
            
        raise ValueError(f"Unknown output type: {self.output_mode}")