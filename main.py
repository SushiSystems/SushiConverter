# --------------------------------------------------------------------------
# main.py
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
import sys
import torch
import argparse
from core.logger import set_color_mode
from core.darknet_parser import DarknetParser
from exporters.engine import ExportDispatcher
from validators.validator import run_validation
from inference.inference import run_inference_test
from core.logger import log_info, log_warning, log_error, log_success

def show_tutorial():
    """
    Shows usage examples from tutorial.txt.
    """
    tutorial_path = os.path.join(os.path.dirname(__file__), 'tutorial.txt')
    if os.path.exists(tutorial_path):
        with open(tutorial_path, 'r') as f:
            print(f.read())
    else:
        log_error("Tutorial file not found.")

def get_args():
    """
    Parses command line arguments.
    @return Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="SushiConverter: Optimize models for NPU deployment.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    group = parser.add_argument_group('Required Arguments')
    group.add_argument('--mode', type=str, choices=['darknet', 'pytorch', 'onnx'], 
                        help='Input model type:\n'
                             '  darknet: Use .cfg and .weights\n'
                             '  pytorch: Use .pt file\n'
                             '  onnx: Use .onnx file')
    
    group = parser.add_argument_group('Configuration')
    group.add_argument('--weights', type=str, help='Path to weight file (.weights, .pt, or .pth)')
    group.add_argument('--graph', '--cfg', dest='graph', type=str, help='Path to model graph definition (.cfg for Darknet)')
    group.add_argument('--output', type=str, default=None, help='Output path (default depends on mode)')
    group.add_argument('--output-mode', type=str, default='onnx', choices=['onnx', 'pytorch', 'caffe', 'source', 'pth'],
                        help='Output format:\n'
                             '  onnx: NPU optimized Opset 11\n'
                             '  pytorch: Standard PyTorch .pt file\n'
                             '  caffe: Caffe prototxt and caffemodel\n'
                             '  source: Standalone Python source code and .pth weights\n'
                             '  pth: Static weight file (state_dict)')
    group.add_argument('--shape', type=int, nargs=4, default=None, 
                        help='Input shape: B C H W (default: 1 3 416 416)')
    
    group = parser.add_argument_group('Advanced Flags')
    group.add_argument('--validate', action='store_true', help='Compare Source and Output results')
    group.add_argument('--no-simplify', action='store_true', help='Skip ONNX optimization step')
    group.add_argument('--yolo-layer', action='store_true', help='Include decoded YOLO predictions (default: output raw tensors)')
    group.add_argument('--tutorial', action='store_true', help='Show usage examples and exit')

    return parser.parse_args()

def main():
    """
    Main entry point for SushiConverter.
    """
    set_color_mode()
    args = get_args()
    
    if args.tutorial:
        show_tutorial()
        return

    if not args.mode:
        log_error("Please specify --mode or use --tutorial.")
        return

    no_yolo_layer = not args.yolo_layer
    
    if args.yolo_layer and args.mode != 'darknet':
        log_warning("--yolo-layer is only for darknet mode. Ignoring flag.")
        no_yolo_layer = True

    if args.shape is None:
        args.shape = [1, 3, 416, 416]
        if not (args.mode == 'darknet' and args.graph):
            log_info(f"Using default input shape: {args.shape}")
    else:
        log_info(f"Using input shape: {args.shape}")

    try:
        dispatcher = ExportDispatcher(
            input_mode=args.mode,
            output_mode=args.output_mode,
            weights_path=args.weights,
            graph_path=args.graph,
            shape=args.shape,
            output_path=args.output or "model",
            no_yolo_layer=no_yolo_layer,
            simplify=not args.no_simplify
        )
        
        final_path, model = dispatcher.run()
        args.shape = dispatcher.shape
        yolo_status = "Raw Tensors" if no_yolo_layer else "Decoded Boxes"

        if args.validate:
            is_valid = run_validation(args.output_mode, model, final_path, args.shape, source_mode=args.mode)
            inference_ok = run_inference_test(args.output_mode, final_path, args.shape, source_mode=args.mode)
            
            if is_valid and inference_ok:
                log_success(f"Full validation pipeline PASSED for {args.output_mode}.")
            elif is_valid:
                log_warning(f"Numerical validation for {args.output_mode} passed, but functional inference failed.")
            else:
                log_error(f"Validation for {args.output_mode} failed.")
        else:
            is_valid, inference_ok = None, None

        print(f"\n" + "="*50)
        print(f"  CONVERSION REPORT")
        print(f"="*50)
        print(f"  Source Model : {args.weights or args.graph}")
        print(f"  Output Path  : {final_path}")
        print(f"  Input Shape  : {args.shape}")
        
        if args.validate and is_valid is not None:
            status = "PASSED" if (is_valid and inference_ok) else ("PARTIAL (Phase 1 OK)" if is_valid else "FAILED")
            print(f"  YOLO Layer   : {yolo_status}")
            print(f"  Validation   : {status}")
        
        print("="*50 + "\n")
        
    except Exception as e:
        log_error(f"Pipeline failed: {e}")
        # # TODO: Add advanced diagnostic logging
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()