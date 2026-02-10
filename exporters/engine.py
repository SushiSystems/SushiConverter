import torch
import torch.nn as nn
import os
import onnx
from core.logger import log_info, log_warning, log_error, log_success

class RawNPUHead(nn.Module):
    def __init__(self, original_layer):
        super().__init__()
        self.nl = original_layer.nl
        self.cv2 = original_layer.cv2

    def forward(self, x):
        res = []
        for i in range(self.nl):
            feat = self.cv2[i](x[i])
            res.append(feat)
        return res

def export_pytorch_to_onnx(model, input_shape, output_path):
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
        log_error(f"Export failed: {e}")
        raise e

def try_export_ultralytics(model, input_shape, output_path):
    OPSET_VERSION = 11

    model_layers = None
    if hasattr(model, 'model') and isinstance(model.model, nn.Sequential):
        model_layers = model.model
    elif isinstance(model, nn.Sequential):
        model_layers = model
    
    if model_layers is None:
        return False

    try:
        last_layer = model_layers[-1]
        if not (hasattr(last_layer, 'cv2') and hasattr(last_layer, 'nl')):
            return False
    except:
        return False

    log_info("Ultralytics model detected. Applying NPU optimization...")

    try:
        new_layer = RawNPUHead(last_layer)
        for attr in ['i', 'f', 'type']:
            if hasattr(last_layer, attr):
                setattr(new_layer, attr, getattr(last_layer, attr))
        
        model_layers[-1] = new_layer
        log_info("Replaced Detect layer with NPU-friendly head.")
        
    except Exception as e:
        log_warning(f"Could not apply NPU transformation: {e}")
        return False

    try:
        device = next(model.parameters()).device
        dummy_input = torch.randn(*input_shape).to(device)
        model.eval()

        log_info(f"Exporting to ONNX (Opset {OPSET_VERSION})...")
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            opset_version=OPSET_VERSION,
            input_names=['images'],
            output_names=['output0'],
            do_constant_folding=True
        )

        onnx_model = onnx.load(output_path)
        onnx.save_model(onnx_model, output_path, save_as_external_data=False)
        
        data_file = output_path + ".data"
        if os.path.exists(data_file):
            os.remove(data_file)

        log_success("Ultralytics export completed.")
        return True

    except Exception as e:
        log_error(f"Ultralytics export failed: {e}")
        raise e