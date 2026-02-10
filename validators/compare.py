import torch
import numpy as np
import onnxruntime as ort
from core.logger import log_info, log_warning, log_error, log_success

def validate_conversion(pt_model, onnx_path, input_shape, tolerance=1e-4):
    dummy_input = torch.randn(*input_shape).float()
    
    pt_model.eval()
    with torch.no_grad():
        pt_out = pt_model(dummy_input)
    
    if isinstance(pt_out, (list, tuple)):
        pt_out = [x.numpy() for x in pt_out]
    else:
        pt_out = [pt_out.numpy()]
        
    try:
        session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    except Exception as e:
        log_error(f"Could not load ONNX model for validation: {e}")
        return False

    input_name = session.get_inputs()[0].name
    try:
        onnx_out = session.run(None, {input_name: dummy_input.numpy()})
    except Exception as e:
        log_error(f"ONNX execution failed during validation: {e}")
        return False
    
    log_info(f"Comparing outputs: {len(pt_out)} predicted tensors.")
    
    if len(pt_out) != len(onnx_out):
        log_error(f"Output count mismatch: PyTorch has {len(pt_out)}, ONNX has {len(onnx_out)}")
        return False

    all_pass = True
    for i, (p, o) in enumerate(zip(pt_out, onnx_out)):
        if p.shape != o.shape:
            log_error(f"Output {i} shape mismatch: PyTorch {p.shape} vs ONNX {o.shape}")
            all_pass = False
            continue
            
        mae = np.mean(np.abs(p - o))
        shape_str = f"Shape: {tuple(o.shape)}"
        
        if np.isnan(mae):
            log_error(f"Output {i} ({shape_str}) contains NaN values.")
            all_pass = False
        elif mae > tolerance:
            log_warning(f"Output {i} ({shape_str}) difference too high: {mae:.6f} > {tolerance}")
            all_pass = False
        else:
            log_success(f"Output {i} ({shape_str}) validated (MAE: {mae:.6f}).")
            
    return all_pass
