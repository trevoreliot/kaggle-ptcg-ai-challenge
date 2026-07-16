import torch
import os
from src.core.models.base import BaseNetwork

def export_onnx():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BaseNetwork().to(device)
    model.eval()
    
    os.makedirs("assets/models", exist_ok=True)
    out_path = os.path.join("assets", "models", "general.onnx")
    
    dummy_input = torch.randn(1, 128).to(device)
    
    print(f"Exporting ONNX model to {out_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        out_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["value", "policy"],
        dynamic_axes={"input": {0: "batch_size"}, "value": {0: "batch_size"}, "policy": {0: "batch_size"}}
    )
    print("Export complete!")

if __name__ == "__main__":
    export_onnx()
