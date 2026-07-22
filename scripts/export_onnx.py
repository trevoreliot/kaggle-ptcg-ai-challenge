import torch
import os
from src.core.models.base import BaseNetwork

def export_onnx():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    models_dir = os.path.join("assets", "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Export all .pt files found in assets/models/
    import glob
    pt_files = glob.glob(os.path.join(models_dir, "*.pt"))
    
    if not pt_files:
        print("No .pt models found to export!")
        return
        
    dummy_input = torch.randn(1, 128).to(device)
    
    for pt_file in pt_files:
        basename = os.path.basename(pt_file)
        model_name = os.path.splitext(basename)[0]
        out_path = os.path.join(models_dir, f"{model_name}.onnx")
        
        print(f"Loading {basename}...")
        model = BaseNetwork().to(device)
        model.load_state_dict(torch.load(pt_file, map_location=device, weights_only=True))
        model.eval()
        
        print(f"Exporting to {out_path}...")
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
    print("All exports complete!")

if __name__ == "__main__":
    export_onnx()
