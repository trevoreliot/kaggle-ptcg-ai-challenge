import glob
import json
import os

def main():
    input_dir = "samples/rules-based-agents"
    output_dir = "src/core/rules_agents"
    os.makedirs(output_dir, exist_ok=True)
    
    # Touch __init__.py so the folder becomes a python package
    init_file = os.path.join(output_dir, "__init__.py")
    if not os.path.exists(init_file):
        with open(init_file, "w") as f:
            f.write("# Auto-generated module for rules-based agents\n")
            
    notebooks = glob.glob(os.path.join(input_dir, "*.ipynb"))
    
    for nb_path in notebooks:
        basename = os.path.basename(nb_path)
        name = basename.replace(".ipynb", "").replace("-", "_")
        
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)
            
        extracted_source = None
        for cell in nb.get("cells", []):
            if cell.get("cell_type") == "code":
                source = "".join(cell.get("source", []))
                if "%%writefile" in source:
                    # Strip the first line (%%writefile main.py)
                    lines = source.split("\n")
                    if lines and "%%writefile" in lines[0]:
                        extracted_source = "\n".join(lines[1:])
                    else:
                        extracted_source = source
                        
                    # Fix the hardcoded deck.csv so the module can actually be imported without throwing FileNotFoundError
                    extracted_source = extracted_source.replace('"deck.csv"', 'os.path.join("assets", "decks", "versatile", "Team_Rockets_Box.csv")')
                    break
                    
        if extracted_source:
            out_path = os.path.join(output_dir, f"{name}.py")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(extracted_source)
            print(f"Extracted {basename} -> {out_path}")
        else:
            print(f"Failed to find %%writefile block in {basename}")

if __name__ == "__main__":
    main()
