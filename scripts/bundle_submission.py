import os
import tarfile
import datetime

import argparse

def bundle_submission(sims: int = 50):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    output_dir = os.path.join("submissions", "draft", f"{timestamp}_{sims}sims")
    os.makedirs(output_dir, exist_ok=True)
    
    tar_path = os.path.join(output_dir, "submission.tar.gz")
    
    targets = [
        "submission_main.py",
        "src",
        "assets"
    ]
    
    print(f"Creating submission bundle at: {tar_path} (Simulations: {sims})")
    
    # Include deck.csv at root level
    deck_path = os.path.join("assets", "decks", "versatile", "Team_Rockets_Box.csv")
    
    # Generate temporary submission_main with injected env vars
    with open("submission_main.py", "r") as f:
        main_content = f.read()
    
    injected_content = f"import os\nos.environ['MCTS_SIMS'] = '{sims}'\n" + main_content
    with open("tmp_submission_main.py", "w") as f:
        f.write(injected_content)
    
    try:
        with tarfile.open(tar_path, "w:gz") as tar:
            if os.path.exists(deck_path):
                tar.add(deck_path, arcname="deck.csv")
                
            for target in targets:
                if not os.path.exists(target):
                    continue
                if os.path.isdir(target):
                    for root, dirs, files in os.walk(target):
                        if "__pycache__" in root or "/bak" in root or "\\bak" in root or root.endswith("bak"):
                            continue
                        for file in files:
                            if file.endswith(".pt") or file.endswith(".pyc") or file.endswith(".pdf"):
                                continue
                            if "latest_snapshot" in file:
                                continue
                            file_path = os.path.join(root, file)
                            tar.add(file_path, arcname=file_path)
                else:
                    if target == "submission_main.py":
                        tar.add("tmp_submission_main.py", arcname="main.py")
                    else:
                        tar.add(target, arcname=target)
    finally:
        if os.path.exists("tmp_submission_main.py"):
            os.remove("tmp_submission_main.py")
                
    size_mb = os.path.getsize(tar_path) / (1024 * 1024)
    print("Bundle successfully created!")
    print(f"Total Size: {size_mb:.2f} MB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bundle Kaggle submission")
    parser.add_argument("--sims", type=int, default=50, help="Number of MCTS simulations for the agent to run")
    args = parser.parse_args()
    bundle_submission(args.sims)
