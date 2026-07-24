import os
import tarfile
import datetime

def bundle_submission():
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    output_dir = os.path.join("submissions", "draft", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    tar_path = os.path.join(output_dir, "submission.tar.gz")
    
    targets = [
        "main.py",
        "src",
        "assets"
    ]
    
    print(f"Creating submission bundle at: {tar_path}")
    
    # Include deck.csv at root level
    deck_path = os.path.join("assets", "decks", "versatile", "Team_Rockets_Box.csv")
    
    with tarfile.open(tar_path, "w:gz") as tar:
        if os.path.exists(deck_path):
            tar.add(deck_path, arcname="deck.csv")
            
        for target in targets:
            if not os.path.exists(target):
                continue
            if os.path.isdir(target):
                for root, dirs, files in os.walk(target):
                    if "__pycache__" in root:
                        continue
                    for file in files:
                        if file.endswith(".pt") or file.endswith(".pyc") or file.endswith(".pdf"):
                            continue
                        file_path = os.path.join(root, file)
                        tar.add(file_path, arcname=file_path)
            else:
                tar.add(target, arcname=target)
                
    size_mb = os.path.getsize(tar_path) / (1024 * 1024)
    print("Bundle successfully created!")
    print(f"Total Size: {size_mb:.2f} MB")

if __name__ == "__main__":
    bundle_submission()
