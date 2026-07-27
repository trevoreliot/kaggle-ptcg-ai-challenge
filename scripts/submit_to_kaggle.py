import os
import glob
import subprocess
import argparse

def submit_latest(message: str):
    drafts = sorted(glob.glob("submissions/draft/*"))
    if not drafts:
        print("No drafts found to submit!")
        return
        
    latest_draft = drafts[-1]
    tar_path = os.path.join(latest_draft, "submission.tar.gz")
    
    if not os.path.exists(tar_path):
        print(f"File not found: {tar_path}")
        return
        
    print(f"Submitting {tar_path}...")
    
    cmd = [
        "kaggle",
        "competitions",
        "submit",
        "-c", "pokemon-tcg-ai-battle",
        "-f", tar_path,
        "-m", message
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("Submission successful!")
    except subprocess.CalledProcessError as e:
        print(f"Submission failed with error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit latest draft to Kaggle")
    parser.add_argument("-m", "--message", type=str, default="Automated submission via AI",
                        help="The submission message")
    args = parser.parse_args()
    submit_latest(args.message)
