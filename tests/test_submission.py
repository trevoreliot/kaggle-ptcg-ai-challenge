import os
from kaggle_environments import make

def test_submission():
    import glob
    drafts = sorted(glob.glob("submissions/draft/*"))
    if not drafts:
        print("No drafts found")
        return
        
    latest = drafts[-1]
    tar_path = os.path.abspath(os.path.join(latest, "submission.tar.gz"))
    
    print(f"Testing {tar_path}...")
    
    # Load 2 simple decks
    from main import load_deck
    p1_deck = load_deck("assets/decks/versatile/Team_Rockets_Box.csv")
    p2_deck = load_deck("assets/decks/versatile/Team_Rockets_Box.csv")
    
    env = make("cabt", configuration={"decks": [p1_deck, p2_deck]}, debug=True)
    
    # In a kaggle env, the agent is loaded from the tarball
    print("Running match...")
    env.run([tar_path, tar_path])
    
    print("Match finished!")
    if env.state[0].status == "ERROR":
        print("P1 ERROR:", env.state[0].get("error", "No error info"))
        print(env.state[0].get("traceback", ""))
    if env.state[1].status == "ERROR":
        print("P2 ERROR:", env.state[1].get("error", "No error info"))
        print(env.state[1].get("traceback", ""))
        
if __name__ == "__main__":
    test_submission()
