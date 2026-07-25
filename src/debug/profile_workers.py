import multiprocessing, time, os, cProfile, copy
from kaggle_environments import make

os.environ['LITELLM_LOG'] = 'ERROR'
os.environ['SUPPRESS_LITELLM_WARNINGS'] = 'True'

def worker(i):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    _orig_deepcopy = copy.deepcopy
    def fast_deepcopy(x, memo=None, _nil=[]):
        if isinstance(x, dict):
            if "step" in x and "remainingOverageTime" in x:
                return x
            if "observation" in x and "reward" in x:
                new_dict = x.copy()
                new_dict["observation"] = x["observation"]
                return new_dict
        return _orig_deepcopy(x, memo)
    copy.deepcopy = fast_deepcopy
    
    import torch
    torch.set_num_threads(1)
    
    from src.core.agent import agent, load_deck
    
    deck = load_deck()
    env = make('cabt', configuration={"decks": [deck, deck]})
    
    pr = cProfile.Profile()
    pr.enable()
    
    env.run([agent, agent])
    
    pr.disable()
    out_dir = os.path.join("logs", "debug", "worker_profiles")
    os.makedirs(out_dir, exist_ok=True)
    pr.dump_stats(os.path.join(out_dir, f"worker_prof_{i}.prof"))

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    with multiprocessing.Pool(16) as p:
        p.map(worker, range(16))
