import math
import random
from typing import List, Dict, Any, Callable
from kaggle_environments.core import Environment

class MCTSNode:
    def __init__(self, parent=None, action=None):
        self.parent = parent
        self.action = action # The action that led to this node (from perspective of active player)
        self.children = []
        
        self.visits = 0
        self.value = 0.0
        self.prior = 0.0 # From Policy network
        
        self.untried_actions = None # Populated upon expansion
        self.is_terminal = False
        
    def ucb1(self, c_puct=1.0):
        if self.visits == 0:
            return float('inf')
        q_value = self.value / self.visits
        u_value = c_puct * self.prior * math.sqrt(self.parent.visits) / (1 + self.visits)
        return q_value + u_value

class MCTSEngine:
    """
    Monte Carlo Tree Search that utilizes kaggle_environments `env.clone()` 
    to branch the simulation forward.
    """
    def __init__(self, evaluator: Callable, num_simulations: int = 10, c_puct: float = 1.0):
        self.evaluator = evaluator
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        
    def _get_valid_actions(self, obs: dict) -> List[List[int]]:
        """Helper to extract a list of valid single-choice actions from an observation."""
        select_data = obs.get("select")
        if not select_data:
            return [[]]
            
        options = select_data.get("option", [])
        if not options:
            return [[]]
            
        # Simplified: Generate single-index actions. 
        # A robust version handles maxCount > 1 combinations.
        return [[i] for i in range(len(options))]
        
    def search(self, initial_env: Environment, agent_index: int) -> List[int]:
        root = MCTSNode()
        
        obs = initial_env.state[0].observation
        root.untried_actions = self._get_valid_actions(obs)
        
        if not root.untried_actions or root.untried_actions == [[]]:
            return []
            
        for _ in range(self.num_simulations):
            node = root
            env = initial_env.clone()
            
            # 1. Select
            while node.untried_actions is not None and len(node.untried_actions) == 0 and len(node.children) > 0:
                node = max(node.children, key=lambda c: c.ucb1(self.c_puct))
                
                # Determine whose turn it is currently in the simulation
                curr_obs = env.state[0].observation
                active_index = curr_obs.get("current", {}).get("yourIndex", agent_index)
                
                actions = [[]] * 2
                actions[active_index] = node.action
                try:
                    env.step(actions)
                except Exception:
                    node.is_terminal = True
                    break
                    
            # 2. Expand
            if not node.is_terminal and node.untried_actions is not None and len(node.untried_actions) > 0:
                action = node.untried_actions.pop()
                child = MCTSNode(parent=node, action=action)
                
                # Assign a prior probability uniformly for now (will be replaced by Policy Network)
                child.prior = 1.0 / (len(node.children) + len(node.untried_actions) + 1)
                
                node.children.append(child)
                node = child
                
                curr_obs = env.state[0].observation
                active_index = curr_obs.get("current", {}).get("yourIndex", agent_index)
                actions = [[]] * 2
                actions[active_index] = action
                
                try:
                    steps = env.step(actions)
                    new_obs = steps[0].observation
                    
                    if new_obs.get("current", {}).get("result", -1) >= 0:
                        node.is_terminal = True
                    else:
                        node.untried_actions = self._get_valid_actions(new_obs)
                except Exception:
                    node.is_terminal = True
            
            # 3. Evaluate
            # Use the evaluator callback (e.g., Value Network) to score the current state
            value = self.evaluator(env, agent_index) if not node.is_terminal else -1.0
            
            # 4. Backpropagate
            while node is not None:
                node.visits += 1
                node.value += value
                node = node.parent
                
        # Return most visited action
        if not root.children:
            return root.untried_actions[0] if root.untried_actions else []
            
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.action
