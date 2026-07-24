import math
import random
from typing import List, Dict, Any, Callable
import sys
import os

# Ensure cg is importable
vendor_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "vendor", "cg-lib")
if vendor_path not in sys.path:
    sys.path.append(vendor_path)

try:
    import cg.api as cg_api
except ImportError:
    cg_api = None

class MCTSNode:
    def __init__(self, parent=None, action=None, search_state=None):
        self.parent = parent
        self.action = action # The action index that led to this node (relative to parent's options)
        self.children = []
        
        self.visits = 0
        self.value = 0.0
        self.prior = 0.0
        
        self.search_state = search_state # cg_api.SearchState
        
        self.untried_actions = None 
        self.is_terminal = False
        
        if search_state is not None:
            if search_state.observation.current and search_state.observation.current.result != -1:
                self.is_terminal = True
            elif search_state.observation.select:
                options = search_state.observation.select.option
                if options:
                    self.untried_actions = list(range(len(options)))
                else:
                    self.is_terminal = True
            else:
                self.is_terminal = True
        else:
            self.is_terminal = True
            
    def ucb1(self, c_puct=1.0):
        if self.visits == 0:
            return float('inf')
        q_value = self.value / self.visits
        u_value = c_puct * self.prior * math.sqrt(self.parent.visits) / (1 + self.visits)
        return q_value + u_value

class MCTSEngine:
    """
    Monte Carlo Tree Search utilizing cg-lib `search_begin` / `search_step`
    """
    def __init__(self, evaluator: Callable, num_simulations: int = 10, c_puct: float = 1.0):
        self.evaluator = evaluator
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        
    def search(self, obs_dict: dict, agent_deck: list, opponent_deck_pred: list) -> List[int]:
        if cg_api is None:
            return []
            
        cg_obs = cg_api.to_observation_class(obs_dict)
        
        # Approximate hidden states
        your_deck = [] if cg_obs.select.deck else agent_deck.copy() 
        your_prize = [5] * 6 # approximate with default cards if unknown
        opponent_deck = opponent_deck_pred.copy()
        opponent_prize = [5] * 6
        opponent_hand = [5] * 5
        opponent_active = [] # optional unless required by env
        
        try:
            root_state = cg_api.search_begin(
                cg_obs,
                your_deck,
                your_prize,
                opponent_deck,
                opponent_prize,
                opponent_hand,
                opponent_active
            )
        except Exception as e:
            print(f"MCTS search_begin error: {e}")
            return []
            
        root = MCTSNode(search_state=root_state)
        
        if not root.untried_actions:
            cg_api.search_release(root_state.searchId)
            cg_api.search_end()
            return []
            
        for _ in range(self.num_simulations):
            node = root
            
            # 1. Select
            while node.untried_actions is not None and len(node.untried_actions) == 0 and len(node.children) > 0:
                node = max(node.children, key=lambda c: c.ucb1(self.c_puct))
                    
            # 2. Expand
            if not node.is_terminal and node.untried_actions is not None and len(node.untried_actions) > 0:
                action = node.untried_actions.pop()
                
                try:
                    new_state = cg_api.search_step(node.search_state.searchId, [action])
                    child = MCTSNode(parent=node, action=action, search_state=new_state)
                except Exception as e:
                    child = MCTSNode(parent=node, action=action, search_state=None)
                
                child.prior = 1.0 / (len(node.children) + len(node.untried_actions) + 1)
                node.children.append(child)
                node = child
            
            # 3. Evaluate
            if not node.is_terminal and node.search_state is not None:
                value = self.evaluator(node.search_state) 
            else:
                value = -1.0 # Terminal or error
            
            # 4. Backpropagate
            while node is not None:
                node.visits += 1
                node.value += value
                node = node.parent
                
        # Return most visited action
        best_action_idx = -1
        if root.children:
            best_child = max(root.children, key=lambda c: c.visits)
            best_action_idx = best_child.action
        elif root.untried_actions:
            best_action_idx = root.untried_actions[0]
            
        # Clean up memory
        cg_api.search_end()
        
        if best_action_idx >= 0:
            # Handle maxCount > 1 if required by the root options
            max_c = cg_obs.select.maxCount
            selections = [best_action_idx]
            if max_c > 1:
                others = [x for x in range(len(cg_obs.select.option)) if x != best_action_idx]
                selections.extend(random.sample(others, min(max_c - 1, len(others))))
            return selections
        return []

