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
                    print(f"DEBUG MCTS: node initialized with {len(options)} options, is_root={self.parent is None}")
                    if len(options) > 1:
                        end_idx = next((i for i, opt in enumerate(options) if opt.type == 14), None)
                        if end_idx is not None:
                            self.untried_actions.remove(end_idx)
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
        
    def search(self, obs_dict: dict, agent_deck: list, opponent_deck_pred: list, is_training: bool = False, temperature: float = 1.0, epsilon: float = 0.25, alpha: float = 0.3, valid_mask: list = None, action_phases: dict = None, phase_budgeting: bool = False) -> List[int]:
        if cg_api is None:
            return []
            
        cg_obs = cg_api.to_observation_class(obs_dict)
        
        st = cg_obs.current
        your_p = st.players[st.yourIndex]
        opp_p = st.players[1 - st.yourIndex]
        
        your_deck = [] if cg_obs.select.deck else agent_deck.copy() 
        if len(your_deck) < your_p.deckCount:
            your_deck.extend([5] * (your_p.deckCount - len(your_deck)))
            
        your_prize = [5] * max(1, len(your_p.prize))
        
        opponent_deck = opponent_deck_pred.copy()
        if len(opponent_deck) < opp_p.deckCount:
            opponent_deck.extend([5] * (opp_p.deckCount - len(opponent_deck)))
            
        opponent_prize = [5] * max(1, len(opp_p.prize))
        opponent_hand = [5] * max(1, opp_p.handCount)
        
        opponent_active = []
        if len(opp_p.active) > 0 and opp_p.active[0] is None:
            opponent_active = [722] # 722 is a valid Basic Pokemon ID just in case
        
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
        
        if valid_mask is not None and root.untried_actions:
            filtered_untried = []
            for a in root.untried_actions:
                if a < len(valid_mask) and valid_mask[a]:
                    filtered_untried.append(a)
            root.untried_actions = filtered_untried
        
        if not root.untried_actions:
            cg_api.search_release(root_state.searchId)
            cg_api.search_end()
            return []
            
        import numpy as np
        root_noise_map = {}
        if is_training and root.untried_actions and alpha > 0.0:
            root_noise = np.random.dirichlet([alpha] * len(root.untried_actions))
            root_noise_map = {action: noise for action, noise in zip(root.untried_actions, root_noise)}
            
        phase_schedule = []
        if phase_budgeting and action_phases and root.untried_actions:
            phases_present = {}
            for a in root.untried_actions:
                p = action_phases.get(a, 0)
                if p not in phases_present:
                    phases_present[p] = []
                phases_present[p].append(a)
                
            if len(phases_present) > 0:
                sims_per_phase = self.num_simulations // len(phases_present)
                for p in sorted(phases_present.keys()):
                    phase_schedule.extend([p] * sims_per_phase)
                leftover = self.num_simulations - len(phase_schedule)
                phase_schedule.extend([-1] * leftover)

        for sim_idx in range(self.num_simulations):
            node = root
            
            allowed_root_phase = -1
            if phase_budgeting and action_phases and sim_idx < len(phase_schedule):
                allowed_root_phase = phase_schedule[sim_idx]
            
            # 1. Select
            while node.untried_actions is not None:
                has_untried = len(node.untried_actions) > 0
                
                valid_untried = []
                valid_children = []
                
                if node == root and allowed_root_phase != -1:
                    valid_untried = [a for a in node.untried_actions if action_phases.get(a, 0) == allowed_root_phase]
                    valid_children = [c for c in node.children if action_phases.get(c.action, 0) == allowed_root_phase]
                    
                    if not valid_untried and not valid_children:
                        allowed_root_phase = -1
                    else:
                        has_untried = len(valid_untried) > 0
                        
                if has_untried:
                    break
                    
                if len(node.children) == 0:
                    break
                    
                if node == root and allowed_root_phase != -1:
                    node = max(valid_children, key=lambda c: c.ucb1(self.c_puct))
                else:
                    node = max(node.children, key=lambda c: c.ucb1(self.c_puct))
                    
            # 2. Expand
            if not node.is_terminal and node.untried_actions is not None and len(node.untried_actions) > 0:
                if node == root and allowed_root_phase != -1:
                    valid_untried = [a for a in node.untried_actions if action_phases.get(a, 0) == allowed_root_phase]
                    if valid_untried:
                        action = valid_untried[-1]
                        node.untried_actions.remove(action)
                    else:
                        action = node.untried_actions.pop()
                else:
                    action = node.untried_actions.pop()
                
                try:
                    new_state = cg_api.search_step(node.search_state.searchId, [action])
                    child = MCTSNode(parent=node, action=action, search_state=new_state)
                except Exception as e:
                    child = MCTSNode(parent=node, action=action, search_state=None)
                
                base_prior = 1.0 / (len(node.children) + len(node.untried_actions) + 1)
                if is_training and node == root and action in root_noise_map:
                    child.prior = (1 - epsilon) * base_prior + epsilon * root_noise_map[action]
                else:
                    child.prior = base_prior
                
                node.children.append(child)
                node = child
            
            # 3. Evaluate
            leaf_player = root.search_state.observation.current.yourIndex
            if not node.is_terminal and node.search_state is not None:
                value = self.evaluator(node.search_state) 
                leaf_player = node.search_state.observation.current.yourIndex
            else:
                value = -1.0 # Terminal or error
                if node.search_state is not None and node.search_state.observation.current.result != -1:
                    res = node.search_state.observation.current.result
                    # result 1=win, 2=loss, 3=draw (for the active player)
                    if res == 1: value = 1.0
                    elif res == 2: value = -1.0
                    else: value = 0.0
                    leaf_player = node.search_state.observation.current.yourIndex
            
            # 4. Backpropagate
            curr = node
            while curr is not None:
                curr.visits += 1
                if curr.parent is not None and curr.parent.search_state is not None:
                    parent_player = curr.parent.search_state.observation.current.yourIndex
                    curr.value += value if parent_player == leaf_player else -value
                else:
                    root_player = root.search_state.observation.current.yourIndex
                    curr.value += value if root_player == leaf_player else -value
                curr = curr.parent
                
        # Return action based on training mode
        best_action_idx = -1
        if root.children:
            if is_training:
                # Sample proportionally to visits with temperature
                import numpy as np
                visits = np.array([c.visits for c in root.children], dtype=np.float32)
                
                if temperature != 1.0 and temperature > 0:
                    visits = visits ** (1.0 / temperature)
                    
                probs = visits / visits.sum()
                best_child = np.random.choice(root.children, p=probs)
                best_action_idx = best_child.action
            else:
                # Greedy most visited
                best_child = max(root.children, key=lambda c: c.visits)
                best_action_idx = best_child.action
        elif root.untried_actions:
            best_action_idx = root.untried_actions[0]
            
        # Clean up memory
        cg_api.search_end()
        
        if best_action_idx >= 0:
            # If the engine allows multiple selections (maxCount > 1), MCTS's single-action
            # edge exploration is insufficient and combinatorial. We return empty to let
            # the Policy network in agent.py handle the combination sampling natively.
            if cg_obs.select.maxCount > 1:
                return []
                
            return [best_action_idx]
            
        return []

