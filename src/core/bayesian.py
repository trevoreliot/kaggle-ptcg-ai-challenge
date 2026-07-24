import os
import numpy as np
from src.core.parser import Observation

class BayesianTracker:
    def __init__(self, num_archetypes: int = 4):
        # We track archetypes in this fixed order:
        self.archetypes = ["aggro", "control", "combo", "versatile"]
        self.num_archetypes = len(self.archetypes)
        
        # Prior starts completely uniform.
        self.prior = np.ones(self.num_archetypes) / self.num_archetypes
        
        # Load the Likelihood Matrix
        matrix_path = os.path.join("assets", "prob", "likelihood_matrix.npy")
        if os.path.exists(matrix_path):
            self.likelihood_matrix = np.load(matrix_path)
        else:
            # Fallback uniform likelihoods if matrix hasn't been generated
            self.likelihood_matrix = np.ones((2000, self.num_archetypes)) / self.num_archetypes
            
        # Keep track of card serial numbers we have already evaluated
        self.seen_serials = set()
        
    def _get_opponent_cards(self, obs: Observation) -> list:
        if not obs.current:
            return []
            
        opponent_idx = 1 - obs.current.yourIndex
        opp_state = obs.current.players[opponent_idx]
        
        cards = []
        # Gather all visible opponent cards
        for pkmn in opp_state.active:
            if pkmn is not None:
                cards.append(pkmn)
                cards.extend(pkmn.energyCards)
                cards.extend(pkmn.tools)
            
        for pkmn in opp_state.bench:
            if pkmn is not None:
                cards.append(pkmn)
                cards.extend(pkmn.energyCards)
                cards.extend(pkmn.tools)
            
        cards.extend(opp_state.discard)
        
        # Technically stadium could belong to either player, but it's a visible card.
        # We'll just include it if it's there.
        cards.extend(obs.current.stadium)
        
        return cards
        
    def update(self, obs: Observation):
        """
        Scans the board and updates the posterior probability for the opponent's archetype
        using Bayes Theorem for any newly revealed cards.
        """
        cards = self._get_opponent_cards(obs)
        
        new_cards = [c for c in cards if c.serial not in self.seen_serials]
        if not new_cards:
            return
            
        for card in new_cards:
            self.seen_serials.add(card.serial)
            card_id = card.id
            
            if card_id < self.likelihood_matrix.shape[0]:
                likelihoods = self.likelihood_matrix[card_id]
                
                # Bayes Rule: Posterior = Prior * Likelihood
                posterior = self.prior * likelihoods
                
                # Normalize so probabilities sum to 1
                evidence = np.sum(posterior)
                if evidence > 0:
                    self.prior = posterior / evidence
                    
    def max_confidence(self) -> float:
        return np.max(self.prior)
        
    def best_archetype(self) -> str:
        idx = np.argmax(self.prior)
        return self.archetypes[idx]
