import torch
import torch.nn as nn
from typing import Tuple
from src.core.parser import Observation, Pokemon

MAX_CARD_ID = 2000

class StateEncoder:
    """
    Serializes our Observation dataclass into a fixed size tensor.
    Encodes exact card counts for Hand and Discard (Bag-of-Words) to allow
    the neural network to recognize specific toolbox cards and resources.
    """
    def __init__(self):
        # 2000 (Hand) + 2000 (Discard) + 2000 (Opp Discard) + 160 (Board)
        self.feature_size = 6160
        
    def encode_pokemon(self, p: Pokemon) -> list[float]:
        if not p:
            return [0.0] * 12
        return [
            1.0, # exists
            float(p.id) / MAX_CARD_ID, # normalized id
            float(p.hp) / 300.0,
            float(p.maxHp) / 300.0,
            1.0 if p.appearThisTurn else 0.0,
            len(p.energies) / 10.0,
            len(p.tools) / 2.0,
            len(p.preEvolution) / 2.0,
            0.0, 0.0, 0.0, 0.0 # padding
        ]

    def encode(self, obs: Observation) -> torch.Tensor:
        if not obs.current:
            return torch.zeros(self.feature_size)
            
        me_idx = obs.current.yourIndex
        me = obs.current.players[me_idx]
        opp = obs.current.players[1 - me_idx]
        
        # 1. Bag of Words for Hand and Discard (Crucial for Toolbox Decks)
        hand_bow = [0.0] * MAX_CARD_ID
        if me.hand:
            for c in me.hand:
                if 0 <= c.id < MAX_CARD_ID:
                    hand_bow[c.id] += 1.0
                    
        my_discard_bow = [0.0] * MAX_CARD_ID
        for c in me.discard:
            if 0 <= c.id < MAX_CARD_ID:
                my_discard_bow[c.id] += 1.0
                
        opp_discard_bow = [0.0] * MAX_CARD_ID
        for c in opp.discard:
            if 0 <= c.id < MAX_CARD_ID:
                opp_discard_bow[c.id] += 1.0

        # 2. Game Board Features
        board = []
        board.append(obs.current.turn / 20.0)
        board.append(1.0 if obs.current.supporterPlayed else 0.0)
        board.append(1.0 if obs.current.stadiumPlayed else 0.0)
        board.append(1.0 if obs.current.energyAttached else 0.0)
        board.append(1.0 if obs.current.retreated else 0.0)
        
        # Player counts
        board.extend([
            me.deckCount / 60.0,
            me.handCount / 20.0,
            len(me.prize) / 6.0,
            1.0 if me.poisoned else 0.0,
            1.0 if me.burned else 0.0,
            1.0 if me.asleep else 0.0,
            1.0 if me.paralyzed else 0.0,
            1.0 if me.confused else 0.0
        ])
        
        # Opponent counts
        board.extend([
            opp.deckCount / 60.0,
            opp.handCount / 20.0,
            len(opp.prize) / 6.0,
            1.0 if opp.poisoned else 0.0,
            1.0 if opp.burned else 0.0,
            1.0 if opp.asleep else 0.0,
            1.0 if opp.paralyzed else 0.0,
            1.0 if opp.confused else 0.0
        ])
        
        # Active Pokemon
        board.extend(self.encode_pokemon(me.active[0] if me.active else None))
        board.extend(self.encode_pokemon(opp.active[0] if opp.active else None))
        
        # Bench Pokemon (max 5)
        for i in range(5):
            p = me.bench[i] if i < len(me.bench) else None
            board.extend(self.encode_pokemon(p))
        for i in range(5):
            p = opp.bench[i] if i < len(opp.bench) else None
            board.extend(self.encode_pokemon(p))
            
        # Pad board to exactly 160 dims
        while len(board) < 160:
            board.append(0.0)
            
        # Combine all features
        features = hand_bow + my_discard_bow + opp_discard_bow + board[:160]
        return torch.tensor(features, dtype=torch.float32)

class ResidualBlock(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.fc1 = nn.Linear(size, size)
        self.ln1 = nn.LayerNorm(size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(size, size)
        self.ln2 = nn.LayerNorm(size)
        self.relu2 = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.fc1(x)
        out = self.ln1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.ln2(out)
        out += identity
        out = self.relu2(out)
        return out

class BaseNetwork(nn.Module):
    """
    Deep Residual Network Architecture for the AI agent.
    Scales to millions of parameters while fitting in Kaggle constraints.
    """
    def __init__(self, input_size: int = 6160, hidden_size: int = 512, num_blocks: int = 4, policy_size: int = 512):
        super(BaseNetwork, self).__init__()
        
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU()
        )
        
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_size) for _ in range(num_blocks)]
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size / 2)),
            nn.LayerNorm(int(hidden_size / 2)),
            nn.ReLU(),
            nn.Linear(int(hidden_size / 2), 1),
            nn.Tanh() # Squeeze to [-1, 1]
        )
        
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, policy_size)
        )
        
    def forward(self, state_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.input_layer(state_tensor)
        features = self.res_blocks(x)
        value = self.value_head(features)
        policy = self.policy_head(features)
        return value, policy
