from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

@dataclass
class Card:
    id: int
    serial: int
    playerIndex: int

@dataclass
class EnergyCard(Card):
    pass

@dataclass
class Pokemon(Card):
    hp: int
    maxHp: int
    appearThisTurn: bool
    energies: List[int] = field(default_factory=list)
    energyCards: List[EnergyCard] = field(default_factory=list)
    tools: List[Card] = field(default_factory=list)
    preEvolution: List[Card] = field(default_factory=list)

@dataclass
class PlayerState:
    active: List[Pokemon] = field(default_factory=list)
    bench: List[Pokemon] = field(default_factory=list)
    benchMax: int = 5
    deckCount: int = 0
    discard: List[Card] = field(default_factory=list)
    prize: List[Optional[Card]] = field(default_factory=list)
    handCount: int = 0
    hand: Optional[List[Card]] = None
    poisoned: bool = False
    burned: bool = False
    asleep: bool = False
    paralyzed: bool = False
    confused: bool = False

@dataclass
class CurrentState:
    turn: int = 0
    turnActionCount: int = 0
    yourIndex: int = 0
    firstPlayer: int = -1
    supporterPlayed: bool = False
    stadiumPlayed: bool = False
    energyAttached: bool = False
    retreated: bool = False
    result: int = -1
    stadium: List[Card] = field(default_factory=list)
    players: List[PlayerState] = field(default_factory=list)

@dataclass
class Option:
    type: int
    kwargs: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SelectState:
    type: int = 0
    minCount: int = 1
    maxCount: int = 1
    option: List[Option] = field(default_factory=list)

@dataclass
class Observation:
    step: int = 0
    current: Optional[CurrentState] = None
    select: Optional[SelectState] = None

def _parse_card(data: dict) -> Card:
    return Card(
        id=data.get("id", 0),
        serial=data.get("serial", 0),
        playerIndex=data.get("playerIndex", 0)
    )

def _parse_pokemon(data: dict) -> Pokemon:
    if data is None:
        return None
    return Pokemon(
        id=data.get("id", 0),
        serial=data.get("serial", 0),
        playerIndex=data.get("playerIndex", 0),
        hp=data.get("hp", 0),
        maxHp=data.get("maxHp", 0),
        appearThisTurn=data.get("appearThisTurn", False),
        energies=data.get("energies", []),
        energyCards=[_parse_card(c) for c in data.get("energyCards", [])],
        tools=[_parse_card(c) for c in data.get("tools", [])],
        preEvolution=[_parse_card(c) for c in data.get("preEvolution", [])]
    )

def _parse_player(data: dict) -> PlayerState:
    return PlayerState(
        active=[_parse_pokemon(p) for p in data.get("active", [])],
        bench=[_parse_pokemon(p) for p in data.get("bench", [])],
        benchMax=data.get("benchMax", 5),
        deckCount=data.get("deckCount", 0),
        discard=[_parse_card(c) for c in data.get("discard", []) if c],
        prize=[_parse_card(c) if c else None for c in data.get("prize", [])],
        handCount=data.get("handCount", 0),
        hand=[_parse_card(c) for c in data.get("hand")] if data.get("hand") is not None else None,
        poisoned=data.get("poisoned", False),
        burned=data.get("burned", False),
        asleep=data.get("asleep", False),
        paralyzed=data.get("paralyzed", False),
        confused=data.get("confused", False)
    )

def parse_observation(obs_dict: dict) -> Observation:
    obs = Observation(step=obs_dict.get("step", 0))
    
    if "current" in obs_dict and obs_dict["current"]:
        c = obs_dict["current"]
        obs.current = CurrentState(
            turn=c.get("turn", 0),
            turnActionCount=c.get("turnActionCount", 0),
            yourIndex=c.get("yourIndex", 0),
            firstPlayer=c.get("firstPlayer", -1),
            supporterPlayed=c.get("supporterPlayed", False),
            stadiumPlayed=c.get("stadiumPlayed", False),
            energyAttached=c.get("energyAttached", False),
            retreated=c.get("retreated", False),
            result=c.get("result", -1),
            stadium=[_parse_card(x) for x in c.get("stadium", [])],
            players=[_parse_player(p) for p in c.get("players", [])]
        )
        
    if "select" in obs_dict and obs_dict["select"]:
        s = obs_dict["select"]
        options = []
        for opt_dict in s.get("option", []):
            opt_type = opt_dict.get("type", 0)
            kwargs = {k: v for k, v in opt_dict.items() if k != "type"}
            options.append(Option(type=opt_type, kwargs=kwargs))
            
        obs.select = SelectState(
            type=s.get("type", 0),
            minCount=s.get("minCount", 1),
            maxCount=s.get("maxCount", 1),
            option=options
        )
        
    return obs
