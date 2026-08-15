import os
import json

import csv

_CARD_DB = {}

def _load_card_db():
    global _CARD_DB
    if _CARD_DB:
        return
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        db_path = os.path.join(project_root, "assets", "decks", "EN_Card_Data.csv")
        with open(db_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader) # skip header
            for row in reader:
                if len(row) >= 2:
                    try:
                        _CARD_DB[int(row[0])] = row[1]
                    except ValueError:
                        pass
    except Exception:
        pass

CONTEXT_MAP = {
    0: "MAIN (Main selection)",
    1: "SETUP_ACTIVE_POKEMON (Select Pokémon to put into your Active Spot)",
    2: "SETUP_BENCH_POKEMON (Select Pokémon to put onto your Bench)",
    3: "SWITCH (Select Pokémon to swap with Active Spot)",
    4: "TO_ACTIVE (Select Pokémon to put into Active Spot)",
    5: "TO_BENCH (Select Pokémon to put onto Bench)",
    6: "TO_FIELD (Select Pokémon to put into play)",
    7: "TO_HAND (Select card to add to hand)",
    8: "DISCARD (Select card to discard)",
    9: "TO_DECK (Select card to return to deck)",
    10: "TO_DECK_BOTTOM (Select card to return to bottom of deck)",
    11: "TO_PRIZE (Select card to add to prize)",
    12: "NOT_MOVE (Select card to remain where it is)",
    13: "DAMAGE_COUNTER (Select Pokémon to place damage counters on)",
    14: "DAMAGE_COUNTER_ANY (Select Pokémon to place damage counters on)",
    15: "DAMAGE (Select Pokémon to deal damage)",
    16: "REMOVE_DAMAGE_COUNTER (Select Pokémon to remove damage counters from)",
    17: "HEAL (Select Pokémon to heal)",
    18: "EVOLVES_FROM (Select Pokémon to evolve from)",
    19: "EVOLVES_TO (Select Pokémon to evolve into)",
    20: "DEVOLVE (Select Pokémon to devolve)",
    21: "ATTACH_FROM (Select Pokémon to attach card to)",
    22: "ATTACH_TO (Select card to attach to Pokémon)",
    23: "DETACH_FROM (Select Pokémon to remove card from)",
    24: "LOOK (Select card to look at)",
    25: "EFFECT_TARGET (Select card to apply effect to)",
    26: "DISCARD_ENERGY_CARD (Select Energy card to discard)",
    27: "DISCARD_TOOL_CARD (Select Pokémon tool to trash)",
    28: "SWITCH_ENERGY_CARD (Select energy card to replace)",
    29: "DISCARD_CARD_OR_ATTACHED_CARD (Select card to discard)",
    30: "DISCARD_ENERGY (Select energy to discard)",
    31: "TO_HAND_ENERGY (Select energy to return to hand)",
    32: "TO_DECK_ENERGY (Select energy to return to deck)",
    33: "SWITCH_ENERGY (Select energy to switch)",
    34: "SKILL_ORDER (Select order of effect activation)",
    35: "ATTACK (Select Attack to use)",
    36: "DISABLE_ATTACK (Select Attack to disable)",
    37: "EVOLVE (Select evolution source and target)",
    38: "DRAW_COUNT (Select how many cards to draw)",
    39: "DAMAGE_COUNTER_COUNT (Select how many damage counters to place)",
    40: "REMOVE_DAMAGE_COUNTER_COUNT (Select how many damage counters to remove)",
    41: "IS_FIRST (Would you like to go first?)",
    42: "MULLIGAN (Would you like to redraw the cards?)",
    43: "ACTIVATE (Would you like to activate the effect?)",
    44: "FIRST_EFFECT (Would you like to select the first effect?)",
    45: "MORE_DEVOLVE (Do you want to devolve it further?)",
    46: "COIN_HEAD (Do you want to choose heads?)",
    47: "AFFECT_SPECIAL_CONDITION (Choose special condition to affect)",
    48: "RECOVER_SPECIAL_CONDITION (Choose special condition to recover)"
}

def get_card_name(card):
    if card is None: return "Empty"
    _load_card_db()
    return _CARD_DB.get(card.id, f"Card(id={card.id})")

def render_pokemon(pkmn):
    if pkmn is None: return "[Empty]"
    energies = getattr(pkmn, 'energyCards', [])
    energy_count = len(energies)
    hp = getattr(pkmn, 'hp', 0)
    max_hp = getattr(pkmn, 'maxHp', 0)
    return f'''
    <div style="position: relative; display: inline-block; margin: 2px;">
        <img data-src-id="{pkmn.id}" src="" class="dynamic-img" style="width: 120px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.5);">
        <div style="position: absolute; bottom: 5px; left: 5px; background: rgba(0,0,0,0.8); color: #fff; padding: 3px 6px; border-radius: 4px; font-size: 11px; font-weight: bold; pointer-events: none;">
            {hp}/{max_hp} HP | {energy_count} E
        </div>
    </div>
    '''

def render_board_html(parsed_obs, payload_json="{}") -> str:
    if not parsed_obs or not parsed_obs.current:
        return "<html><body><h3>Waiting for game to start...</h3></body></html>"
        
    my_idx = parsed_obs.current.yourIndex
    opp_idx = 1 - my_idx
    
    my_player = parsed_obs.current.players[my_idx]
    opp_player = parsed_obs.current.players[opp_idx]
    
    unique_ids = set()
    for p in [my_player, opp_player]:
        for zone in [p.hand, p.bench, p.active, getattr(p, 'prize', []), getattr(p, 'discard', []), getattr(p, 'lostZone', [])]:
            if zone:
                for c in zone:
                    if hasattr(c, 'id'): unique_ids.add(c.id)
                    
    stadium_card = parsed_obs.current.stadium[0] if getattr(parsed_obs.current, 'stadium', None) else None
    if stadium_card and hasattr(stadium_card, 'id'):
        unique_ids.add(stadium_card.id)

    import base64
    import os
    local_images_js = []
    for cid in unique_ids:
        img_path = os.path.join("assets", "card_images", f"{cid}.jpeg")
        if os.path.exists(img_path):
            with open(img_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode('utf-8')
                local_images_js.append(f'"{cid}": "data:image/jpeg;base64,{b64}"')
    local_images_json = "{" + ",\n".join(local_images_js) + "}"
    
    opp_bench_html = ''.join([f'<div class="bench-pkmn card-hover" style="background: transparent; padding: 0;" data-card-name="{get_card_name(p)}" data-card-id="{p.id}">{render_pokemon(p)}</div>' for p in opp_player.bench]) if opp_player.bench else 'Empty'
    my_bench_html = ''.join([f'<div class="bench-pkmn card-hover" style="background: transparent; padding: 0;" data-card-name="{get_card_name(p)}" data-card-id="{p.id}">{render_pokemon(p)}</div>' for p in my_player.bench]) if my_player.bench else 'Empty'
    
    my_hand_html = ''.join([
        f'''
        <div class="hand-card card-hover" data-card-name="{get_card_name(c)}" data-card-id="{c.id}" style="background: transparent; padding: 0; position: relative; display: inline-block; margin: 2px;">
            <img data-src-id="{c.id}" src="" class="dynamic-img" style="width: 120px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.5);">
            <div style="position: absolute; top: -8px; right: -8px; background: #007bff; color: white; width: 24px; height: 24px; border-radius: 12px; text-align: center; line-height: 24px; font-size: 14px; font-weight: bold; border: 2px solid #1e1e1e; pointer-events: none; z-index: 10;">
                {i}
            </div>
        </div>
        ''' for i, c in enumerate(my_player.hand)
    ]) if my_player.hand else 'Empty'

    def render_card(c):
        if not c: return "[Empty]"
        return f'''
        <div style="position: relative; display: inline-block; margin: 2px;">
            <img data-src-id="{c.id}" src="" class="dynamic-img" style="width: 120px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.5);">
        </div>
        '''

    stadium_html = f'<div class="card-hover" data-card-name="{get_card_name(stadium_card)}" data-card-id="{stadium_card.id}">{render_card(stadium_card)}</div>' if stadium_card else '<div style="width: 120px; height: 167px; border: 2px dashed #666; border-radius: 5px; display: flex; align-items: center; justify-content: center; color: #666; font-size: 12px; text-transform: uppercase;">Stadium</div>'
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <style>
        body {{ font-family: sans-serif; padding: 20px; background: #1e1e1e; color: #eee; }}
        .board {{ max-width: 1400px; margin: 0 auto; background: #2d2d2d; border-radius: 8px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }}
        .player-area {{ border: 2px solid #444; border-radius: 8px; padding: 15px; margin-bottom: 20px; }}
        .opp-area {{ border-color: #622; }}
        .my-area {{ border-color: #262; }}
        .zone {{ margin: 10px 0; padding: 10px; background: #333; border-radius: 4px; }}
        .zone-title {{ font-weight: bold; color: #aaa; margin-bottom: 5px; text-transform: uppercase; font-size: 0.8em; }}
        .active-pkmn {{ display: inline-block; background: transparent; padding: 5px; border-radius: 4px; border: 4px solid #5af; cursor: help; }}
        .bench-pkmn {{ display: inline-block; background: transparent; padding: 5px 10px; margin: 0 5px 5px 0; border-radius: 4px; font-size: 0.9em; cursor: help; }}
        .hand-card {{ display: inline-block; background: transparent; padding: 5px 10px; margin: 0 5px 5px 0; border-radius: 4px; font-size: 0.9em; cursor: help; }}
        h2 {{ margin-top: 0; color: #fff; border-bottom: 1px solid #555; padding-bottom: 10px; }}
    </style>
</head>
<body>
    <div class="board">
        <div class="player-area opp-area">
            <h2>Opponent (Prizes: {len(opp_player.prize)})</h2>
            <div class="zone">
                <div class="zone-title">Hand Size</div>
                <div>{opp_player.handCount} cards</div>
            </div>
            <div class="zone">
                <div class="zone-title">Bench</div>
                {opp_bench_html}
            </div>
            <div class="zone">
                <div class="zone-title">Active Pokémon</div>
                <div class="active-pkmn card-hover" data-card-name="{get_card_name(opp_player.active[0]) if opp_player.active and opp_player.active[0] else ''}" data-card-id="{opp_player.active[0].id if opp_player.active and opp_player.active[0] else ''}">
                    {render_pokemon(opp_player.active[0]) if opp_player.active and opp_player.active[0] else 'None'}
                </div>
            </div>
        </div>
        
        <div style="display: flex; justify-content: space-between; align-items: center; padding: 10px 0;">
            <div class="zone" style="background: transparent; margin: 0; min-width: 130px; text-align: center;">
                <div class="zone-title" style="color: #622;">Opponent Stadium</div>
                {stadium_html}
            </div>
            <div style="flex-grow: 1; margin: 0 20px;">
                <hr style="border-color: #555;">
            </div>
            <div class="zone" style="background: transparent; margin: 0; min-width: 130px; text-align: center;">
                <div class="zone-title" style="color: #262;">My Stadium</div>
                {stadium_html}
            </div>
        </div>
        
        <div class="player-area my-area">
            <div class="zone">
                <div class="zone-title">Active Pokémon</div>
                <div class="active-pkmn card-hover" data-card-name="{get_card_name(my_player.active[0]) if my_player.active and my_player.active[0] else ''}" data-card-id="{my_player.active[0].id if my_player.active and my_player.active[0] else ''}">
                    {render_pokemon(my_player.active[0]) if my_player.active and my_player.active[0] else 'None'}
                </div>
            </div>
            <div class="zone">
                <div class="zone-title">Bench</div>
                {my_bench_html}
            </div>
            <h2>Me (Prizes: {len(my_player.prize)})</h2>
            <div class="zone">
                <div class="zone-title">Hand</div>
                {my_hand_html}
            </div>
        </div>
    </div>
    
    <script>
        const localImages = {local_images_json};
        
        // Populate all dynamic images
        document.querySelectorAll('.dynamic-img').forEach(img => {{
            const cid = img.getAttribute('data-src-id');
            if (cid && localImages[cid]) {{
                img.src = localImages[cid];
            }} else {{
                img.src = "https://placehold.co/120x167?text=Image+Missing";
            }}
        }});
        
        const tooltip = document.createElement('img');
        tooltip.style.position = 'absolute';
        tooltip.style.display = 'none';
        tooltip.style.zIndex = '1000';
        tooltip.style.width = '360px';
        tooltip.style.borderRadius = '15px';
        tooltip.style.boxShadow = '0 10px 30px rgba(0,0,0,0.6)';
        tooltip.style.pointerEvents = 'none';
        document.body.appendChild(tooltip);

        function updateTooltipPosition(e, target) {{
            const isMyArea = target && target.closest('.my-area') !== null;
            tooltip.style.left = (e.pageX + 20) + 'px';
            if (isMyArea) {{
                tooltip.style.top = (e.pageY - 520) + 'px';
            }} else {{
                tooltip.style.top = (e.pageY + 20) + 'px';
            }}
        }}

        document.addEventListener('mouseover', (e) => {{
            const target = e.target.closest('.card-hover');
            if (!target) {{
                tooltip.style.display = 'none';
                return;
            }}
            
            const cardId = target.getAttribute('data-card-id');
            if (!cardId || cardId === 'undefined' || cardId === '') return;
            
            tooltip.style.display = 'block';
            updateTooltipPosition(e, target);
            
            if (localImages[cardId]) {{
                tooltip.src = localImages[cardId];
            }} else {{
                tooltip.src = "https://placehold.co/240x335?text=Image+Missing";
            }}
        }});
        
        document.addEventListener('mousemove', (e) => {{
            if (tooltip.style.display === 'block') {{
                const target = e.target.closest('.card-hover');
                updateTooltipPosition(e, target);
            }}
        }});
    </script>
</body>
</html>
    """
    return html

def format_option(opt, parsed_obs) -> str:
    my_idx = parsed_obs.current.yourIndex if parsed_obs and parsed_obs.current else 0
    my_player = parsed_obs.current.players[my_idx] if parsed_obs and parsed_obs.current else None
    
    def get_hand_card(index):
        if not my_player or not my_player.hand: return f"Card[{index}]"
        if len(my_player.hand) > index:
            return get_card_name(my_player.hand[index])
        return f"Card[{index}]"
        
    t = getattr(opt, "type", opt.get("type", 0) if isinstance(opt, dict) else 0)
    kw = getattr(opt, "kwargs", opt)
    if isinstance(kw, dict):
        pass
    else:
        kw = {}
        
    if t == 7: # PLAY
        idx = kw.get("index", 0)
        return f"PLAY {get_hand_card(idx)} from hand"
    elif t == 8: # ATTACH
        idx = kw.get("index", 0)
        p_idx = kw.get("inPlayIndex", 0)
        p_area = kw.get("inPlayArea", 4)
        target = "Active" if p_area == 4 else f"Bench[{p_idx}]"
        return f"ATTACH {get_hand_card(idx)} to {target}"
    elif t == 9: # EVOLVE
        idx = kw.get("index", 0)
        p_idx = kw.get("inPlayIndex", 0)
        p_area = kw.get("inPlayArea", 4)
        target = "Active" if p_area == 4 else f"Bench[{p_idx}]"
        return f"EVOLVE {target} using {get_hand_card(idx)}"
    elif t == 13: # ATTACK
        att_id = kw.get("attackId", 0)
        return f"ATTACK (Index {att_id})"
    elif t == 14: # END
        return f"END TURN"
    elif t == 12: # RETREAT
        idx = kw.get("benchIndex", 0)
        return f"RETREAT to Bench[{idx}]"
    elif t == 10: # ABILITY
        idx = kw.get("index", 0)
        p_idx = kw.get("inPlayIndex", 0)
        p_area = kw.get("inPlayArea", 4)
        target = "Active" if p_area == 4 else f"Bench[{p_idx}]"
        return f"USE ABILITY of {target} (Ability Index {idx})"
    elif t == 3: # CARD
        area = kw.get("area", 0)
        idx = kw.get("index", 0)
        p_idx = kw.get("playerIndex", 0)
        if area == 2:
            return f"SELECT HAND CARD: {get_hand_card(idx)}"
        elif area == 4:
            return f"SELECT ACTIVE POKEMON (Player {p_idx})"
        elif area == 5:
            return f"SELECT BENCH POKEMON [{idx}] (Player {p_idx})"
        else:
            return f"SELECT CARD (Area={area}, Idx={idx})"
    elif t == 0: # NUMBER
        return f"SELECT NUMBER: {kw.get('number', 0)}"
    elif t == 1:
        return f"SELECT YES"
    elif t == 2:
        return f"SELECT NO"
    elif t == 11: # DISCARD
        return f"DISCARD {get_hand_card(kw.get('index', 0))}"
    elif t == 15: # SKILL
        return f"USE SKILL (Serial: {kw.get('serial', 0)})"
    elif t == 16: # SPECIAL_CONDITION
        return f"SELECT CONDITION (Type: {kw.get('specialConditionType', 0)})"
    elif t in (4, 5, 6): # TOOL_CARD, ENERGY_CARD, ENERGY
        area = kw.get("area", 0)
        idx = kw.get("index", 0)
        sub_idx = kw.get("toolIndex", kw.get("energyIndex", 0))
        target = "Active" if area == 4 else f"Bench[{idx}]"
        sub_type = "Tool" if t == 4 else "Energy"
        return f"SELECT {sub_type} [{sub_idx}] on {target}"
        
    return f"Option(type={t}, kwargs={kw})"
