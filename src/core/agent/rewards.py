import os
import json

# Defaults
REWARD_CONFIG = {
    "r_prize_taken": 0.5,
    "r_prize_lost": -0.2,
    "r_deck_out": -2.0,
    "r_energy_attach": 0.05,
    "r_evolution": 0.10,
    "r_damage_dealt_per_10": 0.01
}

try:
    _reward_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "assets", "reward", "reward_shaping.json"))
    if os.path.exists(_reward_path):
        with open(_reward_path, "r") as _f:
            REWARD_CONFIG.update(json.load(_f))
except Exception as e:
    print(f"Warning: Could not load reward_shaping.json. Using defaults. ({e})")


def calculate_step_reward(parsed_obs, tracker):
    """
    Calculate dense step rewards and update the state tracker for the current turn.
    Returns the accumulated step_reward.
    """
    step_reward = 0.0
    
    me_idx = parsed_obs.current.yourIndex
    my_player = parsed_obs.current.players[me_idx]
    opp_player = parsed_obs.current.players[1 - me_idx]
    my_prizes = len([p for p in my_player.prize if p is not None])
    opp_prizes = len([p for p in opp_player.prize if p is not None])
    my_deck = my_player.deckCount
    
    my_all_poke = []
    if my_player.active: my_all_poke.extend(my_player.active)
    if my_player.bench: my_all_poke.extend(my_player.bench)
    my_all_poke = [p for p in my_all_poke if p is not None]
        
    opp_all_poke = []
    if opp_player.active: opp_all_poke.extend(opp_player.active)
    if opp_player.bench: opp_all_poke.extend(opp_player.bench)
    opp_all_poke = [p for p in opp_all_poke if p is not None]
    
    current_energies = sum(len(p.energyCards) for p in my_all_poke)
    current_evolutions = sum(len(p.preEvolution) for p in my_all_poke)
    current_opp_damage = sum((p.maxHp - p.hp) for p in opp_all_poke)
    current_my_damage = sum((p.maxHp - p.hp) for p in my_all_poke)
    current_my_active_serial = my_player.active[0].serial if my_player.active and my_player.active[0] is not None else None
    
    has_fezandipiti = any(p.id == 140 for p in my_all_poke)
    has_ursulana = any(p.id == 44 for p in my_all_poke)
    
    # Calculate deltas if state is initialized
    if tracker["initialized"]:
        prizes_taken = tracker["opp_prizes"] - opp_prizes
        prizes_lost = tracker["my_prizes"] - my_prizes
        
        # 1. Prize conditions
        if prizes_taken > 0:
            step_reward += REWARD_CONFIG["r_prize_taken"] * prizes_taken
            if tracker.get("boss_played_turn") == parsed_obs.current.turn:
                step_reward += REWARD_CONFIG.get("r_boss_ko_bonus", 0.5) * prizes_taken
        if prizes_lost > 0:
            step_reward += REWARD_CONFIG["r_prize_lost"] * prizes_lost
            if tracker.get("had_fezandipiti", False) and not has_fezandipiti:
                step_reward += REWARD_CONFIG.get("r_prize_lost_fezandipiti_penalty", -0.5) * prizes_lost
                
        if my_deck == 0 and tracker["my_deck"] > 0:
            step_reward += REWARD_CONFIG["r_deck_out"]
            
        # Card Play Reward
        if tracker.get("last_options") is not None:
            for act in tracker["last_actions"]:
                if act < len(tracker["last_options"]):
                    opt = tracker["last_options"][act]
                    if opt.type == 4: # OptionType.PLAY
                        step_reward += REWARD_CONFIG.get("r_play_trainer", 0.05)
                        if opt.area == 2 and tracker.get("last_hand") is not None:
                            try:
                                played_card = tracker["last_hand"][opt.index]
                                if played_card.id in (1182, 1088, 1218):
                                    tracker["boss_played_turn"] = parsed_obs.current.turn
                                elif played_card.id == 1251:
                                    step_reward += REWARD_CONFIG.get("r_play_stadium", 0.05)
                            except:
                                pass
            
        # 2. Dense Setup Rewards
        energy_delta = current_energies - tracker["my_energies"]
        if energy_delta > 0:
            step_reward += REWARD_CONFIG["r_energy_attach"] * energy_delta
            
        evo_delta = current_evolutions - tracker["my_evolutions"]
        if evo_delta > 0:
            step_reward += REWARD_CONFIG["r_evolution"] * evo_delta
            
        # Retreating Reward
        if current_my_active_serial is not None and tracker.get("my_active_serial") is not None:
            if current_my_active_serial != tracker["my_active_serial"] and prizes_lost == 0:
                step_reward += REWARD_CONFIG.get("r_retreat", 0.10)
                
        # Healing Reward
        my_damage_delta = current_my_damage - tracker.get("my_damage", 0)
        if my_damage_delta < 0 and prizes_lost == 0:
            step_reward += REWARD_CONFIG.get("r_healing_per_10", 0.02) * (-my_damage_delta / 10.0)
            
        # 3. Dense Attack Rewards
        damage_delta = current_opp_damage - tracker["opp_damage"]
        if damage_delta > 0:
            base_dmg_reward = REWARD_CONFIG["r_damage_dealt_per_10"] * (damage_delta / 10.0)
            if has_ursulana and my_player.active and my_player.active[0] is not None and my_player.active[0].id == 44:
                base_dmg_reward += REWARD_CONFIG.get("r_ursulana_attack_bonus_per_prize", 0.02) * (6 - opp_prizes)
            step_reward += base_dmg_reward
            
    tracker["my_prizes"] = my_prizes
    tracker["opp_prizes"] = opp_prizes
    tracker["my_deck"] = my_deck
    tracker["my_energies"] = current_energies
    tracker["my_evolutions"] = current_evolutions
    tracker["opp_damage"] = current_opp_damage
    tracker["my_damage"] = current_my_damage
    tracker["my_active_serial"] = current_my_active_serial
    tracker["had_fezandipiti"] = has_fezandipiti
    tracker["initialized"] = True
    
    return step_reward
