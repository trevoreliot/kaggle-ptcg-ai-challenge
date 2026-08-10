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

# Lookup table for max attack damage of common meta attackers
ATTACK_DAMAGE_LOOKUP = {
    44: 240,    # Bloodmoon Ursaluna ex
    140: 100,   # Fezandipiti ex
    400: 30,    # Team Rocket's Tarountula
    401: 30,    # Team Rocket's Spidops
    414: 60,    # Team Rocket's Articuno
    431: 160,   # Team Rocket's Mewtwo ex
    432: 70,    # Team Rocket's Wobbuffet
}

TR_POKEMON_IDS = [400, 401, 414, 431, 432, 434]

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
    
    current_tr_count = sum(1 for p in my_all_poke if p.id in TR_POKEMON_IDS)
    
    my_all_poke = [p for p in my_player.active + my_player.bench if p is not None]
    current_energies = sum(len(p.energyCards) for p in my_all_poke)
    current_mewtwo_energies = sum(len(p.energyCards) for p in my_all_poke if p.id == 431)
    current_ursaluna_energies = sum(len(p.energyCards) for p in my_all_poke if p.id == 44)
    current_articuno_energies = sum(len(p.energyCards) for p in my_all_poke if p.id == 414)
    current_evolutions = sum(1 for p in my_all_poke if p.preEvolution)
    current_opp_damage = sum((p.maxHp - p.hp) for p in opp_all_poke)
    current_my_damage = sum((p.maxHp - p.hp) for p in my_all_poke)
    current_my_active_serial = my_player.active[0].serial if my_player.active and my_player.active[0] is not None else None
    
    current_pivots = sum(1 for p in my_all_poke if p.id in (140, 414, 400) for t in p.tools if t.id == 1157)
    
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
                bonus = REWARD_CONFIG.get("r_boss_ko_bonus", 0.5) * prizes_taken
                print(f"\n[REWARD] 🎯 BOSS'S ORDERS KO BONUS GRANTED! (+{bonus:.2f}) 🎯\n")
                step_reward += bonus
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
                                    
                                    # Sniper Bonus Check
                                    active_pkmn = my_player.active[0] if my_player.active and my_player.active[0] is not None else None
                                    if active_pkmn:
                                        attack_dmg = ATTACK_DAMAGE_LOOKUP.get(active_pkmn.id, 100)
                                        valid_targets = [p for p in opp_player.bench if p is not None and p.hp <= attack_dmg]
                                        if valid_targets:
                                            snipe_bonus = REWARD_CONFIG.get("r_boss_snipe_bonus", 1.0)
                                            print(f"\n[REWARD] 🎯 SNIPE OPPORTUNITY IDENTIFIED! (+{snipe_bonus:.2f}) 🎯\n")
                                            step_reward += snipe_bonus
                                elif played_card.id == 1251:
                                    step_reward += REWARD_CONFIG.get("r_play_stadium", 0.05)
                                elif played_card.id == 1218: # Cyrano
                                    cyrano_bonus = REWARD_CONFIG.get("r_play_cyrano", 0.25)
                                    step_reward += cyrano_bonus
                                    print(f"\n[REWARD] 🎭 CYRANO PLAYED (TR EX SEARCH)! (+{cyrano_bonus:.2f}) 🎭\n")
                                elif played_card.id == 1105: # Buddy-Buddy Poffin
                                    poffin_bonus = REWARD_CONFIG.get("r_play_poffin", 0.15)
                                    step_reward += poffin_bonus
                                    print(f"\n[REWARD] 🍞 BUDDY-BUDDY POFFIN PLAYED (BASIC TR SEARCH)! (+{poffin_bonus:.2f}) 🍞\n")
                                elif played_card.id in (1121, 1102, 1205, 1132, 1134):
                                    search_bonus = REWARD_CONFIG.get("r_play_search_card", 0.10)
                                    step_reward += search_bonus
                                    print(f"\n[REWARD] 🔍 SEARCH CARD PLAYED! (+{search_bonus:.2f}) 🔍\n")
                                elif played_card.id == 1120:
                                    valid_targets = [
                                        p for p in opp_all_poke
                                        if p is not None and len(p.energyCards) > 0 and (p.maxHp >= 200 or len(p.preEvolution) > 0)
                                    ]
                                    if valid_targets:
                                        hammer_bonus = REWARD_CONFIG.get("r_hammer_tempo_bonus", 0.20)
                                        step_reward += hammer_bonus
                                        print(f"\n[REWARD] 🔨 HAMMER DISRUPTION TARGET SPOTTED! (+{hammer_bonus:.2f}) 🔨\n")
                            except:
                                pass
                    elif opt.type == 14: # OptionType.END
                        pass_penalty = REWARD_CONFIG.get("r_pass_turn", -0.05)
                        step_reward += pass_penalty
                        print(f"\n[REWARD] 💤 TURN PASSED PREMATURELY! ({pass_penalty:.2f}) 💤\n")
                        
                        if any(o.type == 13 for o in tracker["last_options"]):
                            missed_attack_penalty = REWARD_CONFIG.get("r_missed_attack_penalty", -1.0)
                            step_reward += missed_attack_penalty
                            print(f"\n[REWARD] ❌ PASSED TURN WITH ATTACK AVAILABLE! ({missed_attack_penalty:.2f}) ❌\n")
                        
                        benched_pokemon = [p for p in my_player.bench if p is not None]
                        if len(benched_pokemon) == 0:
                            empty_bench_penalty = REWARD_CONFIG.get("r_empty_bench_penalty", -0.50)
                            step_reward += empty_bench_penalty
                            print(f"\n[REWARD] 🚨 TURN ENDED WITH EMPTY BENCH! ({empty_bench_penalty:.2f}) 🚨\n")
                            
                    elif opt.type == 13: # OptionType.ATTACK
                        attack_reward = REWARD_CONFIG.get("r_choose_attack", 0.25)
                        step_reward += attack_reward
                        print(f"\n[REWARD] ⚔️ ATTACK DECLARED! ({attack_reward:.2f}) ⚔️\n")
            
        # 2. Dense Setup Rewards
        energy_delta = current_energies - tracker["my_energies"]
        if energy_delta > 0:
            step_reward += REWARD_CONFIG["r_energy_attach"] * energy_delta
            
        mewtwo_energy_delta = current_mewtwo_energies - tracker.get("my_mewtwo_energies", 0)
        if mewtwo_energy_delta > 0:
            heavy_reward = REWARD_CONFIG.get("r_energy_attach_heavy", 0.10) * mewtwo_energy_delta
            step_reward += heavy_reward
            print(f"\n[REWARD] ⚡ MEWTWO EX CHARGED! (+{heavy_reward:.2f}) ⚡\n")
            
        ursaluna_energy_delta = current_ursaluna_energies - tracker.get("my_ursaluna_energies", 0)
        if ursaluna_energy_delta > 0:
            heavy_reward = REWARD_CONFIG.get("r_energy_attach_heavy", 0.10) * ursaluna_energy_delta
            step_reward += heavy_reward
            print(f"\n[REWARD] ⚡ URSALUNA BLOODMOON EX CHARGED! (+{heavy_reward:.2f}) ⚡\n")
            
        articuno_energy_delta = current_articuno_energies - tracker.get("my_articuno_energies", 0)
        if articuno_energy_delta > 0:
            articuno_reward = REWARD_CONFIG.get("r_energy_attach_articuno", 0.10) * articuno_energy_delta
            step_reward += articuno_reward
            print(f"\n[REWARD] ❄️ TEAM ROCKET'S ARTICUNO CHARGED! (+{articuno_reward:.2f}) ❄️\n")
            
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
            
        # TR Synergy & Mewtwo Jackpot
        tr_delta = current_tr_count - tracker.get("tr_pokemon_count", 0)
        if tr_delta > 0:
            tr_bonus = REWARD_CONFIG.get("r_team_rocket_board_presence", 0.20) * tr_delta
            print(f"\n[REWARD] 🚀 TEAM ROCKET SWARM INCREASED! (+{tr_bonus:.2f}) 🚀\n")
            step_reward += tr_bonus
            
        if my_player.active and my_player.active[0] is not None and my_player.active[0].id == 431:
            if current_tr_count >= 4 and not tracker.get("mewtwo_jackpot_claimed", False):
                jackpot = REWARD_CONFIG.get("r_mewtwo_jackpot", 3.0)
                print(f"\n[REWARD] 🔮 MEWTWO EX POWER SAVER UNLOCKED! JACKPOT! (+{jackpot:.2f}) 🔮\n")
                step_reward += jackpot
                tracker["mewtwo_jackpot_claimed"] = True
                
        if my_player.active and my_player.active[0] is not None and my_player.active[0].id == 44:
            if opp_prizes <= 3 and not tracker.get("ursaluna_active_reward_claimed", False):
                ursa_jackpot = REWARD_CONFIG.get("r_ursaluna_active_jackpot", 1.5)
                print(f"\n[REWARD] 🐻 URSALUNA BLOODMOON ACTIVATED IN ENDGAME! (+{ursa_jackpot:.2f}) 🐻\n")
                step_reward += ursa_jackpot
                tracker["ursaluna_active_reward_claimed"] = True
                
        pivot_delta = current_pivots - tracker.get("rescue_board_pivots", 0)
        if pivot_delta > 0:
            pivot_bonus = REWARD_CONFIG.get("r_rescue_board_pivot", 0.15) * pivot_delta
            print(f"\n[REWARD] 🛹 FREE PIVOT CREATED! (+{pivot_bonus:.2f}) 🛹\n")
            step_reward += pivot_bonus
            
        # 3. Dense Attack Rewards
        damage_delta = current_opp_damage - tracker["opp_damage"]
        if damage_delta > 0:
            base_dmg_reward = REWARD_CONFIG["r_damage_dealt_per_10"] * (damage_delta / 10.0)
            if has_ursulana and my_player.active and my_player.active[0] is not None and my_player.active[0].id == 44:
                bonus = REWARD_CONFIG.get("r_ursulana_attack_bonus_per_prize", 0.02) * (6 - opp_prizes)
                if bonus > 0:
                    print(f"\n[REWARD] 🐻 URSALUNA BLOODMOON ATTACK BONUS GRANTED! (+{bonus:.2f}) 🐻\n")
                base_dmg_reward += bonus
                
            if my_player.active and my_player.active[0] is not None and my_player.active[0].id == 431:
                kicker = REWARD_CONFIG.get("r_mewtwo_attack_kicker", 1.0)
                if kicker > 0:
                    print(f"\n[REWARD] 💥 MEWTWO EX BLAST KICKER! (+{kicker:.2f}) 💥\n")
                base_dmg_reward += kicker
                
            step_reward += base_dmg_reward
            
    tracker["my_prizes"] = my_prizes
    tracker["opp_prizes"] = opp_prizes
    tracker["my_deck"] = my_deck
    tracker["my_energies"] = current_energies
    tracker["my_mewtwo_energies"] = current_mewtwo_energies
    tracker["my_ursaluna_energies"] = current_ursaluna_energies
    tracker["my_articuno_energies"] = current_articuno_energies
    tracker["my_evolutions"] = current_evolutions
    tracker["opp_damage"] = current_opp_damage
    tracker["my_damage"] = current_my_damage
    tracker["my_active_serial"] = current_my_active_serial
    tracker["had_fezandipiti"] = has_fezandipiti
    tracker["tr_pokemon_count"] = current_tr_count
    tracker["rescue_board_pivots"] = current_pivots
    # mewtwo_jackpot_claimed is updated in-place above
    tracker["initialized"] = True
    
    return step_reward
