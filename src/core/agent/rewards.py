import os
import json

def _track_reward(tracker, reward_name, value, count=1):
    if "reward_metrics" not in tracker:
        tracker["reward_metrics"] = {}
    if reward_name not in tracker["reward_metrics"]:
        tracker["reward_metrics"][reward_name] = {"count": 0, "total": 0.0}
    tracker["reward_metrics"][reward_name]["count"] += count
    tracker["reward_metrics"][reward_name]["total"] += value
    return value


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
    _reward_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "assets", "reward", "reward_shaping.json"))
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
    my_prizes = len(my_player.prize)
    opp_prizes = len(opp_player.prize)
    my_deck = my_player.deckCount
    opp_deck = opp_player.deckCount
    
    my_all_poke = []
    if my_player.active: my_all_poke.extend(my_player.active)
    if my_player.bench: my_all_poke.extend(my_player.bench)
    my_all_poke = [p for p in my_all_poke if p is not None]
        
    opp_all_poke = []
    if opp_player.active: opp_all_poke.extend(opp_player.active)
    if opp_player.bench: opp_all_poke.extend(opp_player.bench)
    opp_all_poke = [p for p in opp_all_poke if p is not None]
    
    current_tr_count = sum(1 for p in my_all_poke if p.id in TR_POKEMON_IDS)
    current_energies = sum(len(p.energyCards) for p in my_all_poke)
    current_mewtwo_energies = sum(len(p.energyCards) for p in my_all_poke if p.id == 431)
    current_ursaluna_energies = sum(len(p.energyCards) for p in my_all_poke if p.id == 44)
    current_articuno_energies = sum(len(p.energyCards) for p in my_all_poke if p.id == 414)
    current_high_power_energies = sum(len(p.energyCards) for p in my_all_poke if p.id in (431, 140, 414, 44))
    
    current_fully_charged = 0
    current_over_attached_energies = 0
    for p in my_all_poke:
        val = sum(2 if getattr(e, 'id', None) == 15 else 1 for e in getattr(p, 'energyCards', []))
        if p.id in (431, 140, 414) and val >= 3:
            current_fully_charged += 1
        elif p.id == 44 and val >= max(0, opp_prizes - 1):
            current_fully_charged += 1
            
        if p.id == 44:
            max_cost = max(0, opp_prizes - 1)
        else:
            max_cost = {431: 3, 414: 3, 140: 3, 432: 3, 401: 2, 434: 2, 400: 1}.get(p.id, 3)
            
        # If Mewtwo EX is on board, allow infinite energy on TR bench pokemon for Erasure Ball!
        if any(poke.id == 431 for poke in my_all_poke) and p.id in TR_POKEMON_IDS and p.id != 431:
            max_cost = 999
            
        current_over_attached_energies += max(0, val - max_cost)
            
    current_mewtwo_tr_energies = sum(1 for p in my_all_poke if p.id == 431 for e in p.energyCards if e.id == 15)
    current_articuno_tr_energies = sum(1 for p in my_all_poke if p.id == 414 for e in p.energyCards if e.id == 15)
    current_evolutions = sum(1 for p in my_all_poke if p.preEvolution)
    current_opp_damage = sum((p.maxHp - p.hp) for p in opp_all_poke)
    current_my_damage = sum((p.maxHp - p.hp) for p in my_all_poke)
    current_high_power_damage = sum((p.maxHp - p.hp) for p in my_all_poke if p.id in (431, 140, 414, 44))
    current_my_active_serial = my_player.active[0].serial if my_player.active and my_player.active[0] is not None else None
    
    current_pivots = sum(1 for p in my_all_poke if p.id in (140, 414, 400) for t in p.tools if t.id == 1157)
    
    has_fezandipiti = any(p.id == 140 for p in my_all_poke)
    has_ursulana = any(p.id == 44 for p in my_all_poke)
    
    current_my_top_serials = {p.serial: p.id for p in my_all_poke}
    current_my_all_serials = {}
    for p in my_all_poke:
        current_my_all_serials[p.serial] = p.id
        for pre in getattr(p, "preEvolution", []):
            current_my_all_serials[pre.serial] = pre.id
            
    current_opp_top_serials = {p.serial: p.id for p in opp_all_poke}
    current_opp_all_serials = {}
    for p in opp_all_poke:
        current_opp_all_serials[p.serial] = p.id
        for pre in getattr(p, "preEvolution", []):
            current_opp_all_serials[pre.serial] = pre.id
    
    # Calculate deltas if state is initialized
    if tracker["initialized"]:
        removed_my_serials = set()
        if "my_top_serials" in tracker:
            removed_my_serials = set(tracker["my_top_serials"].keys()) - set(current_my_all_serials.keys())

        prizes_taken = tracker.get("my_prizes", 6) - my_prizes
        prizes_lost = tracker.get("opp_prizes", 6) - opp_prizes
        
        # Detect Opponent's Pokemon KO'd (Prize Taken)
        if prizes_taken > 0:
            step_reward += _track_reward(tracker, "r_their_pkm_koed", REWARD_CONFIG.get("r_their_pkm_koed", 7.0) * prizes_taken, count=prizes_taken)
            
            if tracker.get("boss_active_for_ko"):
                bonus = REWARD_CONFIG.get("r_boss_ko_bonus", 0.5) * prizes_taken
                print(f"\n[REWARD] 🎯 BOSS'S ORDERS KO BONUS GRANTED! (+{bonus:.2f}) 🎯\n")
                step_reward += _track_reward(tracker, "r_boss_ko_bonus", bonus, count=prizes_taken)

        # Detect Agent's Pokemon KO'd (Prize Lost)
        if prizes_lost > 0:
            step_reward += _track_reward(tracker, "r_our_pkm_koed", REWARD_CONFIG.get("r_our_pkm_koed", -0.35) * prizes_lost, count=prizes_lost)
            
            if tracker.get("had_fezandipiti", False) and not has_fezandipiti:
                step_reward += _track_reward(tracker, "r_prize_lost_fezandipiti_penalty", REWARD_CONFIG.get("r_prize_lost_fezandipiti_penalty", -0.5) * prizes_lost, count=prizes_lost)
        if my_deck == 0 and tracker["my_deck"] > 0:
            step_reward += _track_reward(tracker, "r_deck_out", REWARD_CONFIG["r_deck_out"])
            
        if opp_deck == 0 and tracker.get("opp_deck", 1) > 0:
            penalty = REWARD_CONFIG.get("r_win_deck_out_penalty", -7.0)
            step_reward += _track_reward(tracker, "r_win_deck_out_penalty", penalty)
            print(f"\n[REWARD] 🛑 WON BY OPPONENT DECK OUT! PENALTY APPLIED ({penalty:.2f}) 🛑\n")
            
        # Card Play Reward
        if tracker.get("last_options") is not None:
            for act in tracker["last_actions"]:
                if act < len(tracker["last_options"]):
                    opt = tracker["last_options"][act]
                    
                    if opt.type != 14: # Not OptionType.END
                        tracker["actions_taken_this_turn"] = tracker.get("actions_taken_this_turn", 0) + 1
                        
                    if opt.type == 7: # OptionType.PLAY
                        step_reward += _track_reward(tracker, "r_play_trainer", REWARD_CONFIG.get("r_play_trainer", 0.05))
                        opt_idx = opt.kwargs.get("index") if hasattr(opt, "kwargs") else getattr(opt, "index", None)
                        if tracker.get("last_hand") is not None and opt_idx is not None:
                            try:
                                played_card = tracker["last_hand"][opt_idx]
                                if played_card.id in (1182, 1088, 1218):
                                    tracker["boss_active_for_ko"] = True
                                    if played_card.id == 1218:
                                        tracker["giovanni_played_turn"] = parsed_obs.current.turn
                                    
                                    boss_bonus = REWARD_CONFIG.get("r_play_boss", 0.5)
                                    step_reward += _track_reward(tracker, "r_play_boss", boss_bonus)
                                    print(f"\n[REWARD] 🎯 GUST CARD PLAYED (BOSS/GIO/CATCHER)! (+{boss_bonus:.2f}) 🎯\n")
                                    
                                    # Sniper Bonus Check
                                    active_pkmn = my_player.active[0] if my_player.active and my_player.active[0] is not None else None
                                    if active_pkmn:
                                        attack_dmg = ATTACK_DAMAGE_LOOKUP.get(active_pkmn.id, 100)
                                        valid_targets = [p for p in opp_player.bench if p is not None and p.hp <= attack_dmg]
                                        if valid_targets:
                                            snipe_bonus = REWARD_CONFIG.get("r_boss_snipe_bonus", 1.0)
                                            print(f"\n[REWARD] 🎯 SNIPE OPPORTUNITY IDENTIFIED! (+{snipe_bonus:.2f}) 🎯\n")
                                            step_reward += _track_reward(tracker, "r_boss_snipe_bonus", snipe_bonus)
                                elif played_card.id == 1251:
                                    if tracker.get("stadium_in_play") == 1251:
                                        wasted_stadium_penalty = REWARD_CONFIG.get("r_wasted_stadium_penalty", -0.5)
                                        step_reward += _track_reward(tracker, "r_wasted_stadium_penalty", wasted_stadium_penalty)
                                        print(f"\n[REWARD] ⚠️ WASTED STADIUM! LIVELY STADIUM ALREADY IN PLAY ({wasted_stadium_penalty:.2f}) ⚠️\n")
                                    else:
                                        step_reward += _track_reward(tracker, "r_play_stadium", REWARD_CONFIG.get("r_play_stadium", 0.05))
                                elif played_card.id == 1205: # Cyrano
                                    cyrano_bonus = REWARD_CONFIG.get("r_play_cyrano", 0.25)
                                    step_reward += _track_reward(tracker, "r_play_cyrano", cyrano_bonus)
                                    print(f"\n[REWARD] 🎭 CYRANO PLAYED (TR EX SEARCH)! (+{cyrano_bonus:.2f}) 🎭\n")

                                elif played_card.id in (1121, 1102, 1205, 1132, 1134):
                                    search_bonus = REWARD_CONFIG.get("r_play_search_card", 0.10)
                                    step_reward += _track_reward(tracker, "r_play_search_card", search_bonus)
                                    print(f"\n[REWARD] 🔍 SEARCH CARD PLAYED! (+{search_bonus:.2f}) 🔍\n")
                                elif played_card.id == 1120:
                                    valid_targets = [
                                        p for p in opp_all_poke
                                        if p is not None and len(p.energyCards) > 0 and (p.maxHp >= 200 or len(p.preEvolution) > 0)
                                    ]
                                    if valid_targets:
                                        hammer_bonus = REWARD_CONFIG.get("r_hammer_tempo_bonus", 0.20)
                                        step_reward += _track_reward(tracker, "r_hammer_tempo_bonus", hammer_bonus)
                                        print(f"\n[REWARD] 🔨 HAMMER DISRUPTION TARGET SPOTTED! (+{hammer_bonus:.2f}) 🔨\n")
                            except:
                                pass
                    elif opt.type == 14: # OptionType.END
                        if tracker.get("actions_taken_this_turn", 0) == 0:
                            pass_penalty = REWARD_CONFIG.get("r_pass_turn", -0.05)
                            step_reward += _track_reward(tracker, "r_pass_turn", pass_penalty)
                            print(f"\n[REWARD] 💤 TURN PASSED PREMATURELY! ({pass_penalty:.2f}) 💤\n")
                        
                        if any((getattr(o, "type", o.get("type") if isinstance(o, dict) else None) == 13) for o in tracker["last_options"]):
                            missed_attack_penalty = REWARD_CONFIG.get("r_missed_attack_penalty", -1.5)
                            step_reward += _track_reward(tracker, "r_missed_attack_penalty", missed_attack_penalty)
                            print(f"\n[REWARD] ❌ PASSED TURN WITH ATTACK AVAILABLE! ({missed_attack_penalty:.2f}) ❌\n")
                        
                        benched_pokemon = [p for p in my_player.bench if p is not None]
                        if len(benched_pokemon) == 0:
                            empty_bench_penalty = REWARD_CONFIG.get("r_empty_bench_penalty", -0.50)
                            step_reward += _track_reward(tracker, "r_empty_bench_penalty", empty_bench_penalty)
                            print(f"\n[REWARD] 🚨 TURN ENDED WITH EMPTY BENCH! ({empty_bench_penalty:.2f}) 🚨\n")
                            
                    elif opt.type == 13: # OptionType.ATTACK
                        attack_reward = REWARD_CONFIG.get("r_choose_attack", 0.25)
                        step_reward += _track_reward(tracker, "r_choose_attack", attack_reward)
                        print(f"\n[REWARD] ⚔️ ATTACK DECLARED! ({attack_reward:.2f}) ⚔️\n")
            
        # 2. Dense Setup Rewards
        energy_delta = current_energies - tracker["my_energies"]
        if energy_delta > 0:
            attached_to_active = False
            if tracker.get("last_options") is not None and tracker.get("last_actions") is not None:
                for act in tracker["last_actions"]:
                    if act < len(tracker["last_options"]):
                        opt = tracker["last_options"][act]
                        opt_type = getattr(opt, "type", opt.get("type") if isinstance(opt, dict) else None)
                        if opt_type == 8:
                            area = opt.get("inPlayArea", 4) if isinstance(opt, dict) else getattr(opt, "kwargs", {}).get("inPlayArea", getattr(opt, "inPlayArea", 4))
                            idx = opt.get("inPlayIndex", 0) if isinstance(opt, dict) else getattr(opt, "kwargs", {}).get("inPlayIndex", getattr(opt, "inPlayIndex", 0))
                            if area == 4 and idx == 0:
                                attached_to_active = True
                                break
            
            if attached_to_active:
                step_reward += _track_reward(tracker, "r_energy_attach", REWARD_CONFIG.get("r_energy_attach", 0.75) * energy_delta)
                
        damage_delta = current_opp_damage - tracker.get("opp_damage", 0)
        if damage_delta > 0:
            dmg_reward = REWARD_CONFIG.get("r_damage_dealt_per_10", 0.45) * (damage_delta / 10.0)
            step_reward += _track_reward(tracker, "r_damage_dealt_per_10", dmg_reward, count=int(damage_delta // 10))
            print(f"\n[REWARD] 💥 DEALT {damage_delta} DAMAGE! (+{dmg_reward:.2f}) 💥\n")
            
        mewtwo_energy_delta = current_mewtwo_energies - tracker.get("my_mewtwo_energies", 0)
        if mewtwo_energy_delta > 0:
            heavy_reward = REWARD_CONFIG.get("r_energy_attach_heavy", 0.10) * mewtwo_energy_delta
            step_reward += _track_reward(tracker, "r_energy_attach_heavy", heavy_reward)
            print(f"\n[REWARD] ⚡ MEWTWO EX CHARGED! (+{heavy_reward:.2f}) ⚡\n")
            
        ursaluna_energy_delta = current_ursaluna_energies - tracker.get("my_ursaluna_energies", 0)
        if ursaluna_energy_delta > 0:
            heavy_reward = REWARD_CONFIG.get("r_energy_attach_heavy", 0.10) * ursaluna_energy_delta
            step_reward += _track_reward(tracker, "r_energy_attach_heavy", heavy_reward)
            print(f"\n[REWARD] ⚡ URSALUNA BLOODMOON EX CHARGED! (+{heavy_reward:.2f}) ⚡\n")
            
        high_power_energy_delta = current_high_power_energies - tracker.get("my_high_power_energies", 0)
        if high_power_energy_delta > 0:
            hp_reward = REWARD_CONFIG.get("r_energy_attach_high_power", 0.10) * high_power_energy_delta
            step_reward += _track_reward(tracker, "r_energy_attach_high_power", hp_reward)
            print(f"\n[REWARD] 🔥 HIGH POWER ATTACKER CHARGED! (+{hp_reward:.2f}) 🔥\n")
            
        over_attach_delta = current_over_attached_energies - tracker.get("over_attached_energies", 0)
        if over_attach_delta > 0:
            penalty = REWARD_CONFIG.get("r_over_attach_penalty", -0.5) * over_attach_delta
            step_reward += _track_reward(tracker, "r_over_attach_penalty", penalty)
            print(f"\n[REWARD] ⚠️ OVER-ATTACHED ENERGY PENALTY! ({penalty:.2f}) ⚠️\n")
            
        fc_delta = current_fully_charged - tracker.get("my_fully_charged", 0)
        if fc_delta > 0:
            fc_reward = REWARD_CONFIG.get("r_fully_charged", 1.0) * fc_delta
            step_reward += _track_reward(tracker, "r_fully_charged", fc_reward)
            print(f"\n[REWARD] 🔋 POKEMON FULLY CHARGED! (+{fc_reward:.2f}) 🔋\n")
            
        articuno_energy_delta = current_articuno_energies - tracker.get("my_articuno_energies", 0)
        if articuno_energy_delta > 0:
            articuno_reward = REWARD_CONFIG.get("r_energy_attach_articuno", 0.10) * articuno_energy_delta
            step_reward += _track_reward(tracker, "r_energy_attach_articuno", articuno_reward)
            print(f"\n[REWARD] ❄️ TEAM ROCKET'S ARTICUNO CHARGED! (+{articuno_reward:.2f}) ❄️\n")
            
        mewtwo_tr_energy_delta = current_mewtwo_tr_energies - tracker.get("my_mewtwo_tr_energies", 0)
        if mewtwo_tr_energy_delta > 0:
            tr_bonus = REWARD_CONFIG.get("r_energy_attach_tr_specific", 0.15) * mewtwo_tr_energy_delta
            step_reward += _track_reward(tracker, "r_energy_attach_tr_specific", tr_bonus)
            print(f"\n[REWARD] 🚀 TEAM ROCKET'S ENERGY ON MEWTWO EX! (+{tr_bonus:.2f}) 🚀\n")

        articuno_tr_energy_delta = current_articuno_tr_energies - tracker.get("my_articuno_tr_energies", 0)
        if articuno_tr_energy_delta > 0:
            tr_bonus = REWARD_CONFIG.get("r_energy_attach_tr_specific", 0.15) * articuno_tr_energy_delta
            step_reward += _track_reward(tracker, "r_energy_attach_tr_specific", tr_bonus)
            print(f"\n[REWARD] 🚀 TEAM ROCKET'S ENERGY ON ARTICUNO! (+{tr_bonus:.2f}) 🚀\n")

        # Reward for moving fully charged undamaged pokemon to active
        if current_my_active_serial is not None and tracker.get("my_active_serial") is not None:
            if current_my_active_serial != tracker["my_active_serial"]:
                active_poke = my_player.active[0]
                if active_poke.hp == active_poke.maxHp:
                    val = sum(2 if getattr(e, 'id', None) == 15 else 1 for e in getattr(active_poke, 'energyCards', []))
                    charged = False
                    if active_poke.id == 400 and val >= 1: charged = True
                    elif active_poke.id in (401, 432, 434) and val >= 2: charged = True
                    elif active_poke.id in (140, 414, 431) and val >= 3: charged = True
                    elif active_poke.id == 44 and val >= max(0, opp_prizes - 1): charged = True
                    
                    if charged:
                        reward_amt = REWARD_CONFIG.get("r_active_fully_charged_undamaged", 2.0)
                        step_reward += _track_reward(tracker, "r_active_fully_charged_undamaged", reward_amt)
                        print(f"\n[REWARD] ⚡ FULLY CHARGED & UNDAMAGED POKEMON MOVED TO ACTIVE! (+{reward_amt:.2f}) ⚡\n")
                        
                        if tracker.get("giovanni_played_turn") == parsed_obs.current.turn:
                            jackpot = REWARD_CONFIG.get("r_giovanni_jackpot", 4.0)
                            step_reward += _track_reward(tracker, "r_giovanni_jackpot", jackpot)
                            print(f"\n[REWARD] 🎩 GIOVANNI'S FULLY CHARGED SWITCH JACKPOT! (+{jackpot:.2f}) 🎩\n")

        evo_delta = current_evolutions - tracker["my_evolutions"]
        if evo_delta > 0:
            step_reward += _track_reward(tracker, "r_evolution", REWARD_CONFIG["r_evolution"] * evo_delta)
            
        # Retreating Reward
        if current_my_active_serial is not None and tracker.get("my_active_serial") is not None:
            if current_my_active_serial != tracker["my_active_serial"] and len(removed_my_serials) == 0:
                current_turn = parsed_obs.current.turn
                if tracker.get("last_retreat_turn") == current_turn:
                    step_reward += _track_reward(tracker, "r_excessive_retreat_penalty", -0.5)
                else:
                    tracker["last_retreat_turn"] = current_turn
                    step_reward += _track_reward(tracker, "r_retreat", REWARD_CONFIG.get("r_retreat", 0.10))
                
        # Healing Reward
        my_damage_delta = current_my_damage - tracker.get("my_damage", 0)
        if my_damage_delta < 0 and len(removed_my_serials) == 0:
            step_reward += _track_reward(tracker, "r_healing_per_10", REWARD_CONFIG.get("r_healing_per_10", 0.02) * (-my_damage_delta / 10.0))
            
        hp_damage_delta = current_high_power_damage - tracker.get("my_high_power_damage", 0)
        if hp_damage_delta < 0 and len(removed_my_serials) == 0:
            hp_heal_reward = REWARD_CONFIG.get("r_healing_high_power_per_10", 0.10) * (-hp_damage_delta / 10.0)
            step_reward += _track_reward(tracker, "r_healing_high_power_per_10", hp_heal_reward)
            print(f"\n[REWARD] 💖 HIGH POWER POKEMON HEALED! (+{hp_heal_reward:.2f}) 💖\n")

        # TR Synergy & Mewtwo Jackpot
        tr_delta = current_tr_count - tracker.get("tr_pokemon_count", 0)
        if tr_delta > 0:
            tr_bonus = REWARD_CONFIG.get("r_team_rocket_board_presence", 0.20) * tr_delta
            print(f"\n[REWARD] 🚀 TEAM ROCKET SWARM INCREASED! (+{tr_bonus:.2f}) 🚀\n")
            step_reward += _track_reward(tracker, "r_team_rocket_board_presence", tr_bonus)
            
        if my_player.active and my_player.active[0] is not None and my_player.active[0].id == 431:
            if current_tr_count >= 4 and not tracker.get("mewtwo_jackpot_claimed", False):
                jackpot = REWARD_CONFIG.get("r_mewtwo_jackpot", 3.0)
                print(f"\n[REWARD] 🔮 MEWTWO EX POWER SAVER UNLOCKED! JACKPOT! (+{jackpot:.2f}) 🔮\n")
                step_reward += _track_reward(tracker, "r_mewtwo_jackpot", jackpot)
                tracker["mewtwo_jackpot_claimed"] = True
                
        if my_player.active and my_player.active[0] is not None and my_player.active[0].id == 44:
            if opp_prizes <= 3 and not tracker.get("ursaluna_active_reward_claimed", False):
                ursa_jackpot = REWARD_CONFIG.get("r_ursaluna_active_jackpot", 1.5)
                print(f"\n[REWARD] 🐻 URSALUNA BLOODMOON ACTIVATED IN ENDGAME! (+{ursa_jackpot:.2f}) 🐻\n")
                step_reward += _track_reward(tracker, "r_ursaluna_active_jackpot", ursa_jackpot)
                tracker["ursaluna_active_reward_claimed"] = True
                
        pivot_delta = current_pivots - tracker.get("rescue_board_pivots", 0)
        if pivot_delta > 0:
            pivot_bonus = REWARD_CONFIG.get("r_rescue_board_pivot", 0.15) * pivot_delta
            print(f"\n[REWARD] 🛹 FREE PIVOT CREATED! (+{pivot_bonus:.2f}) 🛹\n")
            step_reward += _track_reward(tracker, "r_rescue_board_pivot", pivot_bonus)
            
        # 3. Dense Attack Rewards
        damage_delta = current_opp_damage - tracker["opp_damage"]
        if damage_delta > 0:
            base_dmg_reward = _track_reward(tracker, "r_damage_dealt_per_10", REWARD_CONFIG["r_damage_dealt_per_10"] * (damage_delta / 10.0))
            
            if my_player.active and my_player.active[0] is not None and my_player.active[0].id in (431, 140, 414, 44):
                hp_atk_reward = REWARD_CONFIG.get("r_attack_high_power", 0.50)
                base_dmg_reward += _track_reward(tracker, "r_attack_high_power", hp_atk_reward)
                print(f"\n[REWARD] ⚔️ HIGH POWER ATTACK USED! (+{hp_atk_reward:.2f}) ⚔️\n")
                
            if has_ursulana and my_player.active and my_player.active[0] is not None and my_player.active[0].id == 44:
                bonus = REWARD_CONFIG.get("r_ursulana_attack_bonus_per_prize", 0.04) * (6 - opp_prizes)
                if bonus > 0:
                    print(f"\n[REWARD] 🐻 URSALUNA BLOODMOON ATTACK BONUS GRANTED! (+{bonus:.2f}) 🐻\n")
                    base_dmg_reward += _track_reward(tracker, "r_ursulana_attack_bonus_per_prize", bonus)
                
            if my_player.active and my_player.active[0] is not None and my_player.active[0].id == 431:
                kicker = REWARD_CONFIG.get("r_mewtwo_attack_kicker", 1.0)
                if kicker > 0:
                    print(f"\n[REWARD] 💥 MEWTWO EX BLAST KICKER! (+{kicker:.2f}) 💥\n")
                base_dmg_reward += _track_reward(tracker, "r_mewtwo_attack_kicker", kicker)
                
            step_reward += base_dmg_reward
            
    current_turn = parsed_obs.current.turn
    if tracker.get("current_turn") != current_turn:
        tracker["current_turn"] = current_turn
        tracker["actions_taken_this_turn"] = 0
        tracker["boss_active_for_ko"] = False
            
    tracker["my_prizes"] = my_prizes
    tracker["opp_prizes"] = opp_prizes
    tracker["my_deck"] = my_deck
    tracker["stadium_in_play"] = parsed_obs.current.stadium[0].id if getattr(parsed_obs.current, "stadium", None) else None
    tracker["my_energies"] = current_energies
    tracker["my_mewtwo_energies"] = current_mewtwo_energies
    tracker["my_ursaluna_energies"] = current_ursaluna_energies
    tracker["my_articuno_energies"] = current_articuno_energies
    tracker["my_mewtwo_tr_energies"] = current_mewtwo_tr_energies
    tracker["my_articuno_tr_energies"] = current_articuno_tr_energies
    tracker["my_high_power_energies"] = current_high_power_energies
    tracker["my_fully_charged"] = current_fully_charged
    tracker["my_evolutions"] = current_evolutions
    tracker["opp_damage"] = current_opp_damage
    tracker["my_damage"] = current_my_damage
    tracker["my_high_power_damage"] = current_high_power_damage
    tracker["opp_prizes"] = opp_prizes
    tracker["my_prizes"] = my_prizes
    tracker["my_deck"] = my_deck
    tracker["opp_deck"] = opp_deck
    tracker["my_active_serial"] = current_my_active_serial
    tracker["had_fezandipiti"] = has_fezandipiti
    tracker["my_top_serials"] = current_my_top_serials
    tracker["opp_top_serials"] = current_opp_top_serials
    tracker["tr_pokemon_count"] = current_tr_count
    tracker["rescue_board_pivots"] = current_pivots
    tracker["over_attached_energies"] = current_over_attached_energies
    # mewtwo_jackpot_claimed is updated in-place above
    tracker["initialized"] = True
    
    return step_reward
