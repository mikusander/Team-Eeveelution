import numpy as np
from typing import Dict, Any, List
from collections import Counter

# ... (TYPE_CHART e altre funzioni fino a summary_from_timeline rimangono invariate) ...
TYPE_CHART = {
    'normal': {
        'rock': 0.5, 
        'ghost': 0
    },
    'fire': {
        'fire': 0.5, 
        'water': 0.5, 
        'grass': 2, 
        'ice': 2, 
        'bug': 2, 
        'rock': 0.5, 
        'dragon': 0.5
    },
    'water': {
        'fire': 2, 
        'water': 0.5, 
        'grass': 0.5, 
        'ground': 2, 
        'rock': 2, 
        'dragon': 0.5
    },
    'grass': {
        'fire': 0.5, 
        'water': 2, 
        'grass': 0.5, 
        'poison': 0.5, 
        'ground': 2, 
        'flying': 0.5, 
        'bug': 0.5, 
        'rock': 2, 
        'dragon': 0.5
    },
    'electric': {
        'water': 2, 
        'grass': 0.5, 
        'electric': 0.5, 
        'ground': 0, 
        'flying': 2, 
        'dragon': 0.5
    },
    'ice': {
        'fire': 0.5,
        'water': 0.5,
        'grass': 2, 
        'ground': 2, 
        'flying': 2, 
        'dragon': 2
    },
    'fighting': {
        'normal': 2, 
        'ice': 2, 
        'poison': 0.5, 
        'flying': 0.5, 
        'psychic': 0.5, 
        'bug': 0.5, 
        'rock': 2, 
        'ghost': 0
    },
    'poison': {
        'grass': 2, 
        'poison': 0.5, 
        'ground': 0.5, 
        'bug': 2, # Particolarità della Gen 1
        'rock': 0.5, 
        'ghost': 0.5
    },
    'ground': {
        'fire': 2, 
        'grass': 0.5, 
        'electric': 2, 
        'poison': 2, 
        'flying': 0, 
        'bug': 0.5, 
        'rock': 2
    },
    'flying': {
        'grass': 2, 
        'electric': 0.5, 
        'fighting': 2, 
        'bug': 2, 
        'rock': 0.5
    },
    'psychic': {
        'fighting': 2, 
        'poison': 2, 
        'psychic': 0.5,
        'ghost': 0 # Famoso bug della Gen 1
    },
    'bug': {
        'fire': 0.5, 
        'grass': 2, 
        'fighting': 0.5, 
        'poison': 2, 
        'flying': 0.5, 
        'psychic': 2, # Particolarità della Gen 1
        'ghost': 0.5
    },
    'rock': {
        'fire': 2, 
        'ice': 2, 
        'fighting': 0.5, 
        'ground': 0.5, 
        'flying': 2, 
        'bug': 2
    },
    'ghost': {
        'normal': 0, 
        'psychic': 0, # Famoso bug della Gen 1
        'ghost': 2
    },
    'dragon': {
        'dragon': 2
    }
}

def get_effectiveness(attack_type: str, defense_types: List[str]) -> float:
    if not attack_type or not defense_types:
        return 1.0

    effectiveness = 1.0
    for def_type in defense_types:
        effectiveness *= TYPE_CHART.get(attack_type, {}).get(def_type, 1.0)
    return effectiveness

def calculate_type_advantage(team1: List[Dict], team2_lead: Dict) -> Dict[str, float]:
    out = {
        'p1_vs_lead_avg_effectiveness': 0.0,
        'p1_vs_lead_max_effectiveness': 0.0,
        'p1_super_effective_options': 0,
    }

    if not team1 or not team2_lead:
        return out

    lead_types = [t.lower() for t in team2_lead.get('types', [])]
    if not lead_types:
        return out

    team_effectiveness = []
    for pokemon in team1:
        pokemon_types = [t.lower() for t in pokemon.get('types', [])]
        max_eff = 0
        for p_type in pokemon_types:
            eff = get_effectiveness(p_type, lead_types)
            if eff > max_eff:
                max_eff = eff
        team_effectiveness.append(max_eff)

    if not team_effectiveness:
        return out

    out['p1_vs_lead_avg_effectiveness'] = np.mean(team_effectiveness)
    out['p1_vs_lead_max_effectiveness'] = np.max(team_effectiveness)
    out['p1_super_effective_options'] = sum(1 for eff in team_effectiveness if eff >= 2)

    return out

def team_aggregate_features(team: List[Dict[str, Any]], prefix: str = 'p1_') -> Dict[str, Any]:
    stats = ['base_hp', 'base_atk', 'base_def', 'base_spa', 'base_spd', 'base_spe']
    out = {}
    vals = {s: [] for s in stats}
    levels = []
    types_counter = Counter()
    names = []

    for p in team:
        name = p.get('name', '')
        names.append(name)

        for s in stats:
            vals[s].append(p.get(s, 0))

        levels.append(p.get('level', 0))

        for t in p.get('types', []):
            types_counter[t.lower()] += 1

    # Calcola somma, media, max, min, std per ogni statistica base
    for s in stats:
        arr = np.array(vals[s], dtype=float)
        out[f'{prefix}{s}_sum'] = arr.sum()
        out[f'{prefix}{s}_mean'] = arr.mean()
        out[f'{prefix}{s}_max'] = arr.max()
        out[f'{prefix}{s}_min'] = arr.min()
        out[f'{prefix}{s}_std'] = arr.std()

    # Statistiche sui livelli
    level_arr = np.array(levels, dtype=float)
    out[f'{prefix}level_mean'] = level_arr.mean()
    out[f'{prefix}level_sum'] = level_arr.sum()

    # Numero di tipi unici nel team
    out[f'{prefix}n_unique_types'] = len(types_counter)

    # Conta quanti Pokémon hanno ciascun tipo tra quelli più comuni
    common_types = ['normal','fire','water','electric','grass','psychic','ice','dragon','rock','ground','flying']
    for t in common_types:
        out[f'{prefix}type_{t}_count'] = types_counter.get(t, 0)

    # Nome del lead (primo Pokémon)
    out[f'{prefix}lead_name'] = names[0] if names else ''

    # Numero di nomi unici nel team
    out[f'{prefix}n_unique_names'] = len(set(names))

    return out

def lead_vs_lead_features(p1_lead: Dict[str, Any], p2_lead: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    stats = ['base_hp', 'base_atk', 'base_def', 'base_spa', 'base_spd', 'base_spe']

    for s in stats:
        p1_stat = p1_lead.get(s, 0)
        p2_stat = p2_lead.get(s, 0)
        out[f'lead_diff_{s}'] = p1_stat - p2_stat

    out['lead_speed_advantage'] = p1_lead.get('base_spe', 0) - p2_lead.get('base_spe', 0)

    p1_types = [t.lower() for t in p1_lead.get('types', [])]
    p2_types = [t.lower() for t in p2_lead.get('types', [])]
    
    max_effectiveness = 0.0
    if p1_types and p2_types:
        for p1_type in p1_types:
            eff = get_effectiveness(p1_type, p2_types)
            if eff > max_effectiveness:
                max_effectiveness = eff
                
    out['lead_p1_vs_p2_effectiveness'] = max_effectiveness

    return out

def lead_aggregate_features(pokemon: Dict[str, Any], prefix: str = 'p2_lead_') -> Dict[str, Any]:
    out = {}
    stats = ['base_hp', 'base_atk', 'base_def', 'base_spa', 'base_spd', 'base_spe']

    for s in stats:
        out[f'{prefix}{s}'] = pokemon.get(s, 0)
    out[f'{prefix}level'] = pokemon.get('level', 0)

    types = pokemon.get('types', [])
    common_types = ['normal','fire','water','electric','grass','psychic','ice','dragon','rock','ground','flying']
    for t in common_types:
        out[f'{prefix}type_{t}'] = int(t in [x.lower() for x in types])

    out[f'{prefix}name'] = pokemon.get('name', '')

    return out


def summary_from_timeline(timeline: List[Dict[str, Any]], p1_team: List[Dict[str, Any]]) -> Dict[str, Any]:
    out = {}
    
    if not timeline:
        # ... (default out invariato)
        return { 'tl_p1_moves': 0, 'tl_p2_moves': 0, 'tl_p1_est_damage': 0.0, 'tl_p2_est_damage': 0.0, 'damage_diff': 0.0 }
    
    p1_moves = p2_moves = 0
    p1_damage = p2_damage = 0.0
    p1_last_active = p2_last_active = ''
    p1_last_hp = p2_last_hp = np.nan
    p1_fainted = p2_fainted = 0
    p1_fainted_names, p2_fainted_names = set(), set()
    last_p1_hp, last_p2_hp = {}, {}

    # --- NUOVA FEATURE: MOMENTUM SHIFT ---
    p1_comeback_kos = 0
    p2_comeback_kos = 0

    # --- NUOVA FEATURE: CONTEGGIO STATUS ---
    p1_inflicted_statuses = Counter()
    p2_inflicted_statuses = Counter()
    p1_pokemon_statuses = {}
    p2_pokemon_statuses = {}
    
    for turn in timeline[:30]:
        prev_p1_fainted, prev_p2_fainted = p1_fainted, p2_fainted # Memorizza stato KO prima del turno

        p1_state = turn.get('p1_pokemon_state', {}) or {}
        p2_state = turn.get('p2_pokemon_state', {}) or {}

        # Calcolo KO
        if p1_state.get('fainted') and p1_state.get('name') not in p1_fainted_names:
            p1_fainted += 1
            p1_fainted_names.add(p1_state.get('name'))
        if p2_state.get('fainted') and p2_state.get('name') not in p2_fainted_names:
            p2_fainted += 1
            p2_fainted_names.add(p2_state.get('name'))

        # Calcolo Danno
        p2_name, p2_hp = p2_state.get('name'), p2_state.get('hp_pct')
        if p2_name and p2_hp is not None:
            prev_hp = last_p2_hp.get(p2_name)
            if prev_hp is not None:
                p1_damage += max(0.0, prev_hp - p2_hp)
            last_p2_hp[p2_name] = p2_hp

        p1_name, p1_hp = p1_state.get('name'), p1_state.get('hp_pct')
        if p1_name and p1_hp is not None:
            prev_hp = last_p1_hp.get(p1_name)
            if prev_hp is not None:
                p2_damage += max(0.0, prev_hp - p1_hp)
            last_p1_hp[p1_name] = p1_hp
        
        # --- LOGICA NUOVA FEATURE: MOMENTUM SHIFT ---
        damage_diff_so_far = p1_damage - p2_damage
        if p2_fainted > prev_p2_fainted and damage_diff_so_far < -1.0: # P1 fa un KO mentre è in svantaggio
            p1_comeback_kos += 1
        if p1_fainted > prev_p1_fainted and damage_diff_so_far > 1.0: # P2 fa un KO mentre è in svantaggio
            p2_comeback_kos += 1
            
        # --- LOGICA NUOVA FEATURE: CONTEGGIO STATUS ---
        p2_status = p2_state.get('status')
        if p2_name and p2_status and p2_pokemon_statuses.get(p2_name) != p2_status:
            p1_inflicted_statuses[p2_status] += 1
            p2_pokemon_statuses[p2_name] = p2_status

        p1_status = p1_state.get('status')
        if p1_name and p1_status and p1_pokemon_statuses.get(p1_name) != p1_status:
            p2_inflicted_statuses[p1_status] += 1
            p1_pokemon_statuses[p1_name] = p1_status

        # Altre metriche
        if turn.get('p1_move_details'):
            p1_moves += 1
        if turn.get('p2_move_details'):
            p2_moves += 1
            
        p1_last_hp = p1_state.get('hp_pct', np.nan)
        p2_last_hp = p2_state.get('hp_pct', np.nan)

    # Popola l'output con le feature esistenti...
    out['tl_p1_moves'] = p1_moves
    out['tl_p2_moves'] = p2_moves
    out['tl_p1_est_damage'] = float(p1_damage)
    out['tl_p2_est_damage'] = float(p2_damage)
    out['damage_diff'] = p1_damage - p2_damage
    out['fainted_diff'] = p1_fainted - p2_fainted
    out['tl_p1_last_hp'] = float(p1_last_hp) if p1_last_hp is not None else np.nan
    out['tl_p2_last_hp'] = float(p2_last_hp) if p2_last_hp is not None else np.nan

    # Aggiungi le feature di resilienza
    if p1_team:
        p1_total_hp_sum = sum(p.get('base_hp', 0) for p in p1_team)
        p1_avg_def = np.mean([p.get('base_def', 0) for p in p1_team if p.get('base_def') is not None] or [0])
        p1_avg_spd = np.mean([p.get('base_spd', 0) for p in p1_team if p.get('base_spd') is not None] or [0])
        out['tl_p2_damage_vs_p1_hp_pool'] = p2_damage / (p1_total_hp_sum + 1e-6)
        out['tl_p1_defensive_endurance'] = (p1_avg_def + p1_avg_spd) / (p2_damage + 1e-6)
    
    # --- OUTPUT NUOVE FEATURE: MOMENTUM SHIFT ---
    out['tl_p1_comeback_kos'] = p1_comeback_kos
    out['tl_p2_comeback_kos'] = p2_comeback_kos
    out['tl_comeback_kos_diff'] = p1_comeback_kos - p2_comeback_kos

    # --- OUTPUT NUOVE FEATURE: CONTEGGIO STATUS ---
    common_statuses = ['brn', 'par', 'slp', 'frz', 'psn', 'tox']
    for status in common_statuses:
        out[f'tl_p1_inflicted_{status}_count'] = p1_inflicted_statuses.get(status, 0)
        out[f'tl_p2_inflicted_{status}_count'] = p2_inflicted_statuses.get(status, 0)
        out[f'tl_inflicted_{status}_diff'] = p1_inflicted_statuses.get(status, 0) - p2_inflicted_statuses.get(status, 0)
        
    return out

def ability_features(team: List[Dict[str, Any]], prefix: str) -> Dict[str, Any]:
    immunity_abilities = {
        'levitate': 0,      
        'volt_absorb': 0,   
        'water_absorb': 0,  
        'flash_fire': 0,    
    }
    
    stat_drop_abilities = {
        'intimidate': 0,    
    }
    
    weather_abilities = {
        'drought': 0,       
        'drizzle': 0,       
        'sand_stream': 0,   
    }

    out = {}
    
    for pokemon in team:
        ability = pokemon.get('ability', '').lower().replace(' ', '_')
        if ability in immunity_abilities:
            immunity_abilities[ability] += 1
        if ability in stat_drop_abilities:
            stat_drop_abilities[ability] += 1
        if ability in weather_abilities:
            weather_abilities[ability] += 1
            
    for ability, count in immunity_abilities.items():
        out[f'{prefix}ability_{ability}_count'] = count
    for ability, count in stat_drop_abilities.items():
        out[f'{prefix}ability_{ability}_count'] = count
    for ability, count in weather_abilities.items():
        out[f'{prefix}ability_{weather_abilities}_count'] = count
        
    out[f'{prefix}total_immunity_abilities'] = sum(immunity_abilities.values())
    out[f'{prefix}total_stat_drop_abilities'] = sum(stat_drop_abilities.values())
    
    return out

def prepare_record_features(record: Dict[str, Any], max_turns: int = 30) -> Dict[str, Any]:
    out = {}

    out['battle_id'] = record.get('battle_id')
    if 'player_won' in record:
        out['player_won'] = int(bool(record.get('player_won')))

    p1_team = record.get('p1_team_details', [])
    team_feats = team_aggregate_features(p1_team, prefix='p1_')
    out.update(team_feats)

    p2_lead = record.get('p2_lead_details', {})
    lead_feats = lead_aggregate_features(p2_lead, prefix='p2_lead_')
    out.update(lead_feats)

    p1_abilities = ability_features(p1_team, prefix='p1_')
    out.update(p1_abilities)

    p1_lead = p1_team[0] if p1_team else {}
    lead_matchup_feats = lead_vs_lead_features(p1_lead, p2_lead)
    out.update(lead_matchup_feats)

    p2_abilities = ability_features([p2_lead], prefix='p2_lead_')
    out.update(p2_abilities)

    out['p1_intimidate_vs_lead'] = 1 if p1_abilities.get('p1_ability_intimidate_count', 0) > 0 else 0

    timeline = record.get('battle_timeline', [])
    tl_feats = summary_from_timeline(timeline[:max_turns], p1_team)
    out.update(tl_feats)

    out['team_hp_sum_minus_p2lead_hp'] = out.get('p1_base_hp_sum', 0) - out.get('p2_lead_base_hp', 0)
    out['team_spa_mean_minus_p2spa'] = out.get('p1_base_spa_mean', 0) - out.get('p2_lead_base_spa', 0)
    
    out['speed_advantage'] = out.get('p1_base_spe_sum', 0) - out.get('p2_lead_base_spe', 0)
    
    out['n_unique_types_diff'] = out.get('p1_n_unique_types', 0) - lead_feats.get('p2_lead_n_unique_types', 1)
    
    p1_moves = max(tl_feats.get('tl_p1_moves', 1), 1)
    p2_moves = max(tl_feats.get('tl_p2_moves', 1), 1)
    out['damage_per_turn_diff'] = (tl_feats.get('tl_p1_est_damage', 0.0)/p1_moves) - (tl_feats.get('tl_p2_est_damage', 0.0)/p2_moves)
    
    out['last_pair'] = f"{tl_feats.get('tl_p1_last_active','')}_VS_{tl_feats.get('tl_p2_last_active','')}"

    type_advantage_feats = calculate_type_advantage(p1_team, p2_lead)
    out.update(type_advantage_feats)
    
    p2_lead_bulk = out.get('p2_lead_base_def', 1) + out.get('p2_lead_base_spd', 1)
    out['p1_se_options_vs_lead_bulk'] = out.get('p1_super_effective_options', 0) / (p2_lead_bulk + 1e-6)

    # --- NUOVA FEATURE: AGGREGATE DEL TEAM AVVERSARIO ---
    p2_team = record.get('p2_team_details', [])
    if p2_team:
        p2_team_feats = team_aggregate_features(p2_team, prefix='p2_')
        out.update(p2_team_feats)
        # Differenze tra team
        out['team_hp_sum_diff'] = out.get('p1_base_hp_sum', 0) - out.get('p2_base_hp_sum', 0)
        out['team_spa_mean_diff'] = out.get('p1_base_spa_mean', 0) - out.get('p2_base_spa_mean', 0)
        out['team_spe_mean_diff'] = out.get('p1_base_spe_mean', 0) - out.get('p2_base_spe_mean', 0)
        out['n_unique_types_team_diff'] = out.get('p1_n_unique_types', 0) - out.get('p2_n_unique_types', 0)

    return out