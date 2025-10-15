import numpy as np
from typing import Dict, Any, List
from collections import Counter

def team_aggregate_features(team: List[Dict[str, Any]], prefix: str = 'p1_') -> Dict[str, Any]:
    # team: list of 6 pokemon dicts
    stats = ['base_hp', 'base_atk', 'base_def', 'base_spa', 'base_spd', 'base_spe']
    out = {}
    vals = {s: [] for s in stats}
    types_counter = Counter()
    names = []
    for p in team:
        names.append(p.get('name', ''))
        for s in stats:
            v = p.get(s, 0)
            vals[s].append(v)
        for t in p.get('types', []):
            types_counter[t.lower()] += 1
    # basic aggregates
    for s in stats:
        arr = np.array(vals[s], dtype=float)
        out[f'{prefix}{s}_sum'] = float(arr.sum())
        out[f'{prefix}{s}_mean'] = float(arr.mean())
        out[f'{prefix}{s}_max'] = float(arr.max())
        out[f'{prefix}{s}_min'] = float(arr.min())
        out[f'{prefix}{s}_std'] = float(arr.std())
    # types: top 6 most common types as features
    for t, cnt in types_counter.items():
        out[f'{prefix}type_{t}_count'] = int(cnt)
    # fallback for common types
    common_types = ['normal','fire','water','electric','grass','psychic','ice','dragon','rock','ground','flying']
    for t in common_types:
        out.setdefault(f'{prefix}type_{t}_count', 0)
    # names: we keep the lead name as a categorical
    out[f'{prefix}lead_name'] = names[0] if len(names) > 0 else ''
    out[f'{prefix}n_unique_names'] = len(set(names))
    return out


def summary_from_timeline(timeline: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Create timeline-derived features limited to first 30 turns (timeline already has up to 30)
    out = {}
    if not timeline:
        return out
    # counts
    p1_moves = 0
    p2_moves = 0
    p1_damage = 0.0  # estimated by hp_pct differences of opponent's active mon between successive turns when same name
    p2_damage = 0.0
    p1_status_inflicted = 0
    p2_status_inflicted = 0
    p1_high_power_moves = 0
    p2_high_power_moves = 0
    p1_last_active = None
    p2_last_active = None
    p1_last_hp = None
    p2_last_hp = None

    # We'll track last seen hp_pct by active mon name to estimate damage done per side
    last_p2_hp_by_name = {}
    last_p1_hp_by_name = {}

    for t in timeline:
        p1_state = t.get('p1_pokemon_state', {}) or {}
        p2_state = t.get('p2_pokemon_state', {}) or {}
        # track last active
        p1_last_active = p1_state.get('name')
        p2_last_active = p2_state.get('name')
        p1_last_hp = p1_state.get('hp_pct')
        p2_last_hp = p2_state.get('hp_pct')

        # moves
        p1_move = t.get('p1_move_details')
        p2_move = t.get('p2_move_details')
        if p1_move:
            p1_moves += 1
            bp = p1_move.get('base_power', 0) or 0
            if bp >= 80:
                p1_high_power_moves += 1
            # status inducing move detection via category == STATUS and maybe move name
            if p1_move.get('category') == 'STATUS':
                p1_status_inflicted += 1
        if p2_move:
            p2_moves += 1
            bp = p2_move.get('base_power', 0) or 0
            if bp >= 80:
                p2_high_power_moves += 1
            if p2_move.get('category') == 'STATUS':
                p2_status_inflicted += 1

        # estimate damage by comparing hp_pct for same-name pokemon across turns
        # p1 damage to p2: if same p2 mon name seen previously, delta of last hp - current hp (if positive)
        name = p2_state.get('name')
        hp = p2_state.get('hp_pct')
        if name is not None and hp is not None:
            prev = last_p2_hp_by_name.get(name)
            if prev is not None:
                delta = max(0.0, prev - hp)
                p1_damage += delta
            last_p2_hp_by_name[name] = hp

        name1 = p1_state.get('name')
        hp1 = p1_state.get('hp_pct')
        if name1 is not None and hp1 is not None:
            prev1 = last_p1_hp_by_name.get(name1)
            if prev1 is not None:
                delta1 = max(0.0, prev1 - hp1)
                p2_damage += delta1
            last_p1_hp_by_name[name1] = hp1

    # populate
    out['tl_p1_moves'] = p1_moves
    out['tl_p2_moves'] = p2_moves
    out['tl_p1_high_power_moves'] = p1_high_power_moves
    out['tl_p2_high_power_moves'] = p2_high_power_moves
    out['tl_p1_status_moves'] = p1_status_inflicted
    out['tl_p2_status_moves'] = p2_status_inflicted
    out['tl_p1_est_damage'] = float(p1_damage)
    out['tl_p2_est_damage'] = float(p2_damage)
    out['tl_p1_last_active'] = p1_last_active or ''
    out['tl_p2_last_active'] = p2_last_active or ''
    out['tl_p1_last_hp'] = float(p1_last_hp) if p1_last_hp is not None else np.nan
    out['tl_p2_last_hp'] = float(p2_last_hp) if p2_last_hp is not None else np.nan
    # simple ratios
    out['tl_damage_ratio'] = float((p1_damage + 1e-6) / (p2_damage + 1e-6))
    out['tl_moves_diff'] = p1_moves - p2_moves

        # --- Nuove feature ingegnerizzate ---
    # Differenza netta di danno
    out['damage_diff'] = out['tl_p1_est_damage'] - out['tl_p2_est_damage']

    # Danno medio per mossa (robusto al numero di turni)
    out['damage_per_move_diff'] = (
        (out['tl_p1_est_damage'] / (out['tl_p1_moves'] + 1e-6))
        - (out['tl_p2_est_damage'] / (out['tl_p2_moves'] + 1e-6))
    )

    # Rapporto di HP residui (normalizzato, +1 per stabilità)
    out['hp_diff_ratio'] = (
        (out['tl_p1_last_hp'] + 1e-6) / (out['tl_p2_last_hp'] + 1e-6)
    )

    return out

def prepare_record_features(record: Dict[str, Any], max_turns: int = 30) -> Dict[str, Any]:
    """
    Estrae tutte le feature tabellari da un record singolo
    """
    out = {}

    # ID e target
    out['battle_id'] = record.get('battle_id')
    if 'player_won' in record:
        out['player_won'] = int(bool(record.get('player_won')))

    # Team features
    p1_team = record.get('p1_team_details', [])
    p2_lead = record.get('p2_lead_details', {})
    out.update(team_aggregate_features(p1_team, prefix='p1_'))

    # P2 lead aggregate
    stats = ['base_hp','base_atk','base_def','base_spa','base_spd','base_spe']
    for s in stats:
        out[f'p2_lead_{s}'] = p2_lead.get(s, 0)
    out['p2_lead_name'] = p2_lead.get('name', '')
    types = p2_lead.get('types', [])
    for t in types:
        out[f'p2_lead_type_{t.lower()}'] = 1
    # fallback zero per tipi comuni
    common_types = ['normal','fire','water','electric','grass','psychic','ice','dragon','rock','ground','flying']
    for t in common_types:
        out.setdefault(f'p2_lead_type_{t}', 0)

    # Timeline summary per ML
    timeline = record.get('battle_timeline', [])
    out.update(summary_from_timeline(timeline))

    # Feature ingegnerizzate
    out['team_hp_sum_minus_p2lead_hp'] = out.get('p1_base_hp_sum', 0) - out.get('p2_lead_base_hp', 0)
    out['team_spa_mean_minus_p2spa'] = out.get('p1_base_spa_mean', 0) - out.get('p2_lead_base_spa', 0)
    out['last_pair'] = f"{out.get('tl_p1_last_active','')}_VS_{out.get('tl_p2_last_active','')}"

    return out

def make_features(record: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    # ids and target
    out['battle_id'] = record.get('battle_id')
    if 'player_won' in record:
        out['player_won'] = int(bool(record.get('player_won')))

    # team features
    p1_team = record.get('p1_team_details', [])
    p2_lead = record.get('p2_lead_details', {})
    out.update(team_aggregate_features(p1_team, prefix='p1_'))

    # p2 lead aggregate (single mon)
    for s in ['base_hp', 'base_atk', 'base_def', 'base_spa', 'base_spd', 'base_spe']:
        out[f'p2_lead_{s}'] = p2_lead.get(s, 0)
    out['p2_lead_name'] = p2_lead.get('name', '')
    types = p2_lead.get('types', [])
    for t in types:
        out[f'p2_lead_type_{t.lower()}'] = 1
    # ensure zero for common types
    common_types = ['normal','fire','water','electric','grass','psychic','ice','dragon','rock','ground','flying']
    for t in common_types:
        out.setdefault(f'p2_lead_type_{t}', 0)

    # timeline summaries
    timeline = record.get('battle_timeline', [])
    out.update(summary_from_timeline(timeline))

    # engineered interactions
    out['team_hp_sum_minus_p2lead_hp'] = out.get('p1_base_hp_sum', 0) - out.get('p2_lead_base_hp', 0)
    out['team_spa_mean_minus_p2spa'] = out.get('p1_base_spa_mean', 0) - out.get('p2_lead_base_spa', 0)
    # last active name pair hash (simple)
    out['last_pair'] = f"{out.get('tl_p1_last_active','')}_VS_{out.get('tl_p2_last_active','')}"

    return out