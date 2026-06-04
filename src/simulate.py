from __future__ import annotations

import argparse
import ast
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
import xgboost as xgb

from elo import ELO_INIT, HOME_ADVANTAGE_VALUE, expected_score, goal_diff_multiplier, k_factor
from features import H2H_YEARS, ROLLING_WINDOW


DEFAULT_FEATURES = [
    "elo_diff",
    "home_avg_goals_scored",
    "home_avg_goals_conceded",
    "away_avg_goals_scored",
    "away_avg_goals_conceded",
    "h2h_matches",
    "h2h_home_avg_goals",
    "h2h_away_avg_goals",
    "h2h_home_win_rate",
    "home_wc_participations",
    "away_wc_participations",
    "is_home_home",
    "is_neutral",
    "competition_friendly",
    "competition_qualifier",
]

GROUP_LETTERS = list("ABCDEFGHIJKL")
MAX_GROUP_FIXTURES = 72
DEFAULT_RUNS_DIR = Path("data/processed/monte_carlo_runs")


@dataclass
class TournamentState:
    elo: dict[str, float]
    rolling: dict[str, deque[tuple[int, int]]]
    h2h: dict[tuple[str, str], list[dict]]
    wc_participations: dict[str, int]
    teams_seen_current_wc: set[str]
    fallback_avg_scored: float
    fallback_avg_conceded: float


def latest_model_path(pattern: str) -> Path:
    paths = sorted(Path("models").glob(pattern), key=lambda p: p.stat().st_mtime)
    if not paths:
        raise FileNotFoundError(f"Nenhum modelo encontrado com pattern: models/{pattern}")
    return paths[-1]


def load_xgb_model(path: Path) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor()
    model.load_model(path)
    return model


def load_feature_list() -> list[str]:
    candidates = sorted(
        Path("mlruns").glob("**/params/features"),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        return DEFAULT_FEATURES

    try:
        value = candidates[-1].read_text().strip()
        features = ast.literal_eval(value)
        if isinstance(features, list) and all(isinstance(c, str) for c in features):
            return features
    except (SyntaxError, ValueError):
        pass

    return DEFAULT_FEATURES


def as_naive_timestamp(value) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        return timestamp.tz_convert(None)
    return timestamp


def parse_bool(value, default: bool = False) -> bool:
    if pd.isna(value):
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def prepare_feature_frame(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas de features em falta: {missing}")

    X = df[feature_cols].copy()
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())
        X[col] = X[col].fillna(0)
    return X


def predict_lambdas(
    model_home: xgb.XGBRegressor,
    model_away: xgb.XGBRegressor,
    X: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    lambda_home = np.clip(model_home.predict(X), 0.1, None)
    lambda_away = np.clip(model_away.predict(X), 0.1, None)
    return lambda_home, lambda_away


def load_final_elo(path: str = "data/processed/final_elo.csv") -> dict[str, float]:
    elo_path = Path(path)
    if not elo_path.exists():
        return {}
    final_elo = pd.read_csv(elo_path)
    final_elo.columns = final_elo.columns.str.strip()
    object_cols = final_elo.select_dtypes(include="object").columns
    final_elo[object_cols] = final_elo[object_cols].apply(lambda s: s.str.strip())
    return dict(zip(final_elo["team"], final_elo["elo"]))


def build_initial_state(
    training: pd.DataFrame,
    final_elo: dict[str, float],
) -> TournamentState:
    history = training[training["home_score"].notna() & training["away_score"].notna()].copy()
    history["date"] = pd.to_datetime(history["date"])
    history = history.sort_values("date").reset_index(drop=True)

    elo = defaultdict(lambda: float(ELO_INIT), {team: float(value) for team, value in final_elo.items()})
    rolling: dict[str, deque[tuple[int, int]]] = defaultdict(lambda: deque(maxlen=ROLLING_WINDOW))
    h2h: dict[tuple[str, str], list[dict]] = defaultdict(list)
    wc_participations: dict[str, int] = defaultdict(int)

    for _, row in history.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        home_goals = int(row["home_score"])
        away_goals = int(row["away_score"])
        date = as_naive_timestamp(row["date"])

        rolling[home].append((home_goals, away_goals))
        rolling[away].append((away_goals, home_goals))

        h2h[(home, away)].append(
            {
                "date": date,
                "goals_for": home_goals,
                "goals_against": away_goals,
                "win": int(home_goals > away_goals),
            }
        )
        h2h[(away, home)].append(
            {
                "date": date,
                "goals_for": away_goals,
                "goals_against": home_goals,
                "win": int(away_goals > home_goals),
            }
        )

        for side, team in (("home", home), ("away", away)):
            value = row.get(f"{side}_wc_participations", np.nan)
            if pd.notna(value):
                wc_participations[team] = max(wc_participations[team], int(value))

    fallback_avg_scored = float(history[["home_score", "away_score"]].stack().mean())
    fallback_avg_conceded = fallback_avg_scored

    return TournamentState(
        elo=elo,
        rolling=rolling,
        h2h=h2h,
        wc_participations=wc_participations,
        teams_seen_current_wc=set(),
        fallback_avg_scored=fallback_avg_scored,
        fallback_avg_conceded=fallback_avg_conceded,
    )


def clone_state(state: TournamentState) -> TournamentState:
    return TournamentState(
        elo=defaultdict(lambda: float(ELO_INIT), dict(state.elo)),
        rolling=defaultdict(lambda: deque(maxlen=ROLLING_WINDOW), {
            team: deque(values, maxlen=ROLLING_WINDOW)
            for team, values in state.rolling.items()
        }),
        h2h=defaultdict(list, {
            pair: [dict(record) for record in records]
            for pair, records in state.h2h.items()
        }),
        wc_participations=defaultdict(int, dict(state.wc_participations)),
        teams_seen_current_wc=set(state.teams_seen_current_wc),
        fallback_avg_scored=state.fallback_avg_scored,
        fallback_avg_conceded=state.fallback_avg_conceded,
    )


def rolling_means(state: TournamentState, team: str) -> tuple[float, float]:
    records = list(state.rolling[team])
    if not records:
        return state.fallback_avg_scored, state.fallback_avg_conceded
    return (
        float(np.mean([goals_for for goals_for, _ in records])),
        float(np.mean([goals_against for _, goals_against in records])),
    )


def h2h_features(
    state: TournamentState,
    home_team: str,
    away_team: str,
    date: pd.Timestamp,
    home_avg_scored: float,
    away_avg_scored: float,
) -> tuple[int, float, float, float]:
    cutoff = date - pd.DateOffset(years=H2H_YEARS)
    records = [
        record
        for record in state.h2h[(home_team, away_team)]
        if cutoff <= record["date"] < date
    ]
    if not records:
        return 0, home_avg_scored, away_avg_scored, 0.33
    return (
        len(records),
        float(np.mean([record["goals_for"] for record in records])),
        float(np.mean([record["goals_against"] for record in records])),
        float(np.mean([record["win"] for record in records])),
    )


def make_match_features(
    home_team: str,
    away_team: str,
    state: TournamentState,
    feature_cols: list[str],
    date: pd.Timestamp,
    tournament: str = "FIFA World Cup",
    country: str = "",
    neutral: bool = True,
) -> pd.DataFrame:
    home_elo = state.elo[home_team]
    away_elo = state.elo[away_team]
    home_avg_scored, home_avg_conceded = rolling_means(state, home_team)
    away_avg_scored, away_avg_conceded = rolling_means(state, away_team)
    h2h_matches, h2h_home_avg_goals, h2h_away_avg_goals, h2h_home_win_rate = h2h_features(
        state,
        home_team,
        away_team,
        date,
        home_avg_scored,
        away_avg_scored,
    )
    is_world_cup = int(tournament == "FIFA World Cup")

    row = {
        "elo_diff": home_elo - away_elo,
        "home_avg_goals_scored": home_avg_scored,
        "home_avg_goals_conceded": home_avg_conceded,
        "away_avg_goals_scored": away_avg_scored,
        "away_avg_goals_conceded": away_avg_conceded,
        "h2h_matches": h2h_matches,
        "h2h_home_avg_goals": h2h_home_avg_goals,
        "h2h_away_avg_goals": h2h_away_avg_goals,
        "h2h_home_win_rate": h2h_home_win_rate,
        "home_wc_participations": state.wc_participations[home_team],
        "home_wc_has_experience": int(state.wc_participations[home_team] > 0),
        "away_wc_participations": state.wc_participations[away_team],
        "away_wc_has_experience": int(state.wc_participations[away_team] > 0),
        "is_home_home": int(home_team == country),
        "is_home_away": int(away_team == country),
        "is_neutral": int(neutral),
        "competition_friendly": 0,
        "competition_qualifier": 0,
        "competition_continental": 0,
        "competition_world_cup": is_world_cup,
    }
    return prepare_feature_frame(pd.DataFrame([row]), feature_cols)


def update_state_after_match(
    state: TournamentState,
    home_team: str,
    away_team: str,
    home_goals: int,
    away_goals: int,
    date: pd.Timestamp,
    tournament: str = "FIFA World Cup",
    neutral: bool = True,
    winner: str | None = None,
) -> None:
    home_elo = state.elo[home_team]
    away_elo = state.elo[away_team]
    home_elo_for_expected = home_elo if neutral else home_elo + HOME_ADVANTAGE_VALUE
    expected_home = expected_score(home_elo_for_expected, away_elo)
    expected_away = 1 - expected_home

    if winner == home_team or (winner is None and home_goals > away_goals):
        score_home, score_away = 1.0, 0.0
    elif winner == away_team or (winner is None and home_goals < away_goals):
        score_home, score_away = 0.0, 1.0
    else:
        score_home, score_away = 0.5, 0.5

    k_used = k_factor(tournament, use_competition_weights=True)
    k_used *= goal_diff_multiplier(abs(home_goals - away_goals))
    state.elo[home_team] = home_elo + k_used * (score_home - expected_home)
    state.elo[away_team] = away_elo + k_used * (score_away - expected_away)

    state.rolling[home_team].append((home_goals, away_goals))
    state.rolling[away_team].append((away_goals, home_goals))

    home_win = int(winner == home_team) if winner is not None else int(home_goals > away_goals)
    away_win = int(winner == away_team) if winner is not None else int(away_goals > home_goals)
    state.h2h[(home_team, away_team)].append(
        {"date": date, "goals_for": home_goals, "goals_against": away_goals, "win": home_win}
    )
    state.h2h[(away_team, home_team)].append(
        {"date": date, "goals_for": away_goals, "goals_against": home_goals, "win": away_win}
    )

    if tournament == "FIFA World Cup":
        for team in (home_team, away_team):
            if team not in state.teams_seen_current_wc:
                state.wc_participations[team] += 1
                state.teams_seen_current_wc.add(team)


def init_stats(teams: list[str]) -> dict[str, dict[str, int]]:
    return {
        team: {"points": 0, "gf": 0, "ga": 0, "gd": 0, "wins": 0, "draws": 0, "losses": 0}
        for team in teams
    }


def add_group_result(
    stats: dict[str, dict[str, int]],
    home: str,
    away: str,
    home_goals: int,
    away_goals: int,
) -> None:
    stats[home]["gf"] += home_goals
    stats[home]["ga"] += away_goals
    stats[away]["gf"] += away_goals
    stats[away]["ga"] += home_goals
    stats[home]["gd"] = stats[home]["gf"] - stats[home]["ga"]
    stats[away]["gd"] = stats[away]["gf"] - stats[away]["ga"]

    if home_goals > away_goals:
        stats[home]["points"] += 3
        stats[home]["wins"] += 1
        stats[away]["losses"] += 1
    elif home_goals < away_goals:
        stats[away]["points"] += 3
        stats[away]["wins"] += 1
        stats[home]["losses"] += 1
    else:
        stats[home]["points"] += 1
        stats[away]["points"] += 1
        stats[home]["draws"] += 1
        stats[away]["draws"] += 1


def mini_table_metrics(teams: list[str], matches: list[dict]) -> dict[str, tuple[int, int, int]]:
    table = {team: {"points": 0, "gf": 0, "ga": 0} for team in teams}
    team_set = set(teams)

    for match in matches:
        home, away = match["home_team"], match["away_team"]
        if home not in team_set or away not in team_set:
            continue
        hg, ag = match["home_goals"], match["away_goals"]
        table[home]["gf"] += hg
        table[home]["ga"] += ag
        table[away]["gf"] += ag
        table[away]["ga"] += hg
        if hg > ag:
            table[home]["points"] += 3
        elif hg < ag:
            table[away]["points"] += 3
        else:
            table[home]["points"] += 1
            table[away]["points"] += 1

    return {
        team: (values["points"], values["gf"] - values["ga"], values["gf"])
        for team, values in table.items()
    }


def rank_equal_teams(
    teams: list[str],
    stats: dict[str, dict[str, int]],
    matches: list[dict],
    rng: np.random.Generator,
) -> list[str]:
    h2h = mini_table_metrics(teams, matches)
    ordered = sorted(teams, key=lambda t: h2h[t], reverse=True)
    groups: list[list[str]] = []
    for team in ordered:
        if not groups or h2h[groups[-1][0]] != h2h[team]:
            groups.append([team])
        else:
            groups[-1].append(team)

    if len(groups) > 1:
        result: list[str] = []
        for group in groups:
            if len(group) == 1:
                result.extend(group)
            else:
                result.extend(rank_by_overall(group, stats, rng))
        return result

    return rank_by_overall(teams, stats, rng)


def rank_by_overall(
    teams: list[str],
    stats: dict[str, dict[str, int]],
    rng: np.random.Generator,
) -> list[str]:
    shuffled = list(teams)
    rng.shuffle(shuffled)
    return sorted(shuffled, key=lambda t: (stats[t]["gd"], stats[t]["gf"]), reverse=True)


def rank_group(
    teams: list[str],
    stats: dict[str, dict[str, int]],
    matches: list[dict],
    rng: np.random.Generator,
) -> list[str]:
    by_points: dict[int, list[str]] = defaultdict(list)
    for team in teams:
        by_points[stats[team]["points"]].append(team)

    ranking: list[str] = []
    for points in sorted(by_points.keys(), reverse=True):
        tied = by_points[points]
        if len(tied) == 1:
            ranking.extend(tied)
        else:
            ranking.extend(rank_equal_teams(tied, stats, matches, rng))
    return ranking


def rank_third_placed(
    third_teams: list[str],
    all_stats: dict[str, dict[str, int]],
    team_to_group: dict[str, str],
    rng: np.random.Generator,
) -> list[str]:
    shuffled = list(third_teams)
    rng.shuffle(shuffled)
    return sorted(
        shuffled,
        key=lambda t: (
            all_stats[t]["points"],
            all_stats[t]["gd"],
            all_stats[t]["gf"],
        ),
        reverse=True,
    )


def select_best_third_mapping(
    best_thirds: list[str],
    team_to_group: dict[str, str],
    combinations: pd.DataFrame,
) -> dict[str, str]:
    best_group_set = {team_to_group[team] for team in best_thirds}

    for _, row in combinations.iterrows():
        mapped_groups = {str(value).replace("3", "") for value in row.drop(labels=["Option"])}
        if mapped_groups == best_group_set:
            return {col: str(row[col]) for col in combinations.columns if col != "Option"}

    raise ValueError(f"Sem combinação para melhores terceiros dos grupos: {sorted(best_group_set)}")


def resolve_round32_best_thirds(
    slots: pd.DataFrame,
    combo_mapping: dict[str, str],
) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for _, row in slots[slots["round"].eq("Round of 32")].iterrows():
        slot_home = str(row["slot_home"])
        slot_away = str(row["slot_away"])
        both = [slot_home, slot_away]
        if not any(s.startswith("Best 3rd") for s in both):
            continue

        winner_slot = next((s for s in both if s.startswith("Winner Group ")), None)
        if winner_slot is None:
            raise ValueError(f"Best 3rd sem Winner Group no jogo {row['match_id']}")
        group = winner_slot.removeprefix("Winner Group ").strip()
        mapping[int(row["match_id"])] = combo_mapping[f"1{group}"]

    return mapping


def resolve_slot(
    slot: str,
    group_positions: dict[str, list[str]],
    third_by_group: dict[str, str],
    match_results: dict[int, dict[str, str]],
    round32_third_slots: dict[int, str],
    match_id: int,
) -> str:
    if slot.startswith("Winner Group "):
        group = slot.removeprefix("Winner Group ").strip()
        return group_positions[group][0]
    if slot.startswith("Runner-up Group "):
        group = slot.removeprefix("Runner-up Group ").strip()
        return group_positions[group][1]
    if slot.startswith("Best 3rd"):
        group_code = round32_third_slots[match_id]
        return third_by_group[group_code.replace("3", "")]

    match = re.search(r"(Winner|Loser) Match (\d+)", slot)
    if match:
        key = "winner" if match.group(1) == "Winner" else "loser"
        return match_results[int(match.group(2))][key]

    raise ValueError(f"Slot desconhecido: {slot}")


def simulate_knockout_match(
    home_team: str,
    away_team: str,
    model_home: xgb.XGBRegressor,
    model_away: xgb.XGBRegressor,
    state: TournamentState,
    feature_cols: list[str],
    rng: np.random.Generator,
    date: pd.Timestamp,
) -> tuple[int, int, str, str, bool, dict]:
    X = make_match_features(
        home_team=home_team,
        away_team=away_team,
        state=state,
        feature_cols=feature_cols,
        date=date,
        tournament="FIFA World Cup",
        neutral=True,
    )
    lambda_home, lambda_away = predict_lambdas(model_home, model_away, X)
    home_goals = int(rng.poisson(lambda_home[0]))
    away_goals = int(rng.poisson(lambda_away[0]))
    feature_row = {
        "date": date,
        "home_team": home_team,
        "away_team": away_team,
        "lambda_home": float(lambda_home[0]),
        "lambda_away": float(lambda_away[0]),
        "home_goals": home_goals,
        "away_goals": away_goals,
        **X.iloc[0].to_dict(),
    }

    went_to_penalties = home_goals == away_goals
    if home_goals > away_goals:
        winner, loser = home_team, away_team
        update_state_after_match(state, home_team, away_team, home_goals, away_goals, date, winner=winner)
        return home_goals, away_goals, winner, loser, went_to_penalties, feature_row
    if away_goals > home_goals:
        winner, loser = away_team, home_team
        update_state_after_match(state, home_team, away_team, home_goals, away_goals, date, winner=winner)
        return home_goals, away_goals, winner, loser, went_to_penalties, feature_row

    p_home = 1.0 / (1.0 + 10.0 ** (-(state.elo[home_team] - state.elo[away_team]) / 400.0))
    p_home = float(np.clip(0.40 + (p_home - 0.50) * 0.50, 0.35, 0.65))
    if rng.random() < p_home:
        winner, loser = home_team, away_team
    else:
        winner, loser = away_team, home_team
    update_state_after_match(state, home_team, away_team, home_goals, away_goals, date, winner=winner)
    return home_goals, away_goals, winner, loser, went_to_penalties, feature_row


def simulate_once(
    group_fixtures: pd.DataFrame,
    groups: pd.DataFrame,
    slots: pd.DataFrame,
    combinations: pd.DataFrame,
    model_home: xgb.XGBRegressor,
    model_away: xgb.XGBRegressor,
    initial_state: TournamentState,
    feature_cols: list[str],
    rng: np.random.Generator,
    collect_match_features: bool = False,
) -> tuple[dict[str, str], list[dict], list[dict]]:
    state = clone_state(initial_state)
    team_to_group = dict(zip(groups["team"], groups["group"]))
    group_to_teams = groups.groupby("group")["team"].apply(list).to_dict()
    group_stats = {group: init_stats(teams) for group, teams in group_to_teams.items()}
    group_matches: dict[str, list[dict]] = defaultdict(list)
    feature_log: list[dict] = []

    for _, row in group_fixtures.iterrows():
        group = team_to_group[row["home_team"]]
        date = as_naive_timestamp(row["date"])
        X = make_match_features(
            home_team=row["home_team"],
            away_team=row["away_team"],
            state=state,
            feature_cols=feature_cols,
            date=date,
            tournament=row.get("tournament", "FIFA World Cup"),
            country=row.get("country", ""),
            neutral=parse_bool(row.get("neutral", True), default=True),
        )
        lambda_home, lambda_away = predict_lambdas(model_home, model_away, X)
        home_goals = int(rng.poisson(lambda_home[0]))
        away_goals = int(rng.poisson(lambda_away[0]))
        if collect_match_features:
            feature_log.append(
                {
                    "phase": "Group stage",
                    "date": date,
                    "home_team": row["home_team"],
                    "away_team": row["away_team"],
                    "lambda_home": float(lambda_home[0]),
                    "lambda_away": float(lambda_away[0]),
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    **X.iloc[0].to_dict(),
                }
            )
        add_group_result(group_stats[group], row["home_team"], row["away_team"], home_goals, away_goals)
        group_matches[group].append(
            {
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "home_goals": home_goals,
                "away_goals": away_goals,
            }
        )
        update_state_after_match(
            state=state,
            home_team=row["home_team"],
            away_team=row["away_team"],
            home_goals=home_goals,
            away_goals=away_goals,
            date=date,
            tournament=row.get("tournament", "FIFA World Cup"),
            neutral=parse_bool(row.get("neutral", True), default=True),
        )

    group_positions = {}
    all_stats = {}
    for group in GROUP_LETTERS:
        ranking = rank_group(group_to_teams[group], group_stats[group], group_matches[group], rng)
        group_positions[group] = ranking
        all_stats.update(group_stats[group])

    third_teams = [group_positions[group][2] for group in GROUP_LETTERS]
    best_thirds = rank_third_placed(third_teams, all_stats, team_to_group, rng)[:8]
    combo_mapping = select_best_third_mapping(best_thirds, team_to_group, combinations)
    third_by_group = {team_to_group[team]: team for team in best_thirds}
    round32_third_slots = resolve_round32_best_thirds(slots, combo_mapping)

    match_results: dict[int, dict[str, str]] = {}
    knockout_log: list[dict] = []
    for _, row in slots.sort_values("match_id").iterrows():
        match_id = int(row["match_id"])
        home_team = resolve_slot(
            str(row["slot_home"]),
            group_positions,
            third_by_group,
            match_results,
            round32_third_slots,
            match_id,
        )
        away_team = resolve_slot(
            str(row["slot_away"]),
            group_positions,
            third_by_group,
            match_results,
            round32_third_slots,
            match_id,
        )
        match_date = as_naive_timestamp(row["date_utc"])
        hg, ag, winner, loser, pens, knockout_features = simulate_knockout_match(
            home_team,
            away_team,
            model_home,
            model_away,
            state,
            feature_cols,
            rng,
            match_date,
        )
        if collect_match_features:
            feature_log.append(
                {
                    "phase": row["round"],
                    "match_id": match_id,
                    "winner": winner,
                    "loser": loser,
                    "penalties": pens,
                    **knockout_features,
                }
            )
        match_results[match_id] = {"winner": winner, "loser": loser}
        knockout_log.append(
            {
                "match_id": match_id,
                "round": row["round"],
                "home_team": home_team,
                "away_team": away_team,
                "home_goals": hg,
                "away_goals": ag,
                "winner": winner,
                "loser": loser,
                "penalties": pens,
            }
        )

    champion = match_results[104]["winner"]
    return {
        "champion": champion,
        "runner_up": match_results[104]["loser"],
        "third_place": match_results[103]["winner"],
        "fourth_place": match_results[103]["loser"],
    }, knockout_log, feature_log


def stage_reached(round_name: str) -> str:
    mapping = {
        "Round of 32": "round_of_32",
        "Round of 16": "round_of_16",
        "Quarter-final": "quarter_final",
        "Semi-final": "semi_final",
        "Third-place playoff": "third_place_playoff",
        "Final": "final",
    }
    return mapping[round_name]


def clean_csv_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.columns = result.columns.str.strip()
    object_cols = result.select_dtypes(include="object").columns
    result[object_cols] = result[object_cols].apply(lambda s: s.str.strip())
    return result


def validate_groups(groups: pd.DataFrame, group_fixtures: pd.DataFrame) -> None:
    required_cols = {"team", "group"}
    missing_cols = required_cols - set(groups.columns)
    if missing_cols:
        raise ValueError(f"groups.csv precisa das colunas {sorted(required_cols)}. Em falta: {sorted(missing_cols)}")

    expected_groups = set(GROUP_LETTERS)
    found_groups = set(groups["group"])
    missing_groups = expected_groups - found_groups
    extra_groups = found_groups - expected_groups
    if missing_groups or extra_groups:
        raise ValueError(
            "groups.csv tem grupos invalidos. "
            f"Em falta: {sorted(missing_groups)} | Extra: {sorted(extra_groups)}"
        )

    duplicated_teams = groups.loc[groups["team"].duplicated(), "team"].tolist()
    if duplicated_teams:
        raise ValueError(f"Equipas duplicadas em groups.csv: {duplicated_teams}")

    group_sizes = groups.groupby("group")["team"].size()
    bad_sizes = group_sizes[group_sizes.ne(4)]
    if not bad_sizes.empty:
        raise ValueError(f"Cada grupo deve ter 4 equipas. Tamanhos invalidos: {bad_sizes.to_dict()}")

    fixture_teams = set(group_fixtures["home_team"]).union(group_fixtures["away_team"])
    group_teams = set(groups["team"])
    missing_teams = fixture_teams - group_teams
    extra_teams = group_teams - fixture_teams
    if missing_teams or extra_teams:
        raise ValueError(
            "As equipas dos 72 jogos e de groups.csv nao coincidem. "
            f"Nos jogos mas fora de groups.csv: {sorted(missing_teams)} | "
            f"Em groups.csv mas sem jogos: {sorted(extra_teams)}"
        )


def run_simulations(args: argparse.Namespace) -> None:
    started_at = time.time()
    rng = np.random.default_rng(args.seed)
    training = clean_csv_frame(pd.read_csv(args.training_df))
    groups = clean_csv_frame(pd.read_csv(args.groups))
    slots = clean_csv_frame(pd.read_csv(args.knockout_slots))
    combinations = clean_csv_frame(pd.read_csv(args.best_third_combinations))

    feature_cols = load_feature_list()
    final_elo = load_final_elo()
    model_home_path = Path(args.model_home) if args.model_home else latest_model_path("model_home_*.json")
    model_away_path = Path(args.model_away) if args.model_away else latest_model_path("model_away_*.json")
    model_home = load_xgb_model(model_home_path)
    model_away = load_xgb_model(model_away_path)

    group_fixtures = training.tail(MAX_GROUP_FIXTURES).copy().reset_index(drop=True)
    validate_groups(groups, group_fixtures)
    group_fixtures["date"] = pd.to_datetime(group_fixtures["date"])
    group_fixtures = group_fixtures.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)

    initial_state = build_initial_state(training, final_elo)
    teams = sorted(groups["team"].unique())
    counts = {
        team: defaultdict(int)
        for team in teams
    }
    sample_rows: list[dict] = []
    feature_rows: list[dict] = []

    save_all = args.save_sample_sims < 0
    for sim_id in range(1, args.n_sims + 1):
        collect_features = save_all or sim_id <= args.save_sample_sims
        placements, knockout_log, feature_log = simulate_once(
            group_fixtures=group_fixtures,
            groups=groups,
            slots=slots,
            combinations=combinations,
            model_home=model_home,
            model_away=model_away,
            initial_state=initial_state,
            feature_cols=feature_cols,
            rng=rng,
            collect_match_features=collect_features,
        )

        counts[placements["champion"]]["champion"] += 1
        counts[placements["runner_up"]]["runner_up"] += 1
        counts[placements["third_place"]]["third_place"] += 1
        counts[placements["fourth_place"]]["fourth_place"] += 1

        for match in knockout_log:
            stage = stage_reached(match["round"])
            counts[match["home_team"]][stage] += 1
            counts[match["away_team"]][stage] += 1

        if collect_features:
            for feature_row in feature_log:
                feature_rows.append({"simulation": sim_id, **feature_row})
            for match in knockout_log:
                sample_rows.append({"simulation": sim_id, **match})

        if args.progress_every > 0 and (sim_id % args.progress_every == 0 or sim_id == args.n_sims):
            elapsed = time.time() - started_at
            sims_per_second = sim_id / elapsed if elapsed > 0 else 0
            remaining = args.n_sims - sim_id
            eta_seconds = remaining / sims_per_second if sims_per_second > 0 else 0
            print(
                f"Progresso: {sim_id}/{args.n_sims} simulacoes "
                f"({sim_id / args.n_sims:.1%}) | "
                f"{sims_per_second:.2f} sim/s | "
                f"ETA {eta_seconds / 60:.1f} min"
            )

    summary = pd.DataFrame(
        [
            {
                "team": team,
                "p_round_of_32": counts[team]["round_of_32"] / args.n_sims,
                "p_round_of_16": counts[team]["round_of_16"] / args.n_sims,
                "p_quarter_final": counts[team]["quarter_final"] / args.n_sims,
                "p_semi_final": counts[team]["semi_final"] / args.n_sims,
                "p_final": counts[team]["final"] / args.n_sims,
                "p_champion": counts[team]["champion"] / args.n_sims,
                "p_runner_up": counts[team]["runner_up"] / args.n_sims,
                "p_third_place": counts[team]["third_place"] / args.n_sims,
            }
            for team in teams
        ]
    ).sort_values("p_champion", ascending=False)

    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_RUNS_DIR / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"wc2026_monte_carlo_summary_{timestamp}.csv"
    summary.to_csv(summary_path, index=False)

    sample_path = None
    if sample_rows:
        sample_path = output_dir / f"wc2026_knockout_sample_{timestamp}.csv"
        pd.DataFrame(sample_rows).to_csv(sample_path, index=False)

    group_sample_path = None
    if feature_rows:
        group_rows = [row for row in feature_rows if row.get("phase") == "Group stage"]
        if group_rows:
            group_sample_path = output_dir / f"wc2026_group_sample_{timestamp}.csv"
            pd.DataFrame(group_rows).to_csv(group_sample_path, index=False)

    feature_path = None
    if feature_rows:
        feature_path = output_dir / f"wc2026_dynamic_feature_sample_{timestamp}.csv"
        pd.DataFrame(feature_rows).to_csv(feature_path, index=False)

    print(f"Modelo home: {model_home_path}")
    print(f"Modelo away: {model_away_path}")
    print(f"Simulações: {args.n_sims}")
    print(f"Resumo: {summary_path}")
    if sample_path:
        print(f"Amostra de knockouts: {sample_path}")
    if group_sample_path:
        print(f"Amostra da fase de grupos: {group_sample_path}")
    if feature_path:
        print(f"Amostra de features dinamicas: {feature_path}")
    print("\nTop 10 campeao:")
    print(summary[["team", "p_champion", "p_final", "p_semi_final"]].head(10).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulador Monte Carlo do Mundial 2026.")
    parser.add_argument("--n-sims", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timestamp", default=None, help="Timestamp a usar nos ficheiros de output.")
    parser.add_argument("--training-df", default="data/processed/training_df.csv")
    parser.add_argument("--groups", default="data/raw/groups.csv")
    parser.add_argument("--knockout-slots", default="data/raw/knockout_slots.csv")
    parser.add_argument(
        "--best-third-combinations",
        default="data/raw/fw26_best_third_placed_combinations.csv",
    )
    parser.add_argument("--model-home", default=None)
    parser.add_argument("--model-away", default=None)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Pasta de output. Por defeito cria data/processed/monte_carlo_runs/<timestamp>.",
    )
    parser.add_argument("--save-sample-sims", type=int, default=-1,
        help="Numero de simulacoes a guardar para detalhes de grupo/jogo. -1 = todas. Default: -1.")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Mostra progresso a cada N simulacoes. Usa 0 para desligar.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_simulations(parse_args())
