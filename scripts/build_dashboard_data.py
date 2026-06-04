from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import re

import pandas as pd


ROUND_LABELS = {
    "p_round_of_32": "Reach Round of 32",
    "p_round_of_16": "Reach Round of 16",
    "p_quarter_final": "Reach quarter-finals",
    "p_semi_final": "Reach semi-finals",
    "p_final": "Reach final",
    "p_champion": "Win tournament",
    "p_runner_up": "Finish runner-up",
    "p_third_place": "Finish third",
}

PROCESSED_DIR = Path("data/processed")
RUNS_DIR = PROCESSED_DIR / "monte_carlo_runs"
SUMMARY_RE = re.compile(r"wc2026_monte_carlo_summary_(\d{8}_\d{6})\.csv$")
GROUP_RE = re.compile(r"wc2026_group_sample_(\d{8}_\d{6})\.csv$")
FEATURES_RE = re.compile(r"wc2026_dynamic_feature_sample_(\d{8}_\d{6})\.csv$")
KNOCKOUT_RE = re.compile(r"wc2026_knockout_sample_(\d{8}_\d{6})\.csv$")
RUN_DIR_RE = re.compile(r"^\d{8}_\d{6}$")


def latest_run_dir() -> Path:
    if not RUNS_DIR.exists():
        raise FileNotFoundError(f"No Monte Carlo runs found in {RUNS_DIR}")

    candidates = [path for path in RUNS_DIR.iterdir() if path.is_dir() and RUN_DIR_RE.match(path.name)]
    if not candidates:
        raise FileNotFoundError(f"No timestamped Monte Carlo runs found in {RUNS_DIR}")

    return max(candidates, key=lambda path: path.name)


def latest_timestamped_file(directory: Path, pattern: str, regex: re.Pattern[str]) -> Path:
    candidates = [path for path in directory.glob(pattern) if path.is_file() and regex.match(path.name)]
    if not candidates:
        raise FileNotFoundError(f"No files found for {directory / pattern}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def latest_global_file(pattern: str, regex: re.Pattern[str]) -> Path:
    candidates = [path for path in PROCESSED_DIR.rglob(pattern) if path.is_file() and regex.match(path.name)]
    if not candidates:
        raise FileNotFoundError(f"No files found for {PROCESSED_DIR / pattern}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def resolve_existing_path(args_value: str | None) -> Path | None:
    if not args_value:
        return None
    path = Path(args_value)
    return path if path.exists() else None


def resolve_input_paths(args: argparse.Namespace) -> dict[str, str]:
    summary = resolve_existing_path(args.summary)
    group = resolve_existing_path(args.group)
    knockout = resolve_existing_path(args.knockout)

    base_dir = next((path.parent for path in (summary, group, knockout) if path is not None), None)
    if base_dir is None:
        try:
            base_dir = latest_run_dir()
        except FileNotFoundError:
            base_dir = None

    if summary is None:
        summary = (
            latest_timestamped_file(base_dir, "wc2026_monte_carlo_summary_*.csv", SUMMARY_RE)
            if base_dir is not None
            else latest_global_file("wc2026_monte_carlo_summary_*.csv", SUMMARY_RE)
        )

    if group is None:
        try:
            group = (
                latest_timestamped_file(base_dir, "wc2026_group_sample_*.csv", GROUP_RE)
                if base_dir is not None
                else latest_global_file("wc2026_group_sample_*.csv", GROUP_RE)
            )
        except FileNotFoundError:
            group = (
                latest_timestamped_file(base_dir, "wc2026_dynamic_feature_sample_*.csv", FEATURES_RE)
                if base_dir is not None
                else latest_global_file("wc2026_dynamic_feature_sample_*.csv", FEATURES_RE)
            )

    if knockout is None:
        knockout = (
            latest_timestamped_file(base_dir, "wc2026_knockout_sample_*.csv", KNOCKOUT_RE)
            if base_dir is not None
            else latest_global_file("wc2026_knockout_sample_*.csv", KNOCKOUT_RE)
        )

    return {
        "summary": str(summary),
        "group": str(group),
        "knockout": str(knockout),
    }


def clean_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].map(lambda value: value.strip() if isinstance(value, str) else value)
    return df


def group_table(group_teams: list[str], matches: pd.DataFrame) -> list[dict]:
    stats = {
        team: {"team": team, "pts": 0, "j": 0, "v": 0, "e": 0, "d": 0, "gm": 0, "gs": 0, "dg": 0}
        for team in group_teams
    }

    for _, match in matches.iterrows():
        home = match["home_team"]
        away = match["away_team"]
        hg = int(match["home_goals"])
        ag = int(match["away_goals"])

        stats[home]["j"] += 1
        stats[away]["j"] += 1
        stats[home]["gm"] += hg
        stats[home]["gs"] += ag
        stats[away]["gm"] += ag
        stats[away]["gs"] += hg

        if hg > ag:
            stats[home]["pts"] += 3
            stats[home]["v"] += 1
            stats[away]["d"] += 1
        elif ag > hg:
            stats[away]["pts"] += 3
            stats[away]["v"] += 1
            stats[home]["d"] += 1
        else:
            stats[home]["pts"] += 1
            stats[away]["pts"] += 1
            stats[home]["e"] += 1
            stats[away]["e"] += 1

    for row in stats.values():
        row["dg"] = row["gm"] - row["gs"]

    return sorted(stats.values(), key=lambda r: (-r["pts"], -r["dg"], -r["gm"], r["team"]))


def build_groups(groups: pd.DataFrame, group_samples: pd.DataFrame) -> list[dict]:
    group_stage = group_samples[group_samples["phase"].eq("Group stage")].copy()
    simulations = sorted(group_stage["simulation"].dropna().unique().astype(int))
    team_to_group = dict(zip(groups["team"], groups["group"]))
    group_to_teams = groups.groupby("group")["team"].apply(list).to_dict()
    output = []

    for group in sorted(group_to_teams):
        orders = Counter()
        table_by_order = {}
        position_counts = {team: Counter() for team in group_to_teams[group]}

        for simulation in simulations:
            sim_matches = group_stage[group_stage["simulation"].eq(simulation)]
            group_matches = sim_matches[sim_matches["home_team"].map(team_to_group).eq(group)]
            table = group_table(group_to_teams[group], group_matches)
            order = tuple(row["team"] for row in table)
            orders[order] += 1
            table_by_order.setdefault(order, table)
            for index, row in enumerate(table, start=1):
                position_counts[row["team"]][index] += 1

        most_common_order, count = orders.most_common(1)[0]
        probabilities = [
            {
                "team": team,
                "positions": {
                    str(position): position_counts[team][position] / max(len(simulations), 1)
                    for position in range(1, 5)
                },
            }
            for team in group_to_teams[group]
        ]
        probabilities.sort(
            key=lambda row: (
                -row["positions"]["1"],
                -row["positions"]["2"],
                -row["positions"]["3"],
                row["team"],
            )
        )
        output.append(
            {
                "group": group,
                "teams": group_to_teams[group],
                "mostLikelyTable": table_by_order[most_common_order],
                "positionProbabilities": probabilities,
                "matches": build_group_matches(group, group_to_teams[group], group_stage, team_to_group),
                "confidence": count / max(len(simulations), 1),
            }
        )

    return output


def build_group_matches(
    group: str,
    group_teams: list[str],
    group_stage: pd.DataFrame,
    team_to_group: dict[str, str],
) -> list[dict]:
    group_rows = group_stage[group_stage["home_team"].map(team_to_group).eq(group)].copy()
    match_keys = (
        group_rows[["home_team", "away_team"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    matches = []

    for home, away in match_keys:
        rows = group_rows[group_rows["home_team"].eq(home) & group_rows["away_team"].eq(away)]
        total = max(len(rows), 1)
        home_wins = int((rows["home_goals"] > rows["away_goals"]).sum())
        draws = int((rows["home_goals"] == rows["away_goals"]).sum())
        away_wins = int((rows["away_goals"] > rows["home_goals"]).sum())
        score_counts = Counter(zip(rows["home_goals"].astype(int), rows["away_goals"].astype(int)))
        score, score_count = score_counts.most_common(1)[0]

        matches.append(
            {
                "home": home,
                "away": away,
                "homeWin": home_wins / total,
                "draw": draws / total,
                "awayWin": away_wins / total,
                "mostLikelyScore": f"{score[0]}-{score[1]}",
                "scoreConfidence": score_count / total,
            }
        )

    return matches


def most_common_match_rows(knockout: pd.DataFrame, slots: pd.DataFrame) -> list[dict]:
    rows = []
    slot_lookup = slots.set_index("match_id").to_dict(orient="index")

    for match_id in sorted(knockout["match_id"].dropna().unique().astype(int)):
        match_rows = knockout[knockout["match_id"].eq(match_id)]
        pairing_counts = Counter(zip(match_rows["home_team"], match_rows["away_team"]))
        outcome_counts = Counter(zip(match_rows["home_team"], match_rows["away_team"], match_rows["winner"]))
        winner_counts = Counter(match_rows["winner"])
        home, away = pairing_counts.most_common(1)[0][0]
        pairing_count = pairing_counts.most_common(1)[0][1]
        pairing_rows = match_rows[match_rows["home_team"].eq(home) & match_rows["away_team"].eq(away)]
        pairing_winner_counts = Counter(pairing_rows["winner"])
        winner = pairing_winner_counts.most_common(1)[0][0]
        winner_count = pairing_winner_counts[winner]
        slot = slot_lookup.get(match_id, {})

        rows.append(
            {
                "matchId": int(match_id),
                "round": match_rows["round"].iloc[0],
                "home": home,
                "away": away,
                "winner": winner,
                "pairingConfidence": pairing_count / len(match_rows),
                "winnerConfidence": winner_count / max(len(pairing_rows), 1),
                "homePassProbability": pairing_winner_counts[home] / max(len(pairing_rows), 1),
                "awayPassProbability": pairing_winner_counts[away] / max(len(pairing_rows), 1),
                "homeOverallPassProbability": winner_counts[home] / len(match_rows),
                "awayOverallPassProbability": winner_counts[away] / len(match_rows),
                "slotHome": slot.get("slot_home", ""),
                "slotAway": slot.get("slot_away", ""),
                "venue": slot.get("venue", ""),
                "dateUtc": slot.get("date_utc", ""),
            }
        )

    return rows


def build_payload(args: argparse.Namespace) -> dict:
    paths = resolve_input_paths(args)
    summary = clean_frame(Path(paths["summary"]))
    groups = clean_frame(Path(args.groups))
    group_samples = clean_frame(Path(paths["group"]))
    knockout = clean_frame(Path(paths["knockout"]))
    slots = clean_frame(Path(args.slots))

    teams = []
    for _, row in summary.sort_values("p_champion", ascending=False).iterrows():
        metrics = [
            {"key": key, "label": label, "value": float(row[key])}
            for key, label in ROUND_LABELS.items()
            if key in row
        ]
        teams.append(
            {
                "team": row["team"],
                "group": groups.loc[groups["team"].eq(row["team"]), "group"].iloc[0],
                "rankFinal": len(teams) + 1,
                "metrics": metrics,
            }
        )

    return {
        "generatedFrom": {
            "summary": paths["summary"],
            "group": paths["group"],
            "knockout": paths["knockout"],
        },
        "teams": teams,
        "groups": build_groups(groups, group_samples),
        "bracket": most_common_match_rows(knockout, slots),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary")
    parser.add_argument("--group", "--features", dest="group")
    parser.add_argument("--knockout")
    parser.add_argument("--groups", default="data/raw/groups.csv")
    parser.add_argument("--slots", default="data/raw/knockout_slots.csv")
    parser.add_argument("--output", default="public/dashboard-data.json")
    args = parser.parse_args()

    payload = build_payload(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
