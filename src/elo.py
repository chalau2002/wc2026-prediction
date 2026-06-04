import pandas as pd
import numpy as np

ELO_INIT = 1000
HOME_ADVANTAGE_VALUE = 100

K_VALUES = {
    "world_cup_final": 60,
    "continental_final": 50,
    "qualifier_major": 40,
    "other_tournament": 30,
    "friendly": 20
}

WORLD_CUP_FINALS = {"FIFA World Cup"}
CONTINENTAL_FINALS = {
    "UEFA Euro",
    "Copa América",
    "African Cup of Nations",
    "AFC Asian Cup",
    "Gold Cup",
    "UEFA Nations League"
}
QUALIFIERS = {
    "FIFA World Cup qualification",
    "UEFA Euro qualification",
    "Copa América qualification",
    "African Cup of Nations qualification",
    "AFC Asian Cup qualification"
}
FRIENDLIES = {"Friendly"}

DECAY_RATE = 0.95
DECAY_ENABLED = True


def preprocess(df: pd.DataFrame, start_year: int):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"].dt.year >= start_year]

    if "neutral" not in df.columns:
        df["neutral"] = False
    else:
        df["neutral"] = df["neutral"].fillna(False).astype(bool)

    before = len(df)
    df = df.dropna(subset=["home_score", "away_score"])
    dropped = before - len(df)
    if dropped > 0:
        print(f"[preprocess] Removidas {dropped} linhas com scores nulos.")

    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    return df.sort_values("date").reset_index(drop=True)


def classify_match(tournament: str):
    if tournament in WORLD_CUP_FINALS:
        return "world_cup_final"
    if tournament in CONTINENTAL_FINALS:
        return "continental_final"
    if tournament in QUALIFIERS:
        return "qualifier_major"
    if tournament in FRIENDLIES:
        return "friendly"
    return "other_tournament"


def k_factor(tournament: str, use_competition_weights: bool):
    if not use_competition_weights:
        return 30
    category = classify_match(tournament)
    return K_VALUES.get(category, K_VALUES["other_tournament"])


def expected_score(r_a, r_b):
    return 1 / (1 + 10 ** ((r_b - r_a) / 400))


def goal_diff_multiplier(gd: int):
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    if gd == 3:
        return 1.75
    return 1.75 + (gd - 3) / 8


def apply_decay(elo: dict, active_teams: set):
    """
    Decay inactive teams toward the initial Elo rating.
    """
    for team in elo:
        if team not in active_teams:
            elo[team] = elo[team] * DECAY_RATE + ELO_INIT * (1 - DECAY_RATE)
    return elo


def compute_elo(
    df: pd.DataFrame,
    start_year: int = 1872,
    use_home_advantage: bool = True,
    use_competition_weights: bool = True
):
    df = preprocess(df, start_year)

    elo = {}
    records = []

    current_year = None
    year_active_teams = set()

    for row in df.itertuples(index=False):

        row_year = row.date.year

        if DECAY_ENABLED and current_year is not None and row_year != current_year:
            elo = apply_decay(elo, year_active_teams)
            year_active_teams = set()

        current_year = row_year

        home = row.home_team
        away = row.away_team

        elo.setdefault(home, ELO_INIT)
        elo.setdefault(away, ELO_INIT)

        year_active_teams.add(home)
        year_active_teams.add(away)

        r_home = elo[home]
        r_away = elo[away]

        is_neutral = row.neutral
        if use_home_advantage and not is_neutral:
            r_home_adj = r_home + HOME_ADVANTAGE_VALUE
        else:
            r_home_adj = r_home

        we_home = expected_score(r_home_adj, r_away)
        we_away = 1 - we_home

        if row.home_score > row.away_score:
            w_home, w_away = 1.0, 0.0
        elif row.home_score < row.away_score:
            w_home, w_away = 0.0, 1.0
        else:
            w_home, w_away = 0.5, 0.5

        k = k_factor(row.tournament, use_competition_weights)

        gd = abs(row.home_score - row.away_score)
        k_adj = k * goal_diff_multiplier(gd)

        # Apply the update to base ratings, not the home-advantage-adjusted rating.
        new_home = r_home + k_adj * (w_home - we_home)
        new_away = r_away + k_adj * (w_away - we_away)

        elo[home] = new_home
        elo[away] = new_away

        records.append({
            "date": row.date,
            "home_team": home,
            "away_team": away,
            "tournament": row.tournament,
            "neutral": is_neutral,
            "home_elo_before": r_home,
            "away_elo_before": r_away,
            "home_elo_after": new_home,
            "away_elo_after": new_away,
            "goal_diff": gd,
            "k_base": k,
            "k_used": k_adj
        })

    elo_df = pd.DataFrame(list(elo.items()), columns=["team", "elo"]).sort_values("elo", ascending=False)
    matches_df = pd.DataFrame(records)

    return matches_df, elo_df


def audit_tournaments(df: pd.DataFrame):
    known = WORLD_CUP_FINALS | CONTINENTAL_FINALS | QUALIFIERS | FRIENDLIES
    unknown = df[~df["tournament"].isin(known)]["tournament"].value_counts()
    print("\n[audit] Torneios a cair em 'other_tournament' (K=15):")
    print(unknown.to_string())


if __name__ == "__main__":
    df = pd.read_csv("data/raw/results.csv")

    audit_tournaments(df)

    elo_matches, final_elo = compute_elo(
        df,
        start_year=1800,
        use_home_advantage=True,
        use_competition_weights=True
    )

    print("\nTop 30 rankings finais:")
    print(final_elo.head(30).to_string(index=False))

    elo_matches.to_csv("data/processed/elo_matches.csv", index=False)
    final_elo.to_csv("data/processed/final_elo.csv", index=False)
