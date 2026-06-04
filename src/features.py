import pandas as pd
import numpy as np
from pathlib import Path

H2H_YEARS = 4
ROLLING_WINDOW = 10

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


def build_rolling_stats(results: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    results = results.copy()
    results["date"] = pd.to_datetime(results["date"])
    results = results.sort_values("date").reset_index(drop=True)
    results["game_idx"] = results.index

    home = results[["date", "game_idx", "home_team", "home_score", "away_score"]].copy()
    home.columns = ["date", "game_idx", "team", "goals_scored", "goals_conceded"]
    home["side"] = "home"

    away = results[["date", "game_idx", "away_team", "away_score", "home_score"]].copy()
    away.columns = ["date", "game_idx", "team", "goals_scored", "goals_conceded"]
    away["side"] = "away"

    long = pd.concat([home, away], ignore_index=True).sort_values(["team", "date", "game_idx"])

    long["avg_goals_scored"] = (
        long.groupby("team")["goals_scored"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )
    long["avg_goals_conceded"] = (
        long.groupby("team")["goals_conceded"]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )

    home_stats = (
        long[long["side"] == "home"]
        [["game_idx", "avg_goals_scored", "avg_goals_conceded"]]
        .rename(columns={
            "avg_goals_scored": "home_avg_goals_scored",
            "avg_goals_conceded": "home_avg_goals_conceded",
        })
    )

    away_stats = (
        long[long["side"] == "away"]
        [["game_idx", "avg_goals_scored", "avg_goals_conceded"]]
        .rename(columns={
            "avg_goals_scored": "away_avg_goals_scored",
            "avg_goals_conceded": "away_avg_goals_conceded",
        })
    )

    stats = results[["date", "game_idx", "home_team", "away_team"]].copy()
    stats = stats.merge(home_stats, on="game_idx", how="left")
    stats = stats.merge(away_stats, on="game_idx", how="left")
    stats = stats.drop(columns=["game_idx"])

    return stats


def build_h2h_stats(results: pd.DataFrame, years: int = H2H_YEARS) -> pd.DataFrame:
    results = results.copy()
    results["date"] = pd.to_datetime(results["date"])
    results = results.sort_values("date").reset_index(drop=True)
    results["game_idx"] = results.index

    home = results[["date", "game_idx", "home_team", "away_team", "home_score", "away_score"]].copy()
    home["team"] = home["home_team"]
    home["opponent"] = home["away_team"]
    home["goals_for"] = home["home_score"]
    home["goals_against"] = home["away_score"]
    home["win"] = (home["home_score"] > home["away_score"]).astype(int)

    away = results[["date", "game_idx", "home_team", "away_team", "home_score", "away_score"]].copy()
    away["team"] = away["away_team"]
    away["opponent"] = away["home_team"]
    away["goals_for"] = away["away_score"]
    away["goals_against"] = away["home_score"]
    away["win"] = (away["away_score"] > away["home_score"]).astype(int)

    long = pd.concat([home, away], ignore_index=True)
    long = long[["date", "game_idx", "team", "opponent", "goals_for", "goals_against", "win"]]
    long = long.sort_values("date").reset_index(drop=True)

    records = []

    long["cutoff"] = long["date"] - pd.DateOffset(years=years)
    pair_groups = long.groupby(["team", "opponent"])

    for (team, opponent), grp in pair_groups:
        grp = grp.sort_values("date").reset_index(drop=True)

        for i, row in grp.iterrows():
            date = row["date"]
            cutoff = row["cutoff"]

            mask = (grp["date"] >= cutoff) & (grp["date"] < date)
            h2h = grp[mask]

            records.append({
                "game_idx": row["game_idx"],
                "team": team,
                "h2h_matches": len(h2h),
                "h2h_goals_for": h2h["goals_for"].mean() if len(h2h) > 0 else np.nan,
                "h2h_goals_against": h2h["goals_against"].mean() if len(h2h) > 0 else np.nan,
                "h2h_win_rate": h2h["win"].mean() if len(h2h) > 0 else np.nan,
            })

    feat = pd.DataFrame(records)

    home_feat = feat[feat["game_idx"].isin(results["game_idx"])].copy()

    home_side = results[["game_idx", "home_team"]].copy()
    away_side = results[["game_idx", "away_team"]].copy()

    home_h2h = feat.merge(home_side, left_on=["game_idx", "team"], right_on=["game_idx", "home_team"], how="inner") \
        .rename(columns={
            "h2h_matches": "h2h_matches",
            "h2h_goals_for": "h2h_home_avg_goals",
            "h2h_goals_against": "h2h_away_avg_goals",
            "h2h_win_rate": "h2h_home_win_rate",
        })[["game_idx", "h2h_matches", "h2h_home_avg_goals", "h2h_away_avg_goals", "h2h_home_win_rate"]]

    result = results[["date", "game_idx", "home_team", "away_team"]].merge(
        home_h2h, on="game_idx", how="left"
    ).drop(columns=["game_idx"])

    return result

def build_wc_experience(results: pd.DataFrame) -> pd.DataFrame:
    df = results.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["game_idx"] = df.index

    home = df[["date", "game_idx", "home_team", "tournament"]].copy()
    home["team"] = home["home_team"]
    home["side"] = "home"

    away = df[["date", "game_idx", "away_team", "tournament"]].copy()
    away["team"] = away["away_team"]
    away["side"] = "away"

    long = pd.concat([home, away], ignore_index=True)
    long["is_wc"] = (long["tournament"] == "FIFA World Cup").astype(int)
    long = long.sort_values(["team", "date", "game_idx"])

    records = []

    for team, grp in long.groupby("team"):
        grp = grp.sort_values(["date", "game_idx"]).copy()
        wc_participations = 0
        in_current_wc = False

        for _, row in grp.iterrows():
            records.append({
                "game_idx": row["game_idx"],
                "side": row["side"],
                "team": team,
                "wc_participations": wc_participations,
                "wc_has_experience": int(wc_participations > 0),
            })

            if row["is_wc"] == 1:
                if not in_current_wc:
                    wc_participations += 1
                    in_current_wc = True
            else:
                in_current_wc = False

    long_feat = pd.DataFrame(records)

    home_feat = (
        long_feat[long_feat["side"] == "home"]
        [["game_idx", "wc_participations", "wc_has_experience"]]
        .rename(columns={
            "wc_participations": "home_wc_participations",
            "wc_has_experience": "home_wc_has_experience",
        })
    )

    away_feat = (
        long_feat[long_feat["side"] == "away"]
        [["game_idx", "wc_participations", "wc_has_experience"]]
        .rename(columns={
            "wc_participations": "away_wc_participations",
            "wc_has_experience": "away_wc_has_experience",
        })
    )

    base = df[["date", "game_idx", "home_team", "away_team"]].copy()
    base = base.merge(home_feat, on="game_idx", how="left")
    base = base.merge(away_feat, on="game_idx", how="left")

    wc_cols = [c for c in base.columns if "wc_" in c]
    base[wc_cols] = base[wc_cols].fillna(0)

    return base.drop(columns=["game_idx"])

def classify_competition(tournament: str) -> str:
    if tournament in WORLD_CUP_FINALS:
        return "world_cup"

    if tournament in CONTINENTAL_FINALS:
        return "continental"

    if tournament in QUALIFIERS:
        return "qualifier"

    if tournament in FRIENDLIES:
        return "friendly"

    return "other"


def build_context_features(results: pd.DataFrame) -> pd.DataFrame:
    results = results.copy()
    results["date"] = pd.to_datetime(results["date"])
    results = results.sort_values("date").reset_index(drop=True)
    results["game_idx"] = results.index

    results["is_home_home"] = (results["home_team"] == results["country"]).astype(int)
    results["is_home_away"] = (results["away_team"] == results["country"]).astype(int)
    results["is_neutral"] = results["neutral"].astype(int)

    results["competition_type"] = results["tournament"].apply(classify_competition)

    results["competition_friendly"] = (results["competition_type"] == "friendly").astype(int)
    results["competition_qualifier"] = (results["competition_type"] == "qualifier").astype(int)
    results["competition_continental"] = (results["competition_type"] == "continental").astype(int)
    results["competition_world_cup"] = (results["competition_type"] == "world_cup").astype(int)

    return results[[
        "date", "game_idx", "home_team", "away_team",
        "is_home_home", "is_home_away", "is_neutral",
        "competition_friendly", "competition_qualifier",
        "competition_continental", "competition_world_cup",
    ]].drop(columns=["game_idx"])


def build_training_df(results, elo_matches):
    results = results.copy()
    results["date"] = pd.to_datetime(results["date"])
    results = results.sort_values("date").reset_index(drop=True)
    results["game_idx"] = results.index

    elo = elo_matches[["date", "home_team", "away_team",
                        "home_elo_before", "away_elo_before"]].copy()
    elo["date"] = pd.to_datetime(elo["date"])
    elo["elo_diff"] = elo["home_elo_before"] - elo["away_elo_before"]

    print("A calcular rolling stats...")
    rolling = build_rolling_stats(results)

    print("A calcular head to head...")
    h2h = build_h2h_stats(results)

    print("A calcular experiência em mundiais...")
    wc_exp = build_wc_experience(results)

    print("A calcular features de contexto...")
    context = build_context_features(results)
  
    # Restore game_idx on helper dataframes after feature builders drop it.
    key = results[["date", "game_idx", "home_team", "away_team"]]

    def add_game_idx(df):
        return df.merge(key, on=["date", "home_team", "away_team"], how="left")

    rolling = add_game_idx(rolling).drop(columns=["date", "home_team", "away_team"])
    h2h = add_game_idx(h2h).drop(columns=["date", "home_team", "away_team"])
    wc_exp = add_game_idx(wc_exp).drop(columns=["date", "home_team", "away_team"])
    context = add_game_idx(context).drop(columns=["date", "home_team", "away_team"])

    df = results[[
        "date", "game_idx", "home_team", "away_team",
        "home_score", "away_score",
        "tournament", "city", "country", "neutral",
    ]].copy()

    df = df.merge(elo, on=["date", "home_team", "away_team"], how="left")
    df = df.merge(rolling, on="game_idx", how="left")
    df = df.merge(h2h, on="game_idx", how="left")
    df = df.merge(wc_exp, on="game_idx", how="left")
    df = df.merge(context, on="game_idx", how="left")

    df = df.drop(columns=["game_idx"])

    df = df.drop_duplicates()

    print(f"\nTraining df: {df.shape[0]} jogos x {df.shape[1]} features")
    print(f"NaNs por coluna:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

    return df


if __name__ == "__main__":
    results = pd.read_csv("data/raw/results.csv")
    results["date"] = pd.to_datetime(results["date"])
    results = results.drop_duplicates(subset=["date", "home_team", "away_team"], keep="first")
    results = results.sort_values("date").reset_index(drop=True)

    elo_matches = pd.read_csv("data/processed/elo_matches.csv")
    elo_matches["date"] = pd.to_datetime(elo_matches["date"])
    elo_matches = elo_matches.drop_duplicates(subset=["date", "home_team", "away_team"], keep="first")


    training_df = build_training_df(results, elo_matches)
    training_df.to_csv("data/processed/training_df.csv", index=False)
    print("\nGuardado em data/processed/training_df.csv")
