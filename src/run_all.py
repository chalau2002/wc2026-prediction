from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Tem de ser um numero inteiro positivo.") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Tem de ser maior que zero.")
    return parsed


def ask_n_sims() -> int:
    while True:
        raw = input("Quantas simulacoes Monte Carlo queres correr? ").strip()
        try:
            return positive_int(raw)
        except argparse.ArgumentTypeError as exc:
            print(f"Valor invalido: {exc}")


def run_step(name: str, command: list[str], log_path: Path) -> None:
    print(f"\n=== {name} ===")
    print(" ".join(command))

    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)

    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"{name} falhou com exit code {return_code}. Log: {log_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Corre a pipeline completa: Elo, features, treino do modelo e simulacao Monte Carlo."
    )
    parser.add_argument(
        "--n-sims",
        type=positive_int,
        default=100,
        help="Numero de simulacoes Monte Carlo. Default: 1000.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-sample-sims", type=int, default=-1,
        help="Numero de simulacoes a guardar para detalhes de grupo/jogo. -1 = todas. Default: -1.")
    parser.add_argument(
        "--progress-every",
        type=positive_int,
        default=10,
        help="Mostra progresso da simulacao a cada N simulacoes.",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Pasta para os resultados da simulacao. Default: data/processed/monte_carlo_runs/<timestamp>",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    n_sims = args.n_sims

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.run_dir) if args.run_dir else Path("data/processed/monte_carlo_runs") / timestamp
    run_dir = run_dir if run_dir.is_absolute() else PROJECT_ROOT / run_dir
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    python = sys.executable
    steps = [
        ("1_elo", [python, "src/elo.py"]),
        ("2_features", [python, "src/features.py"]),
        ("3_model", [python, "src/model.py"]),
        (
            "4_simulate",
            [
                python,
                "src/simulate.py",
                "--n-sims",
                str(n_sims),
                "--seed",
                str(args.seed),
                "--timestamp",
                timestamp,
                "--save-sample-sims",
                str(args.save_sample_sims),
                "--progress-every",
                str(args.progress_every),
                "--output-dir",
                str(run_dir),
            ],
        ),
        (
            "5_dashboard",
            [
                python,
                "scripts/build_dashboard_data.py",
                "--summary",
                str(run_dir / f"wc2026_monte_carlo_summary_{timestamp}.csv"),
                "--group",
                str(run_dir / f"wc2026_group_sample_{timestamp}.csv"),
                "--knockout",
                str(run_dir / f"wc2026_knockout_sample_{timestamp}.csv"),
                "--output",
                "public/dashboard-data.json",
            ],
        ),
    ]

    print(f"Run dir: {run_dir}")
    print(f"Simulacoes Monte Carlo: {n_sims}")

    for step_name, command in steps:
        run_step(step_name, command, logs_dir / f"{step_name}.log")

    print("\nPipeline concluida.")
    print(f"Resultados da simulacao: {run_dir}")
    print(f"Logs: {logs_dir}")


if __name__ == "__main__":
    main()
