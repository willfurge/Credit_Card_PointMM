"""
Generate PNG charts from data/sim_history.csv for README screenshots.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DEF_CSV = Path("data/sim_history.csv")
OUT_DIR = Path("assets")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main(csv_path: str | None = None) -> None:
    p = Path(csv_path) if csv_path else DEF_CSV
    if not p.exists():
        raise SystemExit(f"CSV not found: {p} (run the simulation first)")
    df = pd.read_csv(p)

    # 1) Reserve ratio
    fig, ax = plt.subplots()
    ax.plot(df["day"], df["reserve_ratio"])
    ax.set_title("Reserve Ratio Over Time")
    ax.set_xlabel("Day")
    ax.set_ylabel("Ratio")
    ax.grid(True, alpha=0.3)
    fig.savefig(OUT_DIR / "reserve_ratio.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # 2) Quotes (redeem & issue)
    fig, ax = plt.subplots()
    ax.plot(df["day"], df["new_redeem_cpp"] * 100, label="Redeem ¢/pt")
    ax.set_xlabel("Day")
    ax.set_ylabel("¢/pt")
    axb = ax.twinx()
    axb.plot(df["day"], df["new_earn_rate"], label="Issue pts/$", linestyle="--")
    axb.set_ylabel("pts/$")
    ax.set_title("Quotes Over Time")
    ax.grid(True, alpha=0.3)
    fig.savefig(OUT_DIR / "quotes.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # 3) Cash & Liability
    fig, ax = plt.subplots()
    ax.plot(df["day"], df["reserve_cash"] / 1e6, label="Reserve Cash ($m)")
    ax.plot(df["day"], df["liability_points"] / 1e6, label="Liability (pts, millions)")
    ax.set_title("Cash & Liability")
    ax.set_xlabel("Day")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(OUT_DIR / "cash_liability.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote images to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
