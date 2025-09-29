import io
import json
from dataclasses import asdict
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.market_maker.core import MarketMaker, MarketMakerParams, State
from src.market_maker.sim import SimConfig, run_sim

st.set_page_config(page_title="Credit Card Points Market Maker", layout="wide")

st.title("Credit Card Points Market Maker — Interactive Demo")

with st.sidebar:
    st.header("Simulation Controls")

    # Initial State
    st.subheader("Initial State")
    reserve_cash = st.number_input(
        "Reserve Cash ($)",
        min_value=0.0,
        value=5_000_000.0,
        step=100_000.0,
        format="%.2f",
    )
    liability_points = st.number_input(
        "Outstanding Points",
        min_value=0.0,
        value=450_000_000.0,
        step=1_000_000.0,
        format="%.0f",
    )
    earn_rate_ppd = st.slider("Earn Rate (points per $)", 0.5, 5.0, 1.5, 0.1)
    redeem_cpp = (
        st.slider("Redeem $/point (¢/pt)", 0.5, 2.5, 1.1, 0.1) / 100.0
    )  # slider in cents
    ema_cost_cpp = st.slider("EMA Cost (¢/pt)", 0.4, 3.0, 1.2, 0.1) / 100.0

    # Controller Params
    st.subheader("Controller")
    target_reserve_ratio = st.slider("Target Reserve Ratio", 0.30, 0.90, 0.60, 0.01)
    band_width = st.slider("Band Width", 0.05, 0.25, 0.12, 0.01)
    interchange_rate = st.slider("Interchange Rate (%)", 0.0, 3.0, 1.8, 0.1) / 100.0
    elasticity_redeem = st.slider("Redemption Elasticity", 0.5, 4.0, 2.0, 0.1)
    ema_alpha = st.slider("EMA Alpha", 0.01, 0.5, 0.15, 0.01)
    k_redeem = st.slider("K Redeem", 0.0, 1.0, 0.40, 0.05)
    k_issue = st.slider("K Issue", 0.0, 1.0, 0.25, 0.05)
    redeem_lo = st.slider("Redeem Floor (¢/pt)", 0.1, 3.0, 0.5, 0.1) / 100.0
    redeem_hi = st.slider("Redeem Cap (¢/pt)", 0.5, 5.0, 2.5, 0.1) / 100.0
    earn_lo = st.slider("Issue Floor (pts/$)", 0.1, 5.0, 0.5, 0.1)
    earn_hi = st.slider("Issue Cap (pts/$)", 0.5, 10.0, 5.0, 0.1)
    max_redeem_step = (
        st.slider("Max Redeem Step (¢/pt per day)", 0.01, 0.5, 0.15, 0.01) / 100.0
    )
    max_issue_step = st.slider("Max Issue Step (pts/$ per day)", 0.01, 2.0, 0.25, 0.01)

    # Simulation
    st.subheader("Simulation")
    days = st.slider("Days", 30, 365, 120, 10)
    seed = st.number_input("Seed", min_value=0, value=42, step=1)
    base_redeem_rr = (
        st.slider("Baseline Redeem Rate (‰ per day)", 0.0, 5.0, 3.0, 0.1) / 1000.0
    )
    cost_cpp_mean = st.slider("Cost Mean (¢/pt)", 0.4, 3.0, 1.2, 0.1) / 100.0
    cost_cpp_sigma = st.slider("Cost Sigma (¢/pt)", 0.01, 1.0, 0.15, 0.01) / 100.0
    spend_mean = st.number_input(
        "Daily Spend Mean ($)",
        min_value=100.0,
        value=10_000.0,
        step=100.0,
        format="%.2f",
    )
    spend_std = st.number_input(
        "Daily Spend Std ($)", min_value=0.0, value=2_500.0, step=100.0, format="%.2f"
    )

    run_btn = st.button("Run Simulation", type="primary")


def lognormal_params(mean: float, std: float) -> tuple[float, float]:
    if mean <= 0:
        raise ValueError("mean must be positive")
    if std < 0:
        raise ValueError("std must be non-negative")
    if std == 0:
        return (float(np.log(mean)), 1e-9)
    sigma2 = float(np.log(1.0 + (std * std) / (mean * mean)))
    return float(np.log(mean) - 0.5 * sigma2), float(np.sqrt(sigma2))


def run_once() -> dict[str, Any]:
    mu, sigma = lognormal_params(spend_mean, spend_std)
    init = State(
        reserve_cash=reserve_cash,
        liability_points=liability_points,
        earn_rate_ppd=earn_rate_ppd,
        redeem_cpp=redeem_cpp,
        ema_cost_cpp=ema_cost_cpp,
    )
    params = MarketMakerParams(
        target_reserve_ratio=target_reserve_ratio,
        band_width=band_width,
        interchange_rate=interchange_rate,
        elasticity_redeem=elasticity_redeem,
        ema_alpha=ema_alpha,
        k_redeem=k_redeem,
        k_issue=k_issue,
        redeem_cpp_bounds=(min(redeem_lo, redeem_hi), max(redeem_lo, redeem_hi)),
        earn_rate_bounds=(min(earn_lo, earn_hi), max(earn_lo, earn_hi)),
        max_redeem_step=max_redeem_step,
        max_issue_step=max_issue_step,
    )
    simcfg = SimConfig(
        days=days,
        seed=int(seed),
        spend_mu=mu,
        spend_sigma=sigma,
        baseline_redeem_rate=base_redeem_rr,
        cost_cpp_mean=cost_cpp_mean,
        cost_cpp_sigma=cost_cpp_sigma,
    )
    return run_sim(simcfg, init, params)


if run_btn:
    result = run_once()
    hist = pd.DataFrame(result["history"])

    st.success("Simulation complete.")
    c1, c2, c3 = st.columns(3)

    # Plot 1: Reserve ratio
    with c1:
        fig1, ax1 = plt.subplots()
        ax1.plot(hist["day"], hist["reserve_ratio"])
        ax1.set_title("Reserve Ratio")
        ax1.set_xlabel("Day")
        ax1.set_ylabel("Ratio")
        ax1.grid(True, alpha=0.3)
        st.pyplot(fig1, clear_figure=True)

    # Plot 2: Redemption price & Issue rate
    with c2:
        fig2, ax2 = plt.subplots()
        ax2.plot(hist["day"], hist["new_redeem_cpp"] * 100, label="Redeem ¢/pt")
        ax2.set_xlabel("Day")
        ax2.set_ylabel("¢/pt")
        ax2.grid(True, alpha=0.3)
        ax2b = ax2.twinx()
        ax2b.plot(hist["day"], hist["new_earn_rate"], label="Issue pts/$")
        ax2b.set_ylabel("pts/$")
        ax2.set_title("Quotes")
        st.pyplot(fig2, clear_figure=True)

    # Plot 3: Cash & Liability
    with c3:
        fig3, ax3 = plt.subplots()
        ax3.plot(hist["day"], hist["reserve_cash"] / 1e6, label="Reserve Cash ($m)")
        ax3.plot(
            hist["day"],
            hist["liability_points"] / 1e6,
            label="Liability (pts, millions)",
        )
        ax3.set_title("Cash & Liability")
        ax3.set_xlabel("Day")
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        st.pyplot(fig3, clear_figure=True)

    # Download buttons
    csv_buf = io.StringIO()
    hist.to_csv(csv_buf, index=False)
    st.download_button(
        "Download CSV", csv_buf.getvalue(), file_name="sim_history.csv", mime="text/csv"
    )

    json_buf = io.StringIO()
    json.dump(
        result, json_buf, default=lambda o: getattr(o, "__dict__", str(o)), indent=2
    )
    st.download_button(
        "Download JSON",
        json_buf.getvalue(),
        file_name="sim_output.json",
        mime="application/json",
    )

else:
    st.info("Adjust parameters in the sidebar, then click **Run Simulation**.")
