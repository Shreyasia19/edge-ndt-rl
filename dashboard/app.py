from __future__ import annotations

import socket
import threading
import time

import pandas as pd
import streamlit as st

from data_loader import ROOT, artifact_path, exists, read_csv, read_json

st.set_page_config(
    page_title="NDT-RL Experiment Console",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      .stApp { background: #f7f8fa; color: #16202a; }
      .block-container { max-width: 1400px; padding-top: 2rem; }
      [data-testid="stMetric"] {
        background: white;
        border: 1px solid #dbe2ea;
        border-radius: 10px;
        padding: 14px;
      }
      h1, h2, h3 { color: #13253f; }
      .subtitle { color: #607083; margin-top: -0.7rem; margin-bottom: 1.5rem; }
      .source-note {
        color: #607083; font-size: 0.82rem; padding: 0.75rem 0;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def port_is_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex(("127.0.0.1", port)) == 0


@st.cache_resource
def ensure_live_server() -> str:
    """Start the local simulation API once, only when FastAPI is available."""
    if port_is_open(8765):
        return "connected"

    try:
        import uvicorn
        from twin_server import app as twin_api

        thread = threading.Thread(
            target=uvicorn.run,
            kwargs={
                "app": twin_api,
                "host": "127.0.0.1",
                "port": 8765,
                "log_level": "warning",
            },
            daemon=True,
        )
        thread.start()
        time.sleep(0.4)
        return "started" if port_is_open(8765) else "starting"
    except Exception as exc:
        return f"unavailable: {exc}"


def ppo_row(frame: pd.DataFrame | None, volatility: str, label: str):
    if frame is None:
        return None

    rows = frame[
        (frame["volatility"] == volatility)
        & (frame["label"] == label)
    ]

    return rows.iloc[0] if not rows.empty else None


def transfer_table(ppo: pd.DataFrame | None) -> pd.DataFrame | None:
    if ppo is None or ppo.empty:
        return None

    required = {"label", "volatility", "mean_act_ms"}
    if not required.issubset(ppo.columns):
        return None

    pivot = ppo.pivot_table(
        index="volatility",
        columns="label",
        values="mean_act_ms",
        aggfunc="first",
    )

    if "PPO-twin" not in pivot.columns or "PPO-real" not in pivot.columns:
        return None

    result = pivot[["PPO-twin", "PPO-real"]].reset_index()
    result.columns = ["scenario", "twin_act_ms", "real_act_ms"]
    result["gap_pct"] = (
        (result["real_act_ms"] - result["twin_act_ms"])
        / result["twin_act_ms"]
        * 100
    )
    return result


ppo = read_csv("rl_agent/results/ppo_evaluation.csv")
fidelity = read_csv("digital_twin/saved_models/eval_results/fidelity_summary.csv")
benchmarks = read_csv("evaluation/results/benchmark.csv")
transfer = read_csv("gap_analysis/results/transfer_gap.csv")
training_config = read_json("rl_agent/policies/training_config.json")

st.title("NDT-RL Experiment Console")
st.markdown(
    '<div class="subtitle">Data-backed review of the trained policy, twin fidelity, '
    'baseline evaluations, and transfer measurements.</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Experiment data")
    st.code(str(ROOT))
    if st.button("Refresh files"):
        st.rerun()

tabs = st.tabs(
    [
        "Summary",
        "Live monitor",
        "Twin quality",
        "PPO policy",
        "Method comparison",
        "Transfer",
    ]
)

with tabs[0]:
    normal_real = ppo_row(ppo, "normal", "PPO-real")
    normal_twin = ppo_row(ppo, "normal", "PPO-twin")
    normal_fidelity = None

    if fidelity is not None and "volatility" in fidelity.columns:
        rows = fidelity[fidelity["volatility"] == "normal"]
        if not rows.empty:
            normal_fidelity = rows.iloc[0]

    real_act = normal_real["mean_act_ms"] if normal_real is not None else None
    real_sla = normal_real["sla_violation_rate"] if normal_real is not None else None
    r2 = normal_fidelity["r2"] if normal_fidelity is not None else None

    gap = None
    if normal_real is not None and normal_twin is not None:
        gap = (
            (normal_real["mean_act_ms"] - normal_twin["mean_act_ms"])
            / normal_twin["mean_act_ms"]
            * 100
        )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PPO ACT - normal", f"{real_act:,.0f} ms" if real_act is not None else "No result")
    c2.metric("PPO SLA violation", f"{real_sla:.1%}" if real_sla is not None else "No result")
    c3.metric("Twin R2 - normal", f"{r2:.3f}" if r2 is not None else "No result")
    c4.metric("Twin-to-real ACT gap", f"{gap:+.1f}%" if gap is not None else "No result")

    st.subheader("Current evidence")
    st.write(
        "This dashboard shows saved experiment artifacts. It does not invent "
        "baseline, domain-randomization, or fine-tuning results."
    )

    if ppo is not None:
        st.dataframe(ppo, use_container_width=True, hide_index=True)
    else:
        st.warning("Missing `rl_agent/results/ppo_evaluation.csv`.")

    st.markdown(
        '<div class="source-note">Sources: PPO evaluation CSV, twin fidelity CSV, '
        'training configuration JSON.</div>',
        unsafe_allow_html=True,
    )

with tabs[1]:
    state = ensure_live_server()
    st.subheader("Live simulation monitor")
    st.caption(
        "This page is a local NetSim monitor. It is separate from saved PPO/GNN "
        "experiment artifacts and should not be presented as a trained model view."
    )
    st.caption(f"Local simulation API: {state}")

    try:
        from live_twin_tab import render as render_live_twin
        render_live_twin()
    except Exception as exc:
        st.error(f"Live monitor could not render: {exc}")

with tabs[2]:
    st.subheader("Twin fidelity")
    st.caption("Metrics loaded from held-out twin evaluation artifacts.")

    if fidelity is None:
        st.warning("Missing `digital_twin/saved_models/eval_results/fidelity_summary.csv`.")
    else:
        st.dataframe(fidelity, use_container_width=True, hide_index=True)

        if {"volatility", "r2"}.issubset(fidelity.columns):
            st.bar_chart(fidelity.set_index("volatility")["r2"])

        image_columns = st.columns(3)
        for column, scenario in zip(
            image_columns,
            ["normal", "high_churn", "low_bandwidth"],
        ):
            image = artifact_path(
                f"digital_twin/saved_models/eval_results/pred_vs_actual_{scenario}.png"
            )
            if image.exists():
                column.image(str(image), caption=scenario.replace("_", " ").title())

with tabs[3]:
    st.subheader("PPO policy")
    st.caption("Training configuration and saved outputs from the existing PPO run.")

    if training_config:
        left, right = st.columns([1, 2])
        with left:
            st.json(training_config)
        with right:
            curve = artifact_path("rl_agent/results/ppo_training_curves.png")
            if curve.exists():
                st.image(str(curve), caption="PPO training curves")
    else:
        st.warning("Missing `rl_agent/policies/training_config.json`.")

    if ppo is not None:
        st.dataframe(ppo, use_container_width=True, hide_index=True)

with tabs[4]:
    st.subheader("Method comparison")
    st.caption(
        "This page stays empty until the actual baseline evaluator exports results. "
        "It does not use dashboard constants."
    )

    if benchmarks is None:
        st.info(
            "Run `python3 evaluation/benchmark_all.py` to create "
            "`evaluation/results/benchmark.csv`."
        )
    else:
        st.dataframe(benchmarks, use_container_width=True, hide_index=True)

        if {"method", "volatility", "mean_act_ms"}.issubset(benchmarks.columns):
            chart = benchmarks.pivot_table(
                index="volatility",
                columns="method",
                values="mean_act_ms",
                aggfunc="mean",
            )
            st.bar_chart(chart)

with tabs[5]:
    st.subheader("Twin-to-real transfer")
    st.caption(
        "A positive value means the real simulator was slower than the twin prediction."
    )

    computed_transfer = transfer_table(ppo)

    if transfer is not None:
        st.dataframe(transfer, use_container_width=True, hide_index=True)
    elif computed_transfer is not None:
        st.dataframe(computed_transfer, use_container_width=True, hide_index=True)
        st.info(
            "These values come from the current PPO evaluation CSV. Run "
            "`python3 gap_analysis/measure_gap.py` to save a formal transfer artifact."
        )
    else:
        st.warning("Transfer comparison needs both PPO-twin and PPO-real evaluation rows.")
