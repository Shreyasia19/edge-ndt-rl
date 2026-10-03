"""
live_twin_tab.py
Proper Streamlit-native Live Digital Twin tab.
Polls twin_server.py every second via requests.


Import and call render() from inside app.py's Live Digital Twin tab.
"""

import time
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st

SERVER = "http://localhost:8765"
N_NODES = 8
NODE_LABELS = [f"N{i}" for i in range(N_NODES)]

# ── Colors ───────────────────────────────────────────────────────────
REAL_COL  = "#4fd1c5"
TWIN_COL  = "#f0a860"
OK_COL    = "#57c46b"
BAD_COL   = "#e2574c"
OFF_COL   = "#3a4363"
GAP_COL   = "#b48cf2"
RMSE_COL  = "#ef5350"
R2_COL    = "#57c46b"

plt.rcParams.update({
    'figure.facecolor':'#0a0e1a','axes.facecolor':'#0d1b2a',
    'axes.edgecolor':'#1e3a5f','axes.labelcolor':'#90a4ae',
    'xtick.color':'#78909c','ytick.color':'#78909c',
    'text.color':'#cfd8dc','grid.color':'#1a2744','grid.alpha':.5,
    'legend.facecolor':'#0d1b2a','legend.edgecolor':'#1e3a5f',
})


def _fetch_tick(mode: str) -> dict | None:
    """Fetch one live tick from the backend. Returns None on failure."""
    try:
        r = requests.get(f"{SERVER}/sim/{mode}/tick", timeout=2)
        if r.ok:
            return r.json()
    except Exception:
        pass
    return None


st.markdown(
    """
    <div style="background:#0d1b2a;border:1px solid #1e3a5f;
                border-radius:12px;padding:16px 18px;margin-bottom:18px">
      <div style="font-size:1.05rem;font-weight:700;color:#4fc3f7">
        Interactive Network Digital Twin
      </div>
      <div style="font-size:.83rem;color:#90a4ae;margin-top:5px">
        Simulate task arrivals, resource pressure, node churn, and the
        difference between observed network state and twin prediction.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "Select a network condition, start the simulation, and compare the "
    "observed edge-network state with the digital twin prediction."
)

def _node_load_color(load: float, online: bool) -> str:
    """Return a Matplotlib-compatible hex colour from green to red."""
    if not online:
        return OFF_COL

    load = max(0.0, min(1.0, float(load)))
    low = (87, 196, 107)   # green
    high = (226, 87, 76)   # red

    rgb = tuple(round(low[i] + (high[i] - low[i]) * load) for i in range(3))
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _draw_network_canvas(
    loads: list, online: list, queues: list, recv: list,
    tasks_recv: list, color: str, title: str, ax
):
    """Draw one network panel (real OR twin) on a matplotlib axes."""
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor("#0d1b2a")

    cx, cy, R = 0, 0, 0.85
    pos = [
        (cx + R * np.cos((i / N_NODES) * 2 * np.pi - np.pi / 2),
         cy + R * np.sin((i / N_NODES) * 2 * np.pi - np.pi / 2))
        for i in range(N_NODES)
    ]

    # Draw links
    link_pairs = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,0),
                  (0,4),(1,5),(2,6),(3,7)]
    for i, j in link_pairs:
        ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]],
                color="#1e3a5f", linewidth=0.8, zorder=1)

    # Draw nodes
    max_recv = max(tasks_recv) if tasks_recv and max(tasks_recv) > 0 else 1
    for i in range(N_NODES):
        x, y = pos[i]
        load = loads[i]
        is_on = online[i]
        q = queues[i]
        recv_i = recv[i] if recv else 0

        # Node halo (load indicator)
        hue = int((1 - load) * 120) if is_on else 0
        node_col = _node_load_color(load, is_on)

        # Background circle
        circle_bg = plt.Circle((x, y), 0.13, color="#112240",
                                zorder=2, linewidth=0)
        ax.add_patch(circle_bg)

        # Load arc (filled circle proportional to load)
        load_circle = plt.Circle((x, y), 0.11,
                                  color=node_col, alpha=0.85,
                                  zorder=3, linewidth=0)
        ax.add_patch(load_circle)

        # Border ring
        border = plt.Circle((x, y), 0.12, fill=False,
                             edgecolor=color, linewidth=1.5, zorder=4)
        ax.add_patch(border)

        # Node ID
        ax.text(x, y + 0.01, NODE_LABELS[i],
                ha="center", va="center", fontsize=7.5,
                fontweight="bold", color="white", zorder=5)

        # Load %
        status = f"{int(load*100)}%" if is_on else "off"
        ax.text(x, y - 0.19, status,
                ha="center", va="center", fontsize=6.5,
                color="#90a4ae", zorder=5)

        # Queue badge
        if is_on and q > 0:
            badge = plt.Circle((x + 0.09, y + 0.09), 0.055,
                                color=BAD_COL, zorder=6)
            ax.add_patch(badge)
            ax.text(x + 0.09, y + 0.09, str(min(q, 99)),
                    ha="center", va="center", fontsize=5.5,
                    fontweight="bold", color="white", zorder=7)

        # Tasks received bar (tiny bar below node)
        bar_w = 0.18
        bar_h = 0.025
        bx = x - bar_w / 2
        by = y - 0.27
        bar_bg = plt.Rectangle((bx, by), bar_w, bar_h,
                                color="#1e3a5f", zorder=4)
        ax.add_patch(bar_bg)
        filled_w = bar_w * (recv_i / max_recv)
        bar_fill = plt.Rectangle((bx, by), filled_w, bar_h,
                                  color=color, alpha=0.8, zorder=5)
        ax.add_patch(bar_fill)
        ax.text(x + bar_w / 2 + 0.02, by + bar_h / 2,
                str(recv_i), ha="left", va="center",
                fontsize=5.5, color="#78909c", zorder=6)

    ax.set_title(title, color=color, fontsize=9,
                 fontweight="bold", pad=4)


def _draw_sparkline(hist: list, color: str, ylabel: str,
                    zero_base: bool = False) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 1.6))
    ax.set_facecolor("#0d1b2a")
    fig.patch.set_facecolor("#0a0e1a")
    if len(hist) < 2:
        ax.text(0.5, 0.5, "Accumulating data…",
                ha="center", va="center", color="#546e7a",
                transform=ax.transAxes, fontsize=8)
    else:
        x = list(range(len(hist)))
        if zero_base:
            ax.axhline(0, color="#546e7a", linewidth=0.8, alpha=0.6)
            pos = [v if v > 0 else 0 for v in hist]
            neg = [v if v < 0 else 0 for v in hist]
            ax.fill_between(x, pos, alpha=0.3, color=BAD_COL)
            ax.fill_between(x, neg, alpha=0.3, color=OK_COL)
        else:
            ax.fill_between(x, hist, alpha=0.2, color=color)
        ax.plot(x, hist, color=color, linewidth=1.5)
        ax.set_ylabel(ylabel, fontsize=7, color="#90a4ae")
    ax.tick_params(labelsize=6, colors="#78909c")
    ax.spines[["top","right"]].set_visible(False)
    ax.spines[["left","bottom"]].set_color("#1e3a5f")
    plt.tight_layout(pad=0.3)
    return fig


def render():
    """Main entry point — call this from inside the Live Digital Twin tab."""

    # ── Session state ────────────────────────────────────────────────
    if "twin_mode" not in st.session_state:
        st.session_state.twin_mode = "normal"
    if "twin_running" not in st.session_state:
        st.session_state.twin_running = False
    if "twin_history" not in st.session_state:
        st.session_state.twin_history = {
            "real_mean": [], "twin_mean": [],
            "rmse": [], "r2": [], "gap": [],
            "tick": 0
        }

    # ── Server status ────────────────────────────────────────────────
    ok, src_msg = _check_server()

    status_col = OK_COL if ok else BAD_COL
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:10px;'
        f'background:#0d1b2a;border:1px solid #1e3a5f;border-radius:10px;'
        f'padding:10px 16px;margin-bottom:14px">'
        f'<span style="width:9px;height:9px;border-radius:50%;'
        f'background:{status_col};display:inline-block"></span>'
        f'<span style="font-size:.8rem;color:#cfd8dc">Twin server: '
        f'<b style="color:{status_col}">{"Online" if ok else "Offline"}</b></span>'
        f'<span style="font-size:.72rem;color:#78909c;margin-left:8px">{src_msg}</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    if not ok:
        st.error(
            "Twin server not running. Start it with:\n\n"
            "```bash\ncd ~/edge-ndt-rl\n"
            "uvicorn dashboard.twin_server:app --port 8765\n```\n\n"
            "Or use the unified app.py which starts it automatically."
        )
        return

    # ── Controls ─────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns([2, 1, 1, 2])
    with c1:
        mode = st.radio(
            "Network condition",
            ["normal", "high_churn", "low_bandwidth"],
            format_func=lambda x: {
                "normal":        "Normal",
                "high_churn":    "High churn (nodes fail randomly)",
                "low_bandwidth": "Low bandwidth (2 Mbps links)",
            }[x],
            horizontal=True,
            key="twin_mode_radio"
        )
        if mode != st.session_state.twin_mode:
            st.session_state.twin_mode = mode
            # Reset history on mode change
            st.session_state.twin_history = {
                "real_mean":[], "twin_mean":[],
                "rmse":[], "r2":[], "gap":[], "tick":0
            }
            requests.get(f"{SERVER}/sim/{mode}/reset", timeout=2)

    with c2:
        run_btn = st.button(
            "▶ Start live feed" if not st.session_state.twin_running else "⏸ Pause",
            type="primary", use_container_width=True
        )
        if run_btn:
            st.session_state.twin_running = not st.session_state.twin_running

    with c3:
        reset_btn = st.button("↺ Reset", use_container_width=True)
        if reset_btn:
            requests.get(f"{SERVER}/sim/{mode}/reset", timeout=2)
            st.session_state.twin_history = {
                "real_mean":[], "twin_mean":[],
                "rmse":[], "r2":[], "gap":[], "tick":0
            }

    with c4:
        ticks_label = st.empty()

    st.markdown("---")

    # ── Fetch one tick ────────────────────────────────────────────────
    data = None
    if st.session_state.twin_running:
        data = _fetch_tick(st.session_state.twin_mode)

    if data:
        h = st.session_state.twin_history
        h["real_mean"].append(np.mean(data["real_loads"]))
        h["twin_mean"].append(np.mean(data["twin_loads"]))
        h["rmse"].append(data["rmse"])
        h["r2"].append(data["r2"])
        h["gap"].append(data["gap_pct"])
        h["tick"] = data.get("tick_count", h["tick"] + 1)
        # Keep last 80 points
        for k in ["real_mean","twin_mean","rmse","r2","gap"]:
            if len(h[k]) > 80:
                h[k].pop(0)
        st.session_state.twin_history = h

    h = st.session_state.twin_history

    # Use last fetched or session data
    real_loads  = data["real_loads"]   if data else [0.3]*N_NODES
    twin_loads  = data["twin_loads"]   if data else [0.3]*N_NODES
    real_online = data["real_online"]  if data else [True]*N_NODES
    twin_online = data["twin_online"]  if data else [True]*N_NODES
    real_q      = data["real_queues"]  if data else [0]*N_NODES
    twin_q      = data["twin_queues"]  if data else [0]*N_NODES
    real_recv   = data["real_tasks_recv"] if data else [0]*N_NODES
    twin_recv   = data["twin_tasks_recv"] if data else [0]*N_NODES
    rmse_val    = data["rmse"]    if data else 0
    r2_val      = data["r2"]      if data else 0
    risk        = data["risk"]    if data else "—"
    gap_val     = data["gap_pct"] if data else 0
    avg_real    = data["avg_real_ms"] if data else None
    avg_twin    = data["avg_twin_ms"] if data else None
    tasks_real  = data["tasks_real"]  if data else 0
    tasks_twin  = data["tasks_twin"]  if data else 0
    sla_pct     = data["sla_pct"]     if data else 0
    sim_time    = data["sim_time_ms"] if data else 0
    tick_count  = h["tick"]

    ticks_label.markdown(
        f'<div style="text-align:right;font-size:.72rem;'
        f'font-family:monospace;color:#78909c;padding-top:28px">'
        f'tick {tick_count} · sim time {sim_time/1000:.0f}s</div>',
        unsafe_allow_html=True
    )

    # ── Top KPI row ───────────────────────────────────────────────────
    k1,k2,k3,k4,k5,k6 = st.columns(6)
    risk_colors = {"LOW":"#57c46b","MEDIUM":"#ffa726","HIGH":"#e2574c"}
    risk_col = risk_colors.get(risk, "#cfd8dc")

    def _kpi(col, val, label, delta="", col_cls=""):
        with col:
            st.markdown(
                f'<div style="background:#0d1b2a;border:1px solid #1e3a5f;'
                f'border-radius:10px;padding:10px 12px;text-align:center">'
                f'<div style="font-size:1.4rem;font-weight:800;color:{col_cls or "#4fc3f7"}">'
                f'{val}</div>'
                f'<div style="font-size:.65rem;color:#78909c;margin-top:3px;'
                f'text-transform:uppercase;letter-spacing:.06em">{label}</div>'
                f'<div style="font-size:.7rem;color:#66bb6a;margin-top:2px">{delta}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

    _kpi(k1, f"{rmse_val:.4f}", "Load RMSE (rolling)")
    _kpi(k2, f"{r2_val:.3f}", "Load R² (rolling)")
    _kpi(k3, risk, "Fidelity risk", col_cls=risk_col)
    gap_col = "#e2574c" if gap_val > 5 else "#66bb6a" if gap_val < 0 else "#ffa726"
    _kpi(k4, f"{'+' if gap_val>=0 else ''}{gap_val:.1f}%",
         "Sim-to-real gap", col_cls=gap_col)
    _kpi(k5, f"{int(avg_real):,}ms" if avg_real else "—", "Avg ACT — real",
         f"twin: {int(avg_twin):,}ms" if avg_twin else "")
    _kpi(k6, f"{sla_pct:.1f}%", "SLA violations (real)",
         f"{tasks_real:,} tasks done")

    st.markdown("---")

    # ── Network canvas — real vs twin ─────────────────────────────────
    nc1, nc2 = st.columns(2)

    with nc1:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        fig.patch.set_facecolor("#0a0e1a")
        _draw_network_canvas(
            real_loads, real_online, real_q, real_recv,
            real_recv, REAL_COL, "Real edge network", ax
        )
        online_r = sum(1 for o in real_online if o)
        ax.text(0, -1.22,
                f"Online: {online_r}/8 · Tasks completed: {tasks_real:,}",
                ha="center", va="center", fontsize=7, color="#78909c")
        plt.tight_layout(pad=0.2)
        st.pyplot(fig)
        plt.close(fig)

    with nc2:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        fig.patch.set_facecolor("#0a0e1a")
        _draw_network_canvas(
            twin_loads, twin_online, twin_q, twin_recv,
            twin_recv, TWIN_COL, "Digital twin belief", ax
        )
        online_t = sum(1 for o in twin_online if o)
        ax.text(0, -1.22,
                f"Online: {online_t}/8 · Tasks completed: {tasks_twin:,}",
                ha="center", va="center", fontsize=7, color="#78909c")
        plt.tight_layout(pad=0.2)
        st.pyplot(fig)
        plt.close(fig)

    # Legend
    st.markdown(
        '<div style="display:flex;gap:18px;font-size:.72rem;color:#78909c;'
        'margin-top:4px;flex-wrap:wrap">'
        '<span>🟢 low load</span>'
        '<span>🔴 high load</span>'
        '<span>⬛ node offline</span>'
        '<span style="color:#e2574c">■ queue badge = waiting tasks</span>'
        '<span>tiny bar below each node = tasks received (all 8 active)</span>'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    # ── Per-node comparison table ─────────────────────────────────────
    st.markdown(
        '<div style="font-size:.78rem;font-weight:700;color:#4fc3f7;'
        'text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px">'
        'Per-node detail — all 8 nodes</div>',
        unsafe_allow_html=True
    )

    cols = st.columns(8)
    for i, col in enumerate(cols):
        rl = real_loads[i]
        tl = twin_loads[i]
        ro = real_online[i]
        gap_node = round(abs(rl - tl) * 100, 1)
        hue = int((1 - rl) * 120) if ro else 0
        bg_col = f"hsl({hue},65%,25%)" if ro else "#1e2a3f"
        border = f"hsl({hue},65%,48%)" if ro else "#3a4363"
        with col:
            st.markdown(
                f'<div style="background:{bg_col};border:1px solid {border};'
                f'border-radius:8px;padding:7px 8px;text-align:center;'
                f'margin-bottom:4px">'
                f'<div style="font-weight:700;font-size:.75rem;color:#cfd8dc">'
                f'{NODE_LABELS[i]}</div>'
                f'<div style="font-size:.7rem;color:{"#57c46b" if ro else "#78909c"}">'
                f'{"●" if ro else "○"}</div>'
                f'<div style="font-size:.68rem;color:#90a4ae;margin-top:2px">'
                f'R:{int(rl*100)}% T:{int(tl*100)}%</div>'
                f'<div style="font-size:.65rem;color:'
                f'{"#e2574c" if gap_node>15 else "#ffa726" if gap_node>5 else "#57c46b"}'
                f';margin-top:1px">Δ{gap_node}%</div>'
                f'<div style="font-size:.62rem;color:#546e7a;margin-top:2px">'
                f'Q:{real_q[i]} recv:{real_recv[i]}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

    st.markdown("---")

    # ── Time-series charts ────────────────────────────────────────────
    st.markdown(
        '<div style="font-size:.78rem;font-weight:700;color:#4fc3f7;'
        'text-transform:uppercase;letter-spacing:.06em;margin-bottom:10px">'
        'Live time-series — from backend (no synthetic data)</div>',
        unsafe_allow_html=True
    )

    sc1, sc2 = st.columns(2)
    with sc1:
        # Real vs twin load divergence
        fig = _draw_sparkline([], REAL_COL, "")
        if len(h["real_mean"]) >= 2:
            fig, ax = plt.subplots(figsize=(6, 1.8))
            ax.set_facecolor("#0d1b2a")
            fig.patch.set_facecolor("#0a0e1a")
            x = list(range(len(h["real_mean"])))
            ax.fill_between(x, h["real_mean"], alpha=.2, color=REAL_COL)
            ax.fill_between(x, h["twin_mean"], alpha=.2, color=TWIN_COL)
            ax.plot(x, h["real_mean"], color=REAL_COL, lw=1.8, label="Real")
            ax.plot(x, h["twin_mean"], color=TWIN_COL, lw=1.8,
                    linestyle="--", label="Twin belief")
            ax.set_title("Mean node load — real vs twin",
                         color="#4fc3f7", fontsize=8, fontweight="bold")
            ax.set_ylabel("Load", fontsize=7, color="#90a4ae")
            ax.legend(fontsize=7); ax.grid(True)
            ax.tick_params(labelsize=6, colors="#78909c")
            ax.spines[["top","right"]].set_color("#1e3a5f")
            ax.spines[["left","bottom"]].set_color("#1e3a5f")
            plt.tight_layout(pad=0.3)
        st.pyplot(fig); plt.close(fig)

        # R² rolling
        fig2, ax2 = plt.subplots(figsize=(6, 1.8))
        ax2.set_facecolor("#0d1b2a"); fig2.patch.set_facecolor("#0a0e1a")
        if len(h["r2"]) >= 2:
            x = list(range(len(h["r2"])))
            ax2.fill_between(x, h["r2"], alpha=.2, color=R2_COL)
            ax2.plot(x, h["r2"], color=R2_COL, lw=1.8)
            ax2.axhline(0.5, color="#ffa726", lw=0.8, linestyle="--", alpha=0.7)
        ax2.set_ylim(-0.1, 1.05)
        ax2.set_title("Rolling R² (twin fidelity)", color="#4fc3f7",
                      fontsize=8, fontweight="bold")
        ax2.set_ylabel("R²", fontsize=7, color="#90a4ae")
        ax2.grid(True); ax2.tick_params(labelsize=6, colors="#78909c")
        ax2.spines[["top","right"]].set_color("#1e3a5f")
        ax2.spines[["left","bottom"]].set_color("#1e3a5f")
        plt.tight_layout(pad=0.3)
        st.pyplot(fig2); plt.close(fig2)

    with sc2:
        # RMSE rolling
        fig3, ax3 = plt.subplots(figsize=(6, 1.8))
        ax3.set_facecolor("#0d1b2a"); fig3.patch.set_facecolor("#0a0e1a")
        if len(h["rmse"]) >= 2:
            x = list(range(len(h["rmse"])))
            ax3.fill_between(x, h["rmse"], alpha=.2, color=RMSE_COL)
            ax3.plot(x, h["rmse"], color=RMSE_COL, lw=1.8)
        ax3.set_title("Rolling RMSE (lower = better twin accuracy)",
                      color="#4fc3f7", fontsize=8, fontweight="bold")
        ax3.set_ylabel("RMSE", fontsize=7, color="#90a4ae")
        ax3.grid(True); ax3.tick_params(labelsize=6, colors="#78909c")
        ax3.spines[["top","right"]].set_color("#1e3a5f")
        ax3.spines[["left","bottom"]].set_color("#1e3a5f")
        plt.tight_layout(pad=0.3)
        st.pyplot(fig3); plt.close(fig3)

        # Gap sparkline
        fig4, ax4 = plt.subplots(figsize=(6, 1.8))
        ax4.set_facecolor("#0d1b2a"); fig4.patch.set_facecolor("#0a0e1a")
        if len(h["gap"]) >= 2:
            x = list(range(len(h["gap"])))
            ax4.axhline(0, color="#546e7a", lw=0.8, linestyle="--", alpha=0.6)
            pos = [v if v > 0 else 0 for v in h["gap"]]
            neg = [v if v < 0 else 0 for v in h["gap"]]
            ax4.fill_between(x, pos, alpha=.3, color=BAD_COL)
            ax4.fill_between(x, neg, alpha=.3, color=OK_COL)
            ax4.plot(x, h["gap"], color=GAP_COL, lw=1.8)
        ax4.set_title("Sim-to-real ACT gap % (orange=twin over-optimistic)",
                      color="#4fc3f7", fontsize=8, fontweight="bold")
        ax4.set_ylabel("Gap %", fontsize=7, color="#90a4ae")
        ax4.grid(True); ax4.tick_params(labelsize=6, colors="#78909c")
        ax4.spines[["top","right"]].set_color("#1e3a5f")
        ax4.spines[["left","bottom"]].set_color("#1e3a5f")
        plt.tight_layout(pad=0.3)
        st.pyplot(fig4); plt.close(fig4)

    # ── Recent tasks log ──────────────────────────────────────────────
    if data and data.get("recent_tasks"):
        st.markdown("---")
        st.markdown(
            '<div style="font-size:.78rem;font-weight:700;color:#4fc3f7;'
            'text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px">'
            'Recent task completions (from backend)</div>',
            unsafe_allow_html=True
        )
        rows = data["recent_tasks"][-8:]
        df = pd.DataFrame(rows)
        if not df.empty:
            df["status"] = df["violated"].apply(
                lambda v: "❌ SLA violated" if v else "✅ Met"
            )
            df["ct_ms"] = df["ct_ms"].apply(lambda v: f"{v:,} ms")
            df = df[["id","src","dst","ct_ms","status"]].rename(columns={
                "id":"Task ID","src":"From node","dst":"To node",
                "ct_ms":"Completion time","status":"SLA"
            })
            st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Auto-refresh ──────────────────────────────────────────────────
    if st.session_state.twin_running:
        time.sleep(1)
        st.rerun()
