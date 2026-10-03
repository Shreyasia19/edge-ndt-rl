"""
twin_server.py
FastAPI server — serves live NetSim data to digital_twin_dashboard.html

Run: uvicorn twin_server:app --reload --port 8765

This replaces the synthetic SAMPLE_DATA in the HTML with real
physics-engine output from your actual NetSim + MODE_PARAMS.

Endpoints:
  GET /sim/{mode}           → run one sim tick, return state JSON
  GET /sim/{mode}/full      → run full 120s sim, return series JSON
  GET /sim/{mode}/stream    → SSE stream of live ticks (for live canvas)
  GET /health               → ping
"""

import asyncio
import json
import math
import random
import time
from dataclasses import dataclass, field
from typing import AsyncGenerator

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

app = FastAPI(title="EdgeOrchestrate Twin Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # dashboard is a local file
    allow_methods=["GET"],
    allow_headers=["*"],
)

# ── Physics engine (from edge_ndt_digital_twin_sim.py) ─────────────

SEED = 7
N_NODES = 8
DT = 0.10
ROLLING_WINDOW = 40
WARMUP_FRACTION = 0.10
N_RUNS = 5
RECORD_EVERY = 5

MODE_PARAMS = {
    "normal":        dict(fail_real=0.00, fail_twin=0.00, bw_real=1.00, bw_twin=1.00,
                          track=0.78, load_noise=0.045, mean_load=0.32),
    "high_churn":    dict(fail_real=0.16, fail_twin=0.05, bw_real=1.00, bw_twin=1.00,
                          track=0.62, load_noise=0.08,  mean_load=0.34),
    "low_bandwidth": dict(fail_real=0.00, fail_twin=0.00, bw_real=0.32, bw_twin=0.48,
                          track=0.72, load_noise=0.05,  mean_load=0.33),
}


def _gauss(mean, std_frac, rng):
    return max(mean * 0.15, rng.gauss(mean, mean * std_frac))


@dataclass
class _Node:
    cap: float
    bw: float
    online: bool = True
    tasks: list = field(default_factory=list)
    fail_clock: float = 0.0


class NetSim:
    def __init__(self, base_cap, base_bw, fail_p, bw_factor, rng):
        self.rng = rng
        self.fail_p = fail_p
        self.bw_factor = bw_factor
        self.nodes = [_Node(cap=c, bw=b, fail_clock=rng.uniform(0, 2))
                      for c, b in zip(base_cap, base_bw)]
        self.spawn_acc = 0.0
        self.sim_time_ms = 0.0
        self.completions = []
        self.done_count = 0
        self.viol_count = 0

    def tick(self, dt=DT):
        self.sim_time_ms += dt * 1000
        for n in self.nodes:
            n.fail_clock -= dt
            if n.fail_clock <= 0:
                n.fail_clock = self.rng.uniform(1.2, 2.5)
                if self.fail_p > 0:
                    if n.online and self.rng.random() < self.fail_p:
                        n.online = False
                    elif not n.online and self.rng.random() < 0.6:
                        n.online = True
                else:
                    n.online = True
        self.spawn_acc += dt * 2.2
        while self.spawn_acc >= 1:
            self.spawn_acc -= 1
            self._spawn_task()
        for n in self.nodes:
            if not n.online or not n.tasks:
                continue
            rate = (n.cap / 40.0) / len(n.tasks)
            remaining = []
            for rem, created, deadline in n.tasks:
                rem -= dt * 1000 * rate
                if rem <= 0:
                    ct = self.sim_time_ms - created
                    viol = ct > deadline
                    self.completions.append((ct, viol))
                    if len(self.completions) > ROLLING_WINDOW:
                        self.completions.pop(0)
                    self.done_count += 1
                    self.viol_count += int(viol)
                else:
                    remaining.append((rem, created, deadline))
            n.tasks = remaining

    def _spawn_task(self):
        demand = _gauss(500, 0.6, self.rng)
        data = _gauss(500, 0.8, self.rng)
        online = [n for n in self.nodes if n.online]
        if not online:
            return
        target = min(online, key=lambda n: len(n.tasks))
        tx = data / (target.bw * self.bw_factor * 10 + 1e-9)
        comp = demand / (target.cap / 40.0 + 1e-9)
        total_ms = tx + comp
        target.tasks.append((total_ms, self.sim_time_ms, total_ms * 1.8))

    def node_loads(self):
        return [min(1.0, len(n.tasks) / 4.0) for n in self.nodes]

    def node_online(self):
        return [n.online for n in self.nodes]

    def node_queue_lengths(self):
        return [len(n.tasks) for n in self.nodes]

    def avg_completion(self):
        if not self.completions:
            return None
        return sum(c[0] for c in self.completions) / len(self.completions)

    def sla_violation_pct(self):
        if not self.done_count:
            return 0.0
        return self.viol_count / self.done_count * 100

    def reset(self, base_cap, base_bw, rng):
        self.__init__(base_cap, base_bw, self.fail_p, self.bw_factor, rng)


def _rmse_r2(real_arr, pred_arr):
    diff = real_arr - pred_arr
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    ss_res = float(np.sum(diff ** 2))
    ss_tot = float(np.sum((real_arr - real_arr.mean()) ** 2))
    r2 = 1 - (ss_res / (ss_tot + 1e-9))
    return rmse, r2


def _risk(r2):
    if r2 > 0.6:
        return "LOW"
    if r2 > 0.35:
        return "MEDIUM"
    return "HIGH"


# ── Full simulation (for /sim/{mode}/full and sim_export.json) ──────

def run_mode_full(mode_name: str, sim_seconds: int = 120, n_runs: int = N_RUNS):
    params = MODE_PARAMS[mode_name]
    n_steps = int(sim_seconds / DT)
    all_summaries = []
    first_series = None

    for run_idx in range(n_runs):
        rng = random.Random(SEED * 1000 + hash(mode_name) % 997 + run_idx)
        np.random.seed((SEED * 1000 + run_idx) % (2 ** 31 - 1))
        base_cap = [max(20, rng.gauss(40, 12)) for _ in range(N_NODES)]
        base_bw  = [max(2, rng.gauss(10, 8))  for _ in range(N_NODES)]

        real   = NetSim(base_cap, base_bw, params["fail_real"], params["bw_real"],
                        random.Random(rng.random()))
        belief = NetSim(base_cap, base_bw, params["fail_twin"], params["bw_twin"],
                        random.Random(rng.random()))

        t_s, real_s, twin_s, rmse_s, r2_s, gap_s = [], [], [], [], [], []
        real_full, pred_full = [], []
        load_state = [max(0.05, rng.gauss(0.30, 0.12)) for _ in range(N_NODES)]
        load_base  = list(load_state)
        prev_real  = None
        track, noise_scale, mean_load = (
            params["track"], params["load_noise"], params["mean_load"]
        )

        for step in range(n_steps):
            real.tick(DT)
            belief.tick(DT)
            for i in range(N_NODES):
                load_state[i] += (load_base[i] - load_state[i]) * 0.15 + rng.gauss(0, 0.035)
                load_state[i] = min(0.98, max(0.02, load_state[i]))
            cur_real = np.array(load_state)
            prev = prev_real if prev_real is not None else cur_real
            node_noise = np.array([rng.gauss(0, noise_scale) for _ in range(N_NODES)])
            pred = np.clip(track * prev + (1 - track) * mean_load + node_noise, 0.0, 1.0)
            prev_real = cur_real
            real_full.append(cur_real)
            pred_full.append(pred)

            if step % RECORD_EVERY == 0:
                t_s.append(round(step * DT, 2))
                real_s.append(round(float(cur_real.mean()), 4))
                twin_s.append(round(float(pred.mean()), 4))
                w = min(ROLLING_WINDOW, len(real_full))
                rm, r2 = _rmse_r2(np.array(real_full[-w:]), np.array(pred_full[-w:]))
                rmse_s.append(round(rm, 4))
                r2_s.append(round(r2, 4))
                ar = real.avg_completion()
                at = belief.avg_completion()
                gap_s.append(round(((ar - at) / at * 100) if (ar and at) else 0.0, 2))

        warmup = int(len(real_full) * WARMUP_FRACTION)
        rmse_all, r2_all = _rmse_r2(
            np.array(real_full[warmup:]), np.array(pred_full[warmup:])
        )
        ar = real.avg_completion()
        at = belief.avg_completion()
        gap_pct = ((ar - at) / at * 100) if (ar and at) else None

        all_summaries.append({
            "rmse_load": round(rmse_all, 4),
            "r2_load":   round(r2_all, 4),
            "sla_violation_pct_real": round(real.sla_violation_pct(), 2),
            "avg_completion_real_ms": round(ar, 1) if ar else None,
            "avg_completion_twin_ms": round(at, 1) if at else None,
            "gap_pct":         round(gap_pct, 2) if gap_pct is not None else None,
            "tasks_completed_real": real.done_count,
            "tasks_completed_twin": belief.done_count,
        })
        if run_idx == 0:
            first_series = {
                "t": t_s, "real_load_mean": real_s, "twin_load_mean": twin_s,
                "rmse_rolling": rmse_s, "r2_rolling": r2_s, "gap_pct_rolling": gap_s,
            }

    # aggregate
    fields = ["rmse_load", "r2_load", "sla_violation_pct_real",
              "avg_completion_real_ms", "avg_completion_twin_ms", "gap_pct"]
    agg = {"mode": mode_name, "n_runs": n_runs}
    for f in fields:
        vals = [r[f] for r in all_summaries if r[f] is not None]
        agg[f]          = round(float(np.mean(vals)), 4) if vals else None
        agg[f + "_std"] = round(float(np.std(vals)),  4) if vals else None
    agg["risk_level"]          = _risk(agg["r2_load"] or 0)
    agg["tasks_completed_real"] = int(np.mean([r["tasks_completed_real"] for r in all_summaries]))
    agg["tasks_completed_twin"] = int(np.mean([r["tasks_completed_twin"] for r in all_summaries]))

    return agg, first_series


# ── Live tick state (one per mode, shared across SSE clients) ───────

class LiveState:
    def __init__(self, mode: str):
        self.mode = mode
        params = MODE_PARAMS[mode]
        rng = random.Random(SEED)
        self.rng = rng
        base_cap = [max(20, rng.gauss(40, 12)) for _ in range(N_NODES)]
        base_bw  = [max(2,  rng.gauss(10,  8)) for _ in range(N_NODES)]
        self.real   = NetSim(base_cap, base_bw, params["fail_real"],
                             params["bw_real"], random.Random(rng.random()))
        self.belief = NetSim(base_cap, base_bw, params["fail_twin"],
                             params["bw_twin"], random.Random(rng.random()))
        self.load_state = [max(0.05, rng.gauss(0.30, 0.12)) for _ in range(N_NODES)]
        self.load_base  = list(self.load_state)
        self.prev_real  = None
        self.real_hist, self.pred_hist = [], []
        self.gap_hist   = []
        p = params
        self.track, self.noise, self.mean_load = p["track"], p["load_noise"], p["mean_load"]

    def tick(self, dt=0.5):
        """Advance one UI tick (~500ms real time = 0.5s sim time)."""
        for _ in range(5):                  # 5 × DT=0.1
            self.real.tick(DT)
            self.belief.tick(DT)
        for i in range(N_NODES):
            self.load_state[i] += (
                (self.load_base[i] - self.load_state[i]) * 0.15
                + self.rng.gauss(0, 0.035)
            )
            self.load_state[i] = min(0.98, max(0.02, self.load_state[i]))
        cur_real = np.array(self.load_state)
        prev = self.prev_real if self.prev_real is not None else cur_real
        node_noise = np.array([self.rng.gauss(0, self.noise) for _ in range(N_NODES)])
        pred = np.clip(
            self.track * prev + (1 - self.track) * self.mean_load + node_noise,
            0.0, 1.0
        )
        self.prev_real = cur_real
        self.real_hist.append(cur_real)
        self.pred_hist.append(pred)
        if len(self.real_hist) > ROLLING_WINDOW:
            self.real_hist.pop(0)
            self.pred_hist.pop(0)

        # Metrics
        rw = np.array(self.real_hist)
        tw = np.array(self.pred_hist)
        rmse, r2 = _rmse_r2(rw, tw) if len(rw) > 1 else (0.0, 0.0)
        ar = self.real.avg_completion()
        at = self.belief.avg_completion()
        gap = ((ar - at) / at * 100) if (ar and at) else 0.0
        self.gap_hist.append(round(gap, 2))
        if len(self.gap_hist) > 80:
            self.gap_hist.pop(0)

        return {
            "real_loads":    [round(v, 4) for v in self.real.node_loads()],
            "twin_loads":    [round(v, 4) for v in self.belief.node_loads()],
            "real_online":   self.real.node_online(),
            "twin_online":   self.belief.node_online(),
            "real_queues":   self.real.node_queue_lengths(),
            "twin_queues":   self.belief.node_queue_lengths(),
            "rmse":          round(rmse, 4),
            "r2":            round(r2, 4),
            "risk":          _risk(r2),
            "gap_pct":       round(gap, 2),
            "gap_hist":      list(self.gap_hist),
            "avg_real_ms":   round(ar, 1) if ar else None,
            "avg_twin_ms":   round(at, 1) if at else None,
            "tasks_real":    self.real.done_count,
            "tasks_twin":    self.belief.done_count,
            "sla_pct":       round(self.real.sla_violation_pct(), 2),
            "sim_time_ms":   round(self.real.sim_time_ms, 0),
        }


_live_states: dict[str, LiveState] = {}


def _get_live(mode: str) -> LiveState:
    if mode not in _live_states:
        _live_states[mode] = LiveState(mode)
    return _live_states[mode]


# ── Routes ───────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "modes": list(MODE_PARAMS.keys())}


@app.get("/sim/{mode}/tick")
def sim_tick(mode: str):
    if mode not in MODE_PARAMS:
        return JSONResponse({"error": "unknown mode"}, 400)
    state = _get_live(mode)
    return state.tick()


@app.get("/sim/{mode}/reset")
def sim_reset(mode: str):
    if mode in _live_states:
        del _live_states[mode]
    return {"reset": True, "mode": mode}


@app.get("/sim/{mode}/full")
def sim_full(mode: str, seconds: int = 120, runs: int = 3):
    if mode not in MODE_PARAMS:
        return JSONResponse({"error": "unknown mode"}, 400)
    agg, series = run_mode_full(mode, seconds, min(runs, 5))
    return {"summary": agg, "series": series}


@app.get("/sim/export")
def sim_export(seconds: int = 120, runs: int = 3):
    """
    Generates sim_export.json compatible with the HTML dashboard's
    Python-verified metrics tab. Load this via the file picker in the HTML.
    """
    import datetime
    modes_out = {}
    for m in MODE_PARAMS:
        agg, series = run_mode_full(m, seconds, min(runs, 5))
        modes_out[m] = {"summary": agg, "series": series}
    return {
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "sim_seconds_per_mode": seconds,
        "n_nodes": N_NODES,
        "n_runs": runs,
        "synthetic": False,
        "modes": modes_out,
    }


async def _sse_gen(mode: str) -> AsyncGenerator[str, None]:
    """Server-sent events — one JSON tick every 500ms."""
    state = _get_live(mode)
    while True:
        data = state.tick()
        yield f"data: {json.dumps(data)}\n\n"
        await asyncio.sleep(0.5)


@app.get("/sim/{mode}/stream")
async def sim_stream(mode: str):
    if mode not in MODE_PARAMS:
        return JSONResponse({"error": "unknown mode"}, 400)
    return StreamingResponse(
        _sse_gen(mode),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765, reload=False)
