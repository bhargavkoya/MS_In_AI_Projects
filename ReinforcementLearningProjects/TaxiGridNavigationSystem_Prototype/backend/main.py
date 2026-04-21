"""
main.py — FastAPI backend for the Taxi RL Simulation
Run with: uvicorn main:app --reload --port 8000
"""

import asyncio
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rl_logic import (
    TaxiGridEnv,
    QLearningAgent,
    MonteCarloAgent,
    ReachabilityError,
    bfs_with_path,
    validate_reachability,
    recommended_episodes,
    recommended_max_steps,
)

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(title="Taxi RL Simulation API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global in-memory state (single-user localhost demo) ───────────────────────

_state: dict = {
    "env":      None,   # TaxiGridEnv instance
    "ql_agent": None,   # QLearningAgent instance
    "mc_agent": None,   # MonteCarloAgent instance
    "trained":  None,   # "qlearning" | "montecarlo" | None
}

ACTION_NAMES = {0: "down", 1: "up", 2: "right", 3: "left", 4: "pickup", 5: "dropoff"}
MAX_EPISODES_CAP = 10_000


# ── Pydantic request / response models ────────────────────────────────────────

class InitializeRequest(BaseModel):
    rows: int
    cols: int
    obstacles: List[List[int]] = []
    taxi: Optional[List[int]] = None
    passenger: Optional[List[int]] = None
    destination: Optional[List[int]] = None


class InitializeResponse(BaseModel):
    ok: bool
    rows: int
    cols: int
    taxi: List[int]
    passenger: List[int]
    destination: List[int]
    recommended_episodes: int
    recommended_max_steps: int


class BfsLeg(BaseModel):
    start: List[int]
    goal: List[int]
    reachable: bool
    explored: List[List[int]]
    path: List[List[int]]


class BfsCheckResponse(BaseModel):
    leg1: BfsLeg
    leg2: BfsLeg
    both_reachable: bool


class TrainRequest(BaseModel):
    algorithm: str          # "qlearning" | "montecarlo"
    episodes: Optional[int] = None
    max_steps: Optional[int] = None


class TrainResponse(BaseModel):
    ok: bool
    algorithm: str
    train_time_seconds: float
    episodes: int
    final_success_rate: float
    episode_rewards: List[float]
    episode_successes: List[int]
    episode_lengths: List[int]


class StepRequest(BaseModel):
    algorithm: str
    state: List[int]        # 7-element list


class StepResponse(BaseModel):
    action: int
    action_name: str
    next_state: List[int]
    reward: float
    terminated: bool


class SimulateRequest(BaseModel):
    algorithm: str
    max_steps: Optional[int] = None


class SimulateResponse(BaseModel):
    success: bool
    algorithm: str
    total_steps: int
    path: List[List[int]]


class ResetResponse(BaseModel):
    ok: bool


# ── Helper ────────────────────────────────────────────────────────────────────

def _require_env():
    if _state["env"] is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /initialize first.")
    return _state["env"]


def _get_agent(algorithm: str):
    if algorithm == "qlearning":
        return _state["ql_agent"]
    elif algorithm == "montecarlo":
        return _state["mc_agent"]
    raise HTTPException(status_code=400, detail=f"Unknown algorithm '{algorithm}'. Use 'qlearning' or 'montecarlo'.")


def _require_trained_agent(algorithm: str):
    if _state["trained"] != algorithm:
        raise HTTPException(
            status_code=400,
            detail=f"Algorithm '{algorithm}' has not been trained yet. Call /train first."
        )
    agent = _get_agent(algorithm)
    if agent is None:
        raise HTTPException(status_code=400, detail="Environment not initialized.")
    return agent


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/initialize", response_model=InitializeResponse)
def initialize(req: InitializeRequest):
    """
    Create the environment, set positions, validate reachability.
    Clears any previously trained agents.
    """
    try:
        env = TaxiGridEnv(req.rows, req.cols, req.obstacles)
        env.set_positions(
            taxi        = req.taxi,
            passenger   = req.passenger,
            destination = req.destination,
        )
        validate_reachability(env)
    except (ValueError, ReachabilityError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    _state["env"]      = env
    _state["ql_agent"] = QLearningAgent(env)
    _state["mc_agent"] = MonteCarloAgent(env)
    _state["trained"]  = None

    rec_eps   = recommended_episodes(env.rows, env.cols)
    rec_steps = recommended_max_steps(env.rows, env.cols)

    return InitializeResponse(
        ok=True,
        rows=env.rows,
        cols=env.cols,
        taxi=list(env.taxi),
        passenger=list(env.passenger),
        destination=list(env.destination),
        recommended_episodes=rec_eps,
        recommended_max_steps=rec_steps,
    )


@app.post("/bfs-check", response_model=BfsCheckResponse)
def bfs_check():
    """
    Run BFS for taxi→passenger (leg 1) and passenger→destination (leg 2).
    Returns explored node lists (for animation) and optimal paths.
    Uses initial positions, not current mid-simulation state.
    """
    env = _require_env()
    env.reset()  # restore initial positions

    path1, explored1, ok1 = bfs_with_path(env, env.taxi, env.passenger)
    path2, explored2, ok2 = bfs_with_path(env, env.passenger, env.destination)

    return BfsCheckResponse(
        leg1=BfsLeg(
            start=list(env.taxi),
            goal=list(env.passenger),
            reachable=ok1,
            explored=[list(p) for p in explored1],
            path=[list(p) for p in path1],
        ),
        leg2=BfsLeg(
            start=list(env.passenger),
            goal=list(env.destination),
            reachable=ok2,
            explored=[list(p) for p in explored2],
            path=[list(p) for p in path2],
        ),
        both_reachable=ok1 and ok2,
    )


@app.post("/train", response_model=TrainResponse)
async def train(req: TrainRequest):
    """
    Train the specified RL agent. Runs in a thread to avoid blocking the event loop.
    Episodes are capped at 10,000.
    """
    env = _require_env()
    agent = _get_agent(req.algorithm)
    if agent is None:
        raise HTTPException(status_code=400, detail="Initialize environment first.")

    env.reset()

    eps   = min(req.episodes or recommended_episodes(env.rows, env.cols), MAX_EPISODES_CAP)
    steps = req.max_steps or recommended_max_steps(env.rows, env.cols)

    loop = asyncio.get_event_loop()
    try:
        metrics = await loop.run_in_executor(None, agent.train, eps, steps)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training error: {e}")

    _state["trained"] = req.algorithm

    successes = metrics["episode_successes"]
    tail      = max(1, len(successes) // 10)
    final_sr  = float(sum(successes[-tail:]) / tail)

    return TrainResponse(
        ok=True,
        algorithm=req.algorithm,
        train_time_seconds=round(agent.train_time, 3),
        episodes=eps,
        final_success_rate=round(final_sr, 4),
        episode_rewards=[float(r) for r in metrics["episode_rewards"]],
        episode_successes=[int(s) for s in metrics["episode_successes"]],
        episode_lengths=[int(l) for l in metrics["episode_lengths"]],
    )


@app.post("/step", response_model=StepResponse)
def step(req: StepRequest):
    """
    Inject a state into the environment, apply the greedy action from the
    trained agent's Q-table, and return the resulting next state + reward.
    """
    env   = _require_env()
    agent = _require_trained_agent(req.algorithm)

    if len(req.state) != 7:
        raise HTTPException(status_code=400, detail="State must have exactly 7 integers.")

    env.inject_state(req.state)
    action = agent.get_greedy_action(req.state)

    next_obs, reward, terminated, _, _ = env.step(action)

    return StepResponse(
        action=action,
        action_name=ACTION_NAMES[action],
        next_state=[int(x) for x in next_obs],
        reward=float(reward),
        terminated=terminated,
    )


@app.post("/simulate", response_model=SimulateResponse)
def simulate(req: SimulateRequest):
    """
    Run a full greedy episode from initial positions and return the complete path.
    Used by the frontend for Auto-Run mode.
    """
    env   = _require_env()
    agent = _require_trained_agent(req.algorithm)

    max_steps = req.max_steps or recommended_max_steps(env.rows, env.cols)

    env.reset()
    success, path = agent.run(max_steps)

    return SimulateResponse(
        success=success,
        algorithm=req.algorithm,
        total_steps=len(path) - 1,
        path=[[int(x) for x in s] for s in path],
    )


@app.post("/reset", response_model=ResetResponse)
def reset():
    """Clear all environment and agent state."""
    _state["env"]      = None
    _state["ql_agent"] = None
    _state["mc_agent"] = None
    _state["trained"]  = None
    return ResetResponse(ok=True)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "initialized": _state["env"] is not None,
        "trained": _state["trained"],
    }
