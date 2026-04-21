/**
 * api.ts — Typed API service layer for the Taxi RL Simulation backend.
 * All coordinates use [row, col] arrays matching the notebook convention.
 */

const BASE_URL = "http://localhost:8000";

// ── Shared types ──────────────────────────────────────────────────────────────

/** 7-element RL state: [taxi_r, taxi_c, pass_r, pass_c, dest_r, dest_c, in_taxi] */
export type RLState = [number, number, number, number, number, number, number];

export type Algorithm = "qlearning" | "montecarlo";

// ── Request / Response types ──────────────────────────────────────────────────

export interface InitializeRequest {
  rows: number;
  cols: number;
  obstacles: [number, number][];
  taxi: [number, number] | null;
  passenger: [number, number] | null;
  destination: [number, number] | null;
}

export interface InitializeResponse {
  ok: boolean;
  rows: number;
  cols: number;
  taxi: [number, number];
  passenger: [number, number];
  destination: [number, number];
  recommended_episodes: number;
  recommended_max_steps: number;
}

export interface BfsLeg {
  start: [number, number];
  goal: [number, number];
  reachable: boolean;
  explored: [number, number][];
  path: [number, number][];
}

export interface BfsCheckResponse {
  leg1: BfsLeg;
  leg2: BfsLeg;
  both_reachable: boolean;
}

export interface TrainRequest {
  algorithm: Algorithm;
  episodes?: number;
  max_steps?: number;
}

export interface TrainResponse {
  ok: boolean;
  algorithm: string;
  train_time_seconds: number;
  episodes: number;
  final_success_rate: number;
  episode_rewards: number[];
  episode_successes: number[];
  episode_lengths: number[];
}

export interface StepRequest {
  algorithm: Algorithm;
  state: RLState;
}

export interface StepResponse {
  action: number;
  action_name: string;
  next_state: RLState;
  reward: number;
  terminated: boolean;
}

export interface SimulateRequest {
  algorithm: Algorithm;
  max_steps?: number;
}

export interface SimulateResponse {
  success: boolean;
  algorithm: string;
  total_steps: number;
  path: RLState[];
}

export interface ResetResponse {
  ok: boolean;
}

// ── Core fetch helper ─────────────────────────────────────────────────────────

async function post<TResponse>(endpoint: string, body: object): Promise<TResponse> {
  const res = await fetch(`${BASE_URL}${endpoint}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  const data = await res.json().catch(() => ({ detail: res.statusText }));

  if (!res.ok) {
    // FastAPI returns { detail: "..." } for HTTPException
    const message =
      typeof data.detail === "string"
        ? data.detail
        : JSON.stringify(data.detail ?? data);
    throw new Error(message);
  }

  return data as TResponse;
}

// ── API surface ───────────────────────────────────────────────────────────────

export const api = {
  /**
   * Initialize the environment with grid config + positions.
   * Must be called before BFS, train, step, or simulate.
   */
  initialize: (req: InitializeRequest) =>
    post<InitializeResponse>("/initialize", req),

  /**
   * Run BFS for both journey legs.
   * Returns explored node lists (for animation) and shortest paths.
   */
  bfsCheck: () => post<BfsCheckResponse>("/bfs-check", {}),

  /**
   * Train the selected RL algorithm.
   * Runs synchronously on the backend (may take several seconds).
   */
  train: (req: TrainRequest) => post<TrainResponse>("/train", req),

  /**
   * Apply the greedy policy for one step given the current state.
   * Used for step-by-step simulation mode.
   */
  step: (req: StepRequest) => post<StepResponse>("/step", req),

  /**
   * Run a full greedy episode and return the complete path at once.
   * Used for auto-run animation mode.
   */
  simulate: (req: SimulateRequest) => post<SimulateResponse>("/simulate", req),

  /**
   * Clear all backend state (env + agents).
   */
  reset: () => post<ResetResponse>("/reset", {}),
};
