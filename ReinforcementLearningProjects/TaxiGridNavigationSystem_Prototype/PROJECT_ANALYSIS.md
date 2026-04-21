# Taxi RL Simulation — Project Analysis

> **Date:** April 2026
> **Stack:** Python 3.14 · FastAPI · Gymnasium · React 18 · TypeScript · Vite · Tailwind CSS v4

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Backend — Core RL Logic (`rl_logic.py`)](#3-backend--core-rl-logic-rl_logicpy)
   - 3.1 Environment (`TaxiGridEnv`)
   - 3.2 BFS Utilities
   - 3.3 Q-Learning Agent
   - 3.4 Monte Carlo Agent
4. [Backend — API Server (`main.py`)](#4-backend--api-server-mainpy)
   - 4.1 Global State Model
   - 4.2 Endpoint Reference
   - 4.3 Request/Response Schemas
5. [Frontend Architecture](#5-frontend-architecture)
   - 5.1 Component Tree
   - 5.2 `App.tsx` — State Machine
   - 5.3 `TaxiGrid.tsx` — Visualization
   - 5.4 `ControlPanel.tsx` — Controls
   - 5.5 `InfoPanel.tsx` — Metrics
   - 5.6 `api.ts` — Service Layer
6. [Data Flows & Workflows](#6-data-flows--workflows)
   - 6.1 Initialization Flow
   - 6.2 BFS Check Flow
   - 6.3 Training Flow
   - 6.4 Step-by-Step Simulation Flow
   - 6.5 Auto-Run Simulation Flow
   - 6.6 Reset Flow
7. [State Machine (Frontend)](#7-state-machine-frontend)
8. [Algorithm Deep-Dives](#8-algorithm-deep-dives)
   - 8.1 BFS
   - 8.2 Q-Learning
   - 8.3 Monte Carlo
   - 8.4 Algorithm Comparison
9. [Key Design Decisions](#9-key-design-decisions)
10. [Known Bugs Fixed](#10-known-bugs-fixed)
11. [Coordinate Convention](#11-coordinate-convention)
12. [Configuration Reference](#12-configuration-reference)

---

## 1. Project Overview

The **Taxi RL Simulation** is a full-stack interactive teaching tool that lets a user:

1. **Design** a custom 2-D grid environment with a taxi, a passenger pick-up cell, a destination cell, and arbitrary obstacle walls.
2. **Verify** path reachability via an animated Breadth-First Search (BFS) visualisation.
3. **Train** either a Q-Learning or a First-Visit Monte Carlo RL agent directly in the browser, with live metrics returned from the backend.
4. **Simulate** the trained agent's greedy policy either one step at a time or in a smooth auto-run animation.

The system directly connects a Figma-exported React/TypeScript frontend to a Python FastAPI backend that wraps notebook-style RL logic derived from a Jupyter notebook (`CA02`).

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Browser (localhost:5173)             │
│                                                         │
│  ┌──────────────┐  ┌─────────────────┐  ┌───────────┐  │
│  │  ControlPanel │  │    TaxiGrid     │  │ InfoPanel │  │
│  │  (sidebar)    │  │  (canvas/cells) │  │ (metrics) │  │
│  └──────┬───────┘  └────────┬────────┘  └─────┬─────┘  │
│         │                   │                  │        │
│         └──────────┬────────┘──────────────────┘        │
│                    │  App.tsx (state machine)            │
│                    │  api.ts  (fetch wrappers)           │
└────────────────────┼────────────────────────────────────┘
                     │ HTTP / JSON  (localhost:8000)
┌────────────────────┼────────────────────────────────────┐
│                FastAPI (uvicorn)                        │
│                    │                                     │
│         ┌──────────┴─────────────┐                      │
│         │       main.py          │                      │
│         │  6 REST endpoints      │                      │
│         │  Global _state dict    │                      │
│         └──────────┬─────────────┘                      │
│                    │ imports                             │
│         ┌──────────┴─────────────┐                      │
│         │      rl_logic.py       │                      │
│         │  TaxiGridEnv           │                      │
│         │  QLearningAgent        │                      │
│         │  MonteCarloAgent       │                      │
│         │  bfs_with_path()       │                      │
│         └────────────────────────┘                      │
└────────────────────────────────────────────────────────┘
```

**Technology choices:**

| Layer | Technology | Rationale |
|---|---|---|
| UI framework | React 18 + TypeScript | Component model, strict typing |
| Build tool | Vite 6 | Fast HMR, ESM-native |
| Styling | Tailwind CSS v4 | Utility classes; inline styles for dynamic colours |
| Charts | Recharts | Lightweight, composable LineChart |
| Icons | Lucide React | Consistent SVG icon set |
| Backend | FastAPI | Auto-docs, async support, Pydantic validation |
| ASGI server | uvicorn | Standard FastAPI server |
| RL environment | gymnasium | Standardised Env API |
| Numerics | NumPy | Q-table arrays, vectorised argmax |

---

## 3. Backend — Core RL Logic (`rl_logic.py`)

### 3.1 Environment — `TaxiGridEnv`

Inherits from `gymnasium.Env`. Represents a discrete 2-D grid where a taxi must pick up a passenger and deliver them to a destination.

#### State Space

The observation is a **7-element integer array**:

```
[taxi_row, taxi_col, pass_row, pass_col, dest_row, dest_col, in_taxi]
```

| Index | Name | Range |
|---|---|---|
| 0 | `taxi_row` | `[0, rows-1]` |
| 1 | `taxi_col` | `[0, cols-1]` |
| 2 | `pass_row` | `[0, rows-1]` |
| 3 | `pass_col` | `[0, cols-1]` |
| 4 | `dest_row` | `[0, rows-1]` |
| 5 | `dest_col` | `[0, cols-1]` |
| 6 | `in_taxi` | `{0, 1}` |

Defined as `MultiDiscrete([rows, cols, rows, cols, rows, cols, 2])`.

#### Action Space

`Discrete(6)` — 6 actions:

| ID | Name | Effect |
|---|---|---|
| 0 | down | `row += 1` |
| 1 | up | `row -= 1` |
| 2 | right | `col += 1` |
| 3 | left | `col -= 1` |
| 4 | pickup | Pick up passenger if taxi is at passenger cell and `in_taxi == 0` |
| 5 | dropoff | Drop off if `in_taxi == 1` and taxi is at destination |

Movement into an obstacle or out-of-bounds is silently ignored (taxi stays in place). The passenger moves with the taxi whenever `in_taxi == 1`.

#### Reward Structure

| Event | Reward |
|---|---|
| Any movement (valid or blocked) | −1 |
| Successful pickup | +10 |
| Invalid pickup or invalid dropoff | −10 |
| Successful dropoff (terminal) | +20 |

#### Key Methods

| Method | Purpose |
|---|---|
| `set_positions(taxi, passenger, destination)` | Set/randomise initial positions; stores them as `_init_*` for `reset()` |
| `reset()` | Restore `_init_*` positions, clear `in_taxi` flag |
| `step(action)` | Apply action, compute reward, return `(obs, reward, terminated, False, {})` |
| `inject_state(state)` | **Extended method** — forcibly set all 7 state fields; used by `/step` endpoint to enable stateless step-by-step simulation |
| `is_valid(pos)` | Returns True if `pos` is in bounds and not an obstacle |
| `_validate_all_positions()` | Raises `ValueError` if any position is invalid or if two entities share a cell |

#### Constraint Validation

`set_positions` enforces:
- At least 3 free cells exist.
- Every position is in bounds and not on an obstacle.
- No two entities share the same cell.

`validate_reachability` (BFS-based) enforces:
- Taxi can reach passenger.
- Passenger can reach destination.

These checks run at `/initialize` time and throw 400 errors to the frontend.

---

### 3.2 BFS Utilities

#### `bfs_with_path(env, start, goal)`

Standard BFS returning three values:

```python
path, explored, reachable = bfs_with_path(env, start, goal)
```

| Return | Type | Meaning |
|---|---|---|
| `path` | `list[tuple]` | Shortest path from `start` to `goal` (inclusive). Empty if unreachable. |
| `explored` | `list[tuple]` | All cells dequeued, **in BFS order** — drives the frontend animation. |
| `reachable` | `bool` | Whether a path exists. |

The algorithm uses a `deque` for O(1) popleft, a `parent` dict for O(1) visited checks and path reconstruction, and appends the goal cell to `explored` before backtracking so the animation reveals it at the correct moment.

#### `recommended_episodes(rows, cols)`

```python
max(5000, rows * cols * 100)
```

Returns the episode count recommended to the frontend slider. Scales with grid area so larger grids get more training.

#### `recommended_max_steps(rows, cols)`

```python
max(200, rows * cols * 4)
```

Per-episode step cap during training. Scales with grid area.

---

### 3.3 Q-Learning Agent — `QLearningAgent`

#### Algorithm

**Temporal-difference (TD) control** — updates Q-values after every single step using the Bellman equation:

```
Q(s, a)  ←  Q(s, a) + α · [r + γ · max_a' Q(s', a')  −  Q(s, a)]
```

#### Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `alpha` (α) | 0.1 | Learning rate |
| `gamma` (γ) | 0.95 | Discount factor |
| `epsilon_start` | 1.0 | Initial exploration probability |
| `epsilon_end` | 0.01 | Minimum exploration probability |
| `epsilon_decay` | 0.995 | Multiplicative decay per episode |

#### Q-Table

Stored as a Python `dict` mapping `tuple(state) → np.zeros(6)`. Entries are created lazily on first access via `_get_q()`. This avoids pre-allocating all states (which would require rows² × cols² × 2 × 6 entries for a full table).

#### Epsilon Decay

```python
self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
```

With default `epsilon_decay=0.995`, epsilon reaches its floor (0.01) at approximately episode 920, giving adequate exploration for moderate-sized grids.

#### Metrics Collected

Per episode: `total_reward`, `success` (terminated?), `step_count`. Returned as parallel lists in `metrics` dict and forwarded to the frontend for chart rendering.

---

### 3.4 Monte Carlo Agent — `MonteCarloAgent`

#### Algorithm

**First-Visit Monte Carlo Control** — collects a complete episode trajectory, then propagates discounted returns backward from the terminal step:

```
G  ←  r_t  +  γ · G_{t+1}

For each first-visit (s, a) in the episode:
    N(s, a)   +=  1
    Q(s, a)   +=  (G  −  Q(s, a))  /  N(s, a)     # incremental mean
```

This is equivalent to averaging all observed returns `G` from each `(state, action)` pair using Welford's online algorithm, which avoids storing all returns in memory.

#### Critical Design Fix — Epsilon Decay

**Problem (pre-fix):** The default `epsilon_decay=0.995` caused epsilon to reach its floor at ~episode 1000 out of 5000. Unlike Q-Learning (which bootstraps from incomplete trajectories), Monte Carlo requires **complete successful episodes** to produce useful returns. A partially-trained policy at episode 1000 with epsilon ≈ 0.01 would deterministically choose suboptimal actions (defaulting to action 0 = "down"), creating infinite loops. Success rate dropped from ~87% at ep 500 to ~5% from ep 1000 onwards.

**Fix applied in `train()`:**

```python
# Compute decay so epsilon reaches epsilon_end at 80% of training
auto_decay = (self.epsilon_end / self.epsilon_start) ** (
    1.0 / max(1, int(episodes * 0.8))
)
effective_decay = max(self.epsilon_decay, auto_decay)  # higher = slower

# Larger per-episode step budget so exploration can complete trajectories
episode_max_steps = max(max_steps * 3, rows * cols * 15)
```

For 5000 episodes, `auto_decay ≈ 0.99885` vs the prior `0.995`, keeping epsilon at ~0.56 at episode 500 and reaching the floor only at episode 4000. Post-fix diagnostic results:

```
Episode   500 / 5000 | epsilon=0.5623 | success rate=97.6%
Episode  1000 / 5000 | epsilon=0.3162 | success rate=100.0%
...
Episode  5000 / 5000 | epsilon=0.0100 | success rate=100.0%
Greedy run: success=True, steps=12
```

#### Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `gamma` (γ) | 0.95 | Discount factor |
| `epsilon_start` | 1.0 | Initial exploration probability |
| `epsilon_end` | 0.01 | Minimum exploration probability |
| `epsilon_decay` | 0.995 | Starting decay (overridden dynamically in `train()`) |

---

## 4. Backend — API Server (`main.py`)

### 4.1 Global State Model

```python
_state = {
    "env":      None,   # TaxiGridEnv instance
    "ql_agent": None,   # QLearningAgent instance
    "mc_agent": None,   # MonteCarloAgent instance
    "trained":  None,   # "qlearning" | "montecarlo" | None
}
```

This is a **single-user, in-memory** design suitable for local demo use. No database or session management is needed. The `_state` dict is module-level and shared across all requests (single uvicorn worker).

CORS is configured for `http://localhost:5173` and `http://127.0.0.1:5173` (Vite dev server), allowing all HTTP methods and headers.

### 4.2 Endpoint Reference

| Method | Path | Purpose | Requires |
|---|---|---|---|
| `GET` | `/health` | Status check — returns `initialized` and `trained` flags | Nothing |
| `POST` | `/initialize` | Create env, validate positions + reachability | Nothing |
| `POST` | `/bfs-check` | Run BFS for both legs, return paths + explored nodes | Initialized env |
| `POST` | `/train` | Train selected algorithm (runs in thread executor) | Initialized env |
| `POST` | `/step` | Apply one greedy action from injected state | Trained agent |
| `POST` | `/simulate` | Run full greedy episode, return complete path | Trained agent |
| `POST` | `/reset` | Clear all state | Nothing |

#### `/initialize`

Creates a fresh `TaxiGridEnv`, places entities (or randomises positions for any `null`-supplied coordinate), validates in-bounds and non-overlapping constraints, then runs BFS reachability check. On success, creates new `QLearningAgent` and `MonteCarloAgent` wrapping the environment. Clears `trained` flag.

Returns the resolved positions (important when positions were auto-placed) and recommended episode/step counts.

#### `/bfs-check`

Calls `env.reset()` to restore initial positions, then runs `bfs_with_path` twice:
- **Leg 1:** Taxi → Passenger
- **Leg 2:** Passenger → Destination

Returns explored cell lists ordered by BFS dequeue order (for sequential animation), plus the optimal path for each leg. Both legs are always computed even if leg 1 is unreachable.

#### `/train` (async)

Training can take seconds to minutes. Uses `asyncio.get_event_loop().run_in_executor(None, agent.train, eps, steps)` to offload CPU-bound work to a thread pool without blocking the asyncio event loop. The episode count is capped at `MAX_EPISODES_CAP = 10,000`.

After training, computes `final_success_rate` as the mean of the last 10% of episodes.

Returns full per-episode metrics arrays (`episode_rewards`, `episode_successes`, `episode_lengths`) for chart rendering.

#### `/step`

Stateless endpoint — the frontend is responsible for tracking the current state. Accepts the current 7-element state, injects it into the environment via `env.inject_state(state)`, queries `agent.get_greedy_action(state)`, executes `env.step(action)`, and returns the resulting next state, reward, and termination flag.

This stateless design allows step-by-step simulation to be paused, inspected, and potentially rewound at the frontend level.

#### `/simulate`

Calls `agent.run(max_steps)` which internally calls `env.reset()` and then applies greedy actions until terminal or step limit. Returns the entire trajectory as a list of 7-element state arrays. Used by frontend auto-run mode.

### 4.3 Request/Response Schemas

All schemas use Pydantic `BaseModel` for automatic validation and OpenAPI documentation generation.

**Coordinate convention:** All positions are `List[int]` with format `[row, col]` — matching NumPy/notebook row-major convention and the grid's display indexing.

**Error format:** FastAPI's `HTTPException` with `status_code=400` produces `{"detail": "..."}` JSON responses. The frontend's `post<T>()` helper reads `data.detail` and rethrows as a JavaScript `Error`, which surfaces to the user as a warning overlay on the grid.

---

## 5. Frontend Architecture

### 5.1 Component Tree

```
App.tsx  (all shared state, all handlers)
├── TaxiGrid.tsx           (pure display — receives everything via props)
├── InfoPanel.tsx          (pure display — BFS results, train metrics, step info)
└── ControlPanel.tsx       (calls parent handlers via callbacks)
```

All application state lives in `App.tsx`. Child components are essentially **controlled components** — they receive data as props and emit events upward via callback props. There is no Context, no Redux, no Zustand. This keeps the state surface minimal and co-located.

### 5.2 `App.tsx` — State Machine

`App.tsx` holds the following state slices:

#### Setup State
```typescript
rows, cols                    // Grid dimensions (2–20)
placementMode                 // "taxi" | "start" | "dest" | "obstacle"
taxiPos, startPos, destPos    // Nullable positions (null = not placed)
obstacles                     // Position[]
```

#### Backend / Phase State
```typescript
isInitialized   // true after /initialize succeeds
isLoading       // spinner for initialize / bfs / simulate calls
isTraining      // spinner specifically for /train (long-running)
warningMsg      // null | string — shows overlay on grid
```

#### Resolved Positions (post-initialize)
```typescript
resolvedTaxi, resolvedPass, resolvedDest  // confirmed by backend
recEpisodes                               // recommended count from backend
```

#### BFS State
```typescript
bfsResult       // full BfsCheckResponse or null
bfsAnimIndex    // integer — how many explored cells to show
```

#### Algorithm & Training
```typescript
algorithm       // "qlearning" | "montecarlo"
episodeCount    // user-chosen episode count
isTrained       // true after /train succeeds
trainMetrics    // full TrainResponse or null
```

#### Simulation
```typescript
simPath         // RLState[] | null — full path from /simulate or built up via /step
simIndex        // current position in simPath
stepInfo        // current step details for InfoPanel
isAutoRunning   // true during setInterval auto-run
simDone         // true when terminal reached
cumReward       // running cumulative reward
```

#### Key Derived Values
```typescript
currentRLState = simPath?.[simIndex]

rlCurrentPos   = { row: state[0], col: state[1] }

rlPathPositions = simPath.slice(0, simIndex + 1).map(s => ({ row: s[0], col: s[1] }))

rlPhase = state[6] === 1 ? "dropoff" : "pickup"

// Passenger follows taxi when in_taxi=1
displayPassPos = state[6] === 1
  ? { row: state[0], col: state[1] }
  : { row: state[2], col: state[3] }

// BFS animation slicing
bfsExplored1 = bfsResult.leg1.explored.slice(0, Math.min(bfsAnimIndex, leg1ExpLen))
bfsExplored2 = bfsResult.leg2.explored.slice(0, Math.max(0, bfsAnimIndex - leg1ExpLen))
bfsPath1     = (bfsAnimIndex >= leg1ExpLen) ? bfsResult.leg1.path : []
bfsPath2     = (bfsAnimIndex >= totalExplored) ? bfsResult.leg2.path : []
```

#### Timers (via `useRef`)

Three refs hold interval/timeout IDs for cleanup:
- `autoRunRef` — `setInterval` at 450ms per step for auto-run
- `bfsTimerRef` — `setInterval` at 12ms per tick, revealing 2 BFS cells per tick
- `warnTimerRef` — `setTimeout` at 4000ms to auto-clear warning overlay

All three are cleaned up in a `useEffect` cleanup function on component unmount.

---

### 5.3 `TaxiGrid.tsx` — Visualization

A pure presentational component. Renders an `N × M` CSS grid with each cell as a `div`.

#### Props

| Prop | Type | Purpose |
|---|---|---|
| `rows`, `cols` | `number` | Grid dimensions |
| `taxiPos` | `Position \| null` | Taxi icon position |
| `startPos` | `Position \| null` | Passenger icon position |
| `destPos` | `Position \| null` | Destination icon position |
| `obstacles` | `Position[]` | Cells to render as dark walls |
| `onCellClick` | `(r, c) => void` | Cell click handler (setup mode) |
| `isInteractive` | `boolean` | Whether clicks are accepted |
| `bfsExplored1/2` | `[r,c][]` | BFS explored cells (animated slice) |
| `bfsPath1/2` | `[r,c][]` | BFS optimal path cells |
| `rlCurrentPos` | `Position \| null` | Current agent cell (green) |
| `rlPath` | `Position[]` | Trail of visited cells |
| `rlPhase` | `"pickup" \| "dropoff" \| null` | Determines path trail colour |
| `inTaxi` | `boolean` | Whether passenger is in taxi |
| `warningMessage` | `string \| null` | Text shown in bottom overlay |

#### Cell Colour Priority (highest to lowest)

1. **Obstacle** → dark gray (`#1f2937`)
2. **Current position** (`rlCurrentPos`) → green (`#4ade80`)
3. **RL path trail** → blue (pickup phase, `#bfdbfe`) or purple (dropoff phase, `#e9d5ff`)
4. **BFS optimal path** → orange (`#fdba74`)
5. **BFS explored** → yellow (`#fef9c3`)
6. **Default** → off-white (`#f9fafb`)

All colours are applied as **inline `style` props** (not Tailwind classes). This was required because Tailwind v4's JIT scanner purges classes assigned to variables at build time.

#### Entity Icons (Lucide React)

| Entity | Icon | Condition |
|---|---|---|
| Taxi | `Car` | Always at `taxiPos` |
| Passenger | `MapPin` | At `startPos` when `in_taxi=0` |
| Destination | `Flag` | Always at `destPos` |
| Obstacle | `X` | At each obstacle cell |

#### Warning Overlay

Positioned `absolute inset-0` within the grid container. Uses `bg-black/65 backdrop-blur-sm` to create a semi-transparent dark overlay that blends with the grid rather than appearing as an external alert bar. The warning text is rendered inside at the bottom.

---

### 5.4 `ControlPanel.tsx` — Controls

A vertical sidebar with seven logical sections:

| Section | Content |
|---|---|
| **Grid Size** | Row/Col number inputs (range 2–20); triggers `onGridSizeChange` |
| **Placement Mode** | 4 radio/toggle buttons: Taxi · Passenger · Destination · Obstacle |
| **Step 1 — Initialize** | "Initialize Environment" button; disabled until at least taxi is placed |
| **Step 2 — BFS Check** | "Check Reachability" button; enabled after initialization |
| **Step 3 — Train Agent** | Algorithm radio (Q-Learning / Monte Carlo), episode count slider (500–10,000 with recommended label), Train button showing `Loader2` spinner while training |
| **Step 4 — Simulate** | Step button, Auto-Run button (with spinner), Stop button; all enabled after training |
| **Reset + Legend** | Reset button; colour legend for grid cells |

The episode slider `max` is capped to `min(recommendedEpisodes, 10000)`.

---

### 5.5 `InfoPanel.tsx` — Metrics

Three collapsible sections:

#### BFS Results
Shows for each leg:
- Reachable / Unreachable badge
- Path length (number of cells)

#### Training Metrics
- Algorithm name badge
- Episodes trained, training time
- Final success rate (colour-coded progress bar: red < 50%, yellow < 80%, green ≥ 80%)
- **Smoothed reward chart** (Recharts `LineChart`):
  - Applies a 50-episode rolling average to reduce noise
  - Downsamples to ≤200 points to avoid performance issues with large episode counts
  - `key` prop forces full re-mount on each new training run, bypassing Recharts' internal caching

```typescript
// key forces re-mount when training data changes
<LineChart
  key={`${trainMetrics.algorithm}-${trainMetrics.episodes}-${trainMetrics.train_time_seconds}`}
  ...>
```

#### Step Simulation Info
- Current 7-element state as colour-coded badges (taxi=blue, passenger=amber, destination=green, in_taxi=purple)
- Action name + reward for latest step
- Cumulative reward counter
- Step counter (`X / Total`)
- Terminal state message on episode completion
- Pickup / Dropoff phase indicator

---

### 5.6 `api.ts` — Service Layer

A thin typed wrapper over the browser Fetch API. All communication is `POST` with `Content-Type: application/json` to `http://localhost:8000`.

The core helper:

```typescript
async function post<TResponse>(endpoint: string, body: object): Promise<TResponse> {
  const res = await fetch(`${BASE_URL}${endpoint}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  const data = await res.json().catch(() => ({ detail: res.statusText }));

  if (!res.ok) {
    const message = typeof data.detail === "string"
      ? data.detail
      : JSON.stringify(data.detail ?? data);
    throw new Error(message);
  }

  return data as TResponse;
}
```

FastAPI's `HTTPException` produces `{ "detail": "..." }` JSON. This helper reads `data.detail` and rethrows as a native `Error`, which propagates to `App.tsx`'s `catch` blocks and is forwarded to `showWarning()`.

The exported `api` object provides six typed methods: `initialize`, `bfsCheck`, `train`, `step`, `simulate`, `reset`.

---

## 6. Data Flows & Workflows

### 6.1 Initialization Flow

```
User places entities on grid → clicks "Initialize Environment"
  │
  ├─ Frontend: validates taxiPos is set
  ├─ POST /initialize { rows, cols, obstacles, taxi, passenger, destination }
  │    │
  │    ├─ Backend: TaxiGridEnv(rows, cols, obstacles)
  │    ├─ env.set_positions(taxi, passenger, destination)
  │    │    └─ auto-randomises any null position
  │    ├─ validate_reachability(env)  [BFS check]
  │    │    └─ raises ReachabilityError → 400 if path blocked
  │    ├─ QLearningAgent(env) + MonteCarloAgent(env) created
  │    └─ Returns: resolved positions, recommended_episodes, recommended_max_steps
  │
  ├─ Frontend: stores resolvedTaxi/Pass/Dest
  ├─ Syncs displayed positions back to resolved values
  ├─ Sets episodeCount = min(recommended, 10000)
  └─ isInitialized = true  →  grid becomes non-interactive
```

### 6.2 BFS Check Flow

```
User clicks "Check Reachability"
  │
  ├─ POST /bfs-check {}
  │    │
  │    ├─ Backend: env.reset()  (restore initial positions)
  │    ├─ bfs_with_path(env, taxi, passenger)  → (path1, explored1, ok1)
  │    ├─ bfs_with_path(env, passenger, destination) → (path2, explored2, ok2)
  │    └─ Returns: leg1{start,goal,reachable,explored,path}, leg2{...}, both_reachable
  │
  ├─ Frontend: setBfsResult(res), setBfsAnimIndex(0)
  └─ Starts setInterval at 12ms:
       └─ every tick: bfsAnimIndex += 2
            ├─ reveals explored1 cells (yellow) up to bfsAnimIndex
            ├─ reveals explored2 cells (yellow) from bfsAnimIndex - leg1ExpLen
            ├─ bfsPath1 (orange) appears when bfsAnimIndex ≥ leg1ExpLen
            └─ bfsPath2 (orange) appears when bfsAnimIndex ≥ totalExplored
```

### 6.3 Training Flow

```
User selects algorithm + episodes → clicks "Train"
  │
  ├─ Frontend: isTraining=true, isTrained=false, resets sim
  ├─ POST /train { algorithm, episodes }
  │    │
  │    ├─ Backend: env.reset()
  │    ├─ episodes = min(requested, 10000)
  │    ├─ loop.run_in_executor(None, agent.train, eps, steps)
  │    │    └─ runs in thread pool (non-blocking asyncio)
  │    │
  │    │   [Q-Learning: ep * max_steps Bellman updates]
  │    │   [Monte Carlo: ep episodes, auto-scaled epsilon decay]
  │    │
  │    ├─ trained["algorithm"] = req.algorithm
  │    └─ Returns: train_time, final_success_rate, episode_rewards[], episode_successes[], episode_lengths[]
  │
  ├─ Frontend: isTrained=true, trainMetrics=res
  └─ InfoPanel renders smoothed reward chart
```

### 6.4 Step-by-Step Simulation Flow

```
User clicks "Step" (repeatedly)
  │
  ├─ [First step only] clears BFS overlay
  │
  ├─ currentState = simPath?.[simIndex] ?? initial_state_from_resolved_positions
  │
  ├─ POST /step { algorithm, state: currentState }
  │    │
  │    ├─ Backend: env.inject_state(state)
  │    ├─ action = agent.get_greedy_action(state)
  │    ├─ next_obs, reward, terminated = env.step(action)
  │    └─ Returns: action, action_name, next_state, reward, terminated
  │
  ├─ Frontend: simPath.push(next_state), simIndex++
  ├─ cumReward += reward
  ├─ setStepInfo({ action, reward, cumReward, terminated, ... })
  └─ Grid updates: rlCurrentPos → next_state[0:2], rlPath grows, rlPhase = next_state[6]
```

### 6.5 Auto-Run Simulation Flow

```
User clicks "Auto-Run"
  │
  ├─ Frontend: resets sim, clears BFS overlay
  ├─ POST /simulate { algorithm }
  │    │
  │    ├─ Backend: env.reset()
  │    ├─ success, path = agent.run(max_steps)
  │    └─ Returns: success, total_steps, path: RLState[]  (full trajectory)
  │
  ├─ Frontend: setSimPath(path), setSimIndex(0), setIsAutoRunning(true)
  └─ setInterval at 450ms:
       └─ each tick: idx++
            ├─ setSimIndex(idx)
            ├─ infer reward from state diff (simplified heuristic)
            ├─ setStepInfo(...)
            └─ if idx >= path.length - 1: clearInterval, setIsAutoRunning(false)
```

### 6.6 Reset Flow

```
User clicks "Reset"
  │
  ├─ Stop auto-run interval
  ├─ Clear BFS and warning timers
  ├─ POST /reset {}  → backend clears _state dict
  ├─ Frontend: all state → initial values
  │    rows=5, cols=5
  │    taxiPos=null, startPos=null, destPos=null
  │    obstacles=[]
  │    placementMode="taxi"
  │    isInitialized=false, isTrained=false
  │    trainMetrics=null, bfsResult=null
  │    simPath=null, simIndex=0, stepInfo=null
  └─ Grid becomes interactive again (empty 5×5)
```

---

## 7. State Machine (Frontend)

The application moves through these phases driven by `isInitialized` and `isTrained` booleans:

```
┌─────────────┐
│    SETUP    │  Grid is interactive (click to place entities)
│             │  ControlPanel shows: Grid Size, Placement Mode, Initialize button
└──────┬──────┘
       │ /initialize succeeds
       ▼
┌─────────────┐
│    READY    │  Grid locked (non-interactive)
│             │  BFS Check available
│             │  Train section enabled
└──────┬──────┘
       │ /bfs-check (optional, can skip)
       │ /train succeeds
       ▼
┌─────────────┐
│   TRAINED   │  Step and Auto-Run buttons enabled
│             │  Can switch algorithm → goes back to READY (isTrained=false)
└──────┬──────┘
       │ /step or /simulate called
       ▼
┌─────────────┐
│  SIMULATING │  Step: user clicks Step repeatedly
│             │  Auto-Run: 450ms interval plays animation
│             │  Stop button available during Auto-Run
└──────┬──────┘
       │ terminated=true or path exhausted
       ▼
┌─────────────┐
│    DONE     │  Terminal state message shown
│             │  Step button still advances (continues past done)
└──────┬──────┘
       │ /reset (any phase → SETUP)
       └──────────────────────────────►
```

**Algorithm switch** at any phase after initialization resets `isTrained=false` and `trainMetrics=null`, requiring a new `/train` call for the new algorithm.

---

## 8. Algorithm Deep-Dives

### 8.1 BFS

- **Type:** Uninformed graph search
- **Complexity:** O(V + E) where V = free cells, E = adjacency edges
- **Guarantees:** Optimal path (fewest cells) in an unweighted grid
- **Role:** Validates environment before training; provides path length baseline; drives the animated yellow/orange overlay

The animation reveals cells in BFS dequeue order, which visually demonstrates the "flood-fill" nature of BFS expanding outward from the start node.

### 8.2 Q-Learning

- **Type:** Off-policy TD control
- **Update frequency:** Every step (online learning)
- **Convergence:** Guaranteed to converge to optimal Q* under standard conditions (sufficient exploration, decaying step size)
- **Strength:** Bootstraps from incomplete trajectories; works even when episodes rarely complete early in training
- **Weakness:** Requires many steps to propagate rewards backward across long action sequences

**Epsilon decay rate** with default `0.995` across 5000 episodes:
```
ep    920 → epsilon ≈ 0.01 (floor reached)
```

### 8.3 Monte Carlo

- **Type:** Model-free, Monte Carlo control
- **Update frequency:** End of each episode (batch learning)
- **Convergence:** Converges under first-visit policy evaluation conditions
- **Strength:** Uses actual sampled returns (no bootstrap bias); often converges to better policies with sufficient exploration
- **Weakness:** Must complete full episodes to learn; degenerates if exploration collapses before Q-values converge

**Epsilon decay (post-fix)** with `auto_decay` for 5000 episodes:
```
auto_decay ≈ 0.99885
ep    500 → epsilon ≈ 0.56
ep   1000 → epsilon ≈ 0.32
ep   4000 → epsilon ≈ 0.01 (floor reached at 80% of training)
ep   5000 → epsilon = 0.01 (clamped)
```

**Per-episode step budget (post-fix):** `max(max_steps * 3, rows * cols * 15)` — gives random exploration enough steps to stumble upon complete pickup+dropoff trajectories in early episodes.

### 8.4 Algorithm Comparison

| Aspect | Q-Learning | Monte Carlo |
|---|---|---|
| Update trigger | Every step | End of episode |
| Bootstrapping | Yes | No |
| Needs complete episodes | No | Yes |
| Suitable for long episodes | Yes | Harder |
| Convergence sensitivity | Lower | Higher (exploration must persist) |
| Typical performance (5×5 grid, 5000 ep) | ~100% success | ~100% success (post-fix) |
| Epsilon floor timing | ~ep 920 | ~ep 4000 (dynamic scaling) |

---

## 9. Key Design Decisions

### Stateless `/step` endpoint
The `/step` endpoint accepts the full 7-element state on every call. The frontend owns the state cursor. This makes step-by-step simulation resumable after page reload (if state is persisted) and decouples the backend from managing per-session simulation cursors.

### `inject_state()` on the environment
Rather than maintaining a parallel simulation loop in the API layer, the environment's own `step()` logic is reused by injecting the client-supplied state directly. This ensures simulation results exactly match the training environment's physics.

### Async training with `run_in_executor`
Training is CPU-bound Python code. Running it synchronously in a FastAPI route handler would block the uvicorn event loop, preventing `/health` checks and other requests from being served. `run_in_executor` offloads the work to a thread pool without requiring restructuring the training code as `async`.

### BFS explored-node ordering for animation
The `explored` list in `bfs_with_path()` records cells in dequeue order — the exact sequence BFS visits them. This allows the frontend to animate the exploration wave by simply slicing the list up to an incrementing index, without any additional ordering logic.

### Inline styles instead of dynamic Tailwind classes
Tailwind v4's JIT scanner analyses source files at build time and removes unused class names. Classes assigned to variables (e.g., `let bg = isObstacle ? "bg-gray-800" : "bg-blue-200"`) are not detected. All dynamic cell colours are applied via `style={{ backgroundColor: "#..." }}` with hardcoded hex values, which are invisible to the scanner.

### `key` prop on Recharts `LineChart`
Recharts memoises its internal chart state. Updating the `data` prop alone does not reliably trigger a full re-render across training runs. Providing a `key` that changes per run (derived from algorithm + episodes + train_time) forces React to unmount and remount the chart, resetting all internal state and guaranteeing the new data is displayed.

### Dual simulation modes
- **Step mode** (`/step`): Calls the backend per click. Good for inspecting each decision. The path is built incrementally on the client.
- **Auto-run mode** (`/simulate`): Fetches the entire trajectory in one call, then animates it client-side with `setInterval`. This avoids N sequential HTTP round-trips for large grids.

---

## 10. Known Bugs Fixed

### Bug 1 — Tailwind v4 Dynamic Class Purging
**Symptom:** Grid cells showed incorrect/default colours for obstacles, BFS paths, and RL trail in production build.
**Root cause:** Tailwind v4 JIT scanner could not detect class names assigned inside ternary expressions to `let` variables.
**Fix:** Replaced all dynamic class names in `TaxiGrid.tsx` with inline `style` props using hardcoded hex colour strings.

### Bug 2 — Reset Not Clearing Grid
**Symptom:** Clicking Reset cleared the backend state but left the grid in its previous configuration.
**Root cause:** `handleReset` in `App.tsx` was missing state setter calls for `rows`, `cols`, `taxiPos`, `startPos`, `destPos`, `obstacles`, and `placementMode`.
**Fix:** Added all missing `set*` calls to restore initial values (5×5, all null positions, empty obstacles, taxi placement mode).

### Bug 3 — Success Rate Chart Not Updating
**Symptom:** InfoPanel's reward chart showed the previous training run's data even after training a new run.
**Root cause:** Recharts `LineChart` cached its internal state and did not re-render when only the `data` prop changed across re-renders of the same component instance.
**Fix:** Added a composite `key` prop to `LineChart` combining `algorithm + episodes + train_time_seconds`. Any new training run changes at least one of these, forcing React to remount the chart.

### Bug 4 — BFS Overlay Persisting During Simulation
**Symptom:** Yellow BFS explored cells and orange BFS path cells remained visible underneath the blue/purple RL trail, creating visual confusion.
**Root cause:** `bfsResult` state was not cleared when simulation began.
**Fix:** In both `handleStep` (first step only) and `handleAutoRun`, added `setBfsResult(null); setBfsAnimIndex(0)` before the simulation starts.

### Bug 5 — Monte Carlo 0% Success Rate
**Symptom:** Q-Learning trained successfully but Monte Carlo with identical settings showed 0% success rate and "agent could not complete" during simulation.
**Root cause:** Default `epsilon_decay=0.995` caused epsilon to reach 0.01 at ~episode 1000/5000. Monte Carlo requires complete successful episodes to update Q-values. A partially-converged policy with near-zero exploration degenerates into deterministic loops (always choosing action 0 = "down").
**Fix:** Dynamic epsilon decay computed in `MonteCarloAgent.train()` to target the floor at 80% of total episodes. Tripled per-episode step budget to allow random exploration to occasionally complete full trajectories.

---

## 11. Coordinate Convention

All coordinates throughout the system use **`[row, col]`** (row-major) format:

- Row 0 is the **top** of the grid
- Column 0 is the **left** of the grid
- Backend Python: `(row, col)` tuples
- Backend API: `[row, col]` JSON arrays (`List[int]` in Pydantic)
- Frontend TypeScript: `{ row, col }` objects (for React state), `[row, col]` arrays (in `RLState` and API payloads)

The `RLState` tuple positions map as:

```
index: [0,    1,    2,    3,    4,    5,    6     ]
name:  [tr,   tc,   pr,   pc,   dr,   dc,   in_taxi]
```

---

## 12. Configuration Reference

### Backend Constants (`main.py`)

| Constant | Value | Purpose |
|---|---|---|
| `MAX_EPISODES_CAP` | 10,000 | Hard cap on episodes per `/train` call |
| `ACTION_NAMES` | `{0:"down",1:"up",2:"right",3:"left",4:"pickup",5:"dropoff"}` | Human-readable action labels |

### Frontend Constants (`App.tsx`)

| Constant | Value | Purpose |
|---|---|---|
| `AUTO_RUN_INTERVAL_MS` | 450 ms | Delay between steps in auto-run animation |
| `WARNING_DURATION_MS` | 4000 ms | How long warning overlays persist |
| `BFS_EXPLORE_INTERVAL` | 12 ms | Interval between BFS cell reveals |
| Cells per BFS tick | 2 | Cells revealed per interval tick |

### Grid Constraints

| Property | Min | Max |
|---|---|---|
| Grid rows | 2 | 20 |
| Grid cols | 2 | 20 |
| Free cells required | 3 | — |
| Entities | 3 (taxi, passenger, destination) | — |

### Episode Slider

| Property | Value |
|---|---|
| Minimum | 500 |
| Maximum | `min(recommended_episodes, 10000)` |
| Default | `recommended_episodes` from backend |
| Recommended formula | `max(5000, rows × cols × 100)` |

---

*End of Analysis*
