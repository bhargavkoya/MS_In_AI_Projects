# rl_logic.py — All RL logic extracted from the notebook
# Cells 1-4 reproduced verbatim, plus extended BFS utilities.

# ── Imports ───────────────────────────────────────────────────────────────────
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random as _random
import random
import time
from collections import deque, defaultdict
from datetime import datetime


# ═════════════════════════════════════════════════════════════════════════════
# Cell 1 — TaxiGridEnv
# ═════════════════════════════════════════════════════════════════════════════

class TaxiGridEnv(gym.Env):

    def __init__(self, rows, cols, obstacles=None):
        super().__init__()
        if rows < 2 or cols < 2:
            raise ValueError(f"Grid must be at least 2x2, got {rows}x{cols}.")

        self.rows      = rows
        self.cols      = cols
        self.obstacles = set(map(tuple, obstacles)) if obstacles else set()

        self.action_space      = spaces.Discrete(6)
        self.observation_space = spaces.MultiDiscrete(
            [rows, cols, rows, cols, rows, cols, 2]
        )

        self.taxi              = None
        self.passenger         = None
        self.destination       = None
        self.passenger_in_taxi = 0
        self._init_taxi        = None
        self._init_passenger   = None
        self._init_destination = None

    def _in_bounds(self, pos):
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_valid(self, pos):
        return self._in_bounds(pos) and tuple(pos) not in self.obstacles

    def _free_cells(self):
        return [
            (r, c) for r in range(self.rows) for c in range(self.cols)
            if (r, c) not in self.obstacles
        ]

    def _validate_position(self, pos, label):
        if pos is None:
            raise ValueError(f"{label} position has not been set.")
        pos = tuple(pos)
        if not self._in_bounds(pos):
            raise ValueError(
                f"{label} position {pos} is out of bounds. "
                f"Grid is {self.rows}x{self.cols} "
                f"(rows 0-{self.rows-1}, cols 0-{self.cols-1})."
            )
        if pos in self.obstacles:
            raise ValueError(
                f"{label} position {pos} overlaps an obstacle. "
                f"Obstacles: {sorted(self.obstacles)}"
            )

    def _validate_all_positions(self):
        self._validate_position(self.taxi,        'Taxi')
        self._validate_position(self.passenger,   'Passenger')
        self._validate_position(self.destination, 'Destination')
        seen = {}
        for label, pos in [('Taxi', self.taxi), ('Passenger', self.passenger),
                            ('Destination', self.destination)]:
            if pos in seen:
                raise ValueError(
                    f"{label} and {seen[pos]} share cell {pos}. "
                    "Each entity must occupy a unique cell."
                )
            seen[pos] = label

    def set_positions(self, taxi, passenger, destination):
        """Set positions. Pass None to auto-place randomly on a free cell."""
        free = self._free_cells()
        if len(free) < 3:
            raise ValueError("Grid has fewer than 3 free cells.")
        used = set()

        def pick(pos, label):
            if pos is not None:
                return tuple(pos)
            candidates = [c for c in free if c not in used]
            if not candidates:
                raise ValueError(f"No free cell available for {label}.")
            return _random.choice(candidates)

        self.taxi        = pick(taxi,        'Taxi');        used.add(self.taxi)
        self.passenger   = pick(passenger,   'Passenger');   used.add(self.passenger)
        self.destination = pick(destination, 'Destination')

        self._validate_all_positions()
        self._init_taxi        = self.taxi
        self._init_passenger   = self.passenger
        self._init_destination = self.destination

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.taxi              = self._init_taxi
        self.passenger         = self._init_passenger
        self.destination       = self._init_destination
        self.passenger_in_taxi = 0
        return self._get_obs(), {}

    def _get_obs(self):
        return np.array([
            self.taxi[0],        self.taxi[1],
            self.passenger[0],   self.passenger[1],
            self.destination[0], self.destination[1],
            self.passenger_in_taxi
        ])

    def step(self, action):
        reward     = -1
        terminated = False
        r, c       = self.taxi
        moves      = {0: (r+1,c), 1: (r-1,c), 2: (r,c+1), 3: (r,c-1)}

        if action in moves:
            nxt = moves[action]
            if self.is_valid(nxt):
                self.taxi = nxt
                if self.passenger_in_taxi:
                    self.passenger = self.taxi
        elif action == 4:
            if self.taxi == self.passenger and self.passenger_in_taxi == 0:
                self.passenger_in_taxi = 1
                reward = 10
            else:
                reward = -10
        elif action == 5:
            if self.passenger_in_taxi == 1 and self.taxi == self.destination:
                reward                 = 20
                terminated             = True
                self.passenger_in_taxi = 0
            else:
                reward = -10

        return self._get_obs(), reward, terminated, False, {}

    def inject_state(self, state):
        """Force the environment into a specific 7-element state (for /step endpoint)."""
        self.taxi              = (int(state[0]), int(state[1]))
        self.passenger         = (int(state[2]), int(state[3]))
        self.destination       = (int(state[4]), int(state[5]))
        self.passenger_in_taxi = int(state[6])

    def render(self, phase=''):
        grid = [[' '] * self.cols for _ in range(self.rows)]
        for r, c in self.obstacles:
            grid[r][c] = 'X'
        dr, dc = self.destination
        grid[dr][dc] = 'D'
        if self.passenger_in_taxi == 0:
            pr, pc = self.passenger
            grid[pr][pc] = 'P'
        tr, tc = self.taxi
        grid[tr][tc] = 'T'
        sep   = '-' * (self.cols * 4)
        label = f'  [{phase}]' if phase else ''
        print(sep + label)
        for row in grid:
            print('| ' + ' | '.join(row) + ' |')
            print(sep)


# ═════════════════════════════════════════════════════════════════════════════
# Cell 2 — BFS Utilities (extended)
# ═════════════════════════════════════════════════════════════════════════════

class ReachabilityError(Exception):
    """Raised when BFS confirms a required path does not exist."""
    pass


def bfs_with_path(env, start, goal):
    """
    Extended BFS that returns the optimal path AND the exploration order.

    Returns:
        path     : list of (row, col) tuples from start to goal (inclusive).
                   Empty list if unreachable.
        explored : all cells dequeued during BFS, in order (for animation).
        reachable: bool
    """
    start = tuple(start)
    goal  = tuple(goal)

    if start == goal:
        return [start], [start], True

    # Maps each node to its parent for path reconstruction
    parent   = {start: None}
    queue    = deque([start])
    explored = []

    while queue:
        r, c = queue.popleft()
        explored.append((r, c))

        for nxt in [(r+1, c), (r-1, c), (r, c+1), (r, c-1)]:
            if nxt == goal:
                # Reconstruct path
                explored.append(goal)
                path = [goal]
                cur  = (r, c)
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                path.reverse()
                return path, explored, True

            if nxt not in parent and env.is_valid(nxt):
                parent[nxt] = (r, c)
                queue.append(nxt)

    return [], explored, False


def bfs_reachable(env, start, goal):
    """Thin wrapper — backward compatibility."""
    _, _, reachable = bfs_with_path(env, start, goal)
    return reachable


def validate_reachability(env):
    """
    Validate taxi->passenger and passenger->destination reachability.
    Raises ReachabilityError with a descriptive message on failure.
    """
    if not bfs_reachable(env, env.taxi, env.passenger):
        raise ReachabilityError(
            f"Taxi at {env.taxi} cannot reach Passenger at {env.passenger}. "
            "Path is blocked by obstacles or grid boundaries."
        )
    if not bfs_reachable(env, env.passenger, env.destination):
        raise ReachabilityError(
            f"Passenger at {env.passenger} cannot reach Destination at {env.destination}. "
            "Path is blocked by obstacles or grid boundaries."
        )


def recommended_episodes(rows, cols):
    return max(5000, rows * cols * 100)


def recommended_max_steps(rows, cols):
    return max(200, rows * cols * 4)


# ═════════════════════════════════════════════════════════════════════════════
# Cell 3 — QLearningAgent
# ═════════════════════════════════════════════════════════════════════════════

class QLearningAgent:
    """
    Q-Learning: updates the Q-table after EVERY step using the Bellman equation.
    Q(s,a) <- Q(s,a) + alpha * [r + gamma * max Q(s',a') - Q(s,a)]
    """
    name = 'Q-Learning'

    def __init__(self, env, alpha=0.1, gamma=0.95,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.env           = env
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end   = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.q_table       = {}
        self.train_time    = 0.0
        self.metrics       = {}

    def _get_q(self, state):
        key = tuple(state)
        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.env.action_space.n)
        return self.q_table[key]

    def get_greedy_action(self, state):
        return int(np.argmax(self._get_q(state)))

    def train(self, episodes=None, max_steps=None):
        rows, cols = self.env.rows, self.env.cols
        if episodes  is None: episodes  = recommended_episodes(rows, cols)
        if max_steps is None: max_steps = recommended_max_steps(rows, cols)

        self.q_table = {}
        self.epsilon = self.epsilon_start

        print(f'[Q-Learning] Training {episodes} episodes | max_steps={max_steps}')
        success_count    = 0
        log_every        = max(1, episodes // 10)
        t0               = time.time()

        episode_rewards   = []
        episode_successes = []
        episode_lengths   = []

        for ep in range(1, episodes + 1):
            state, _     = self.env.reset()
            total_reward = 0
            steps        = 0

            for _ in range(max_steps):
                if random.random() < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = int(np.argmax(self._get_q(state)))

                next_state, reward, terminated, _, _ = self.env.step(action)

                best_future = np.max(self._get_q(next_state))
                self._get_q(state)[action] += self.alpha * (
                    reward + self.gamma * best_future - self._get_q(state)[action]
                )

                state        = next_state
                total_reward += reward
                steps        += 1
                if terminated:
                    success_count += 1
                    break

            episode_rewards.append(total_reward)
            episode_successes.append(1 if terminated else 0)
            episode_lengths.append(steps)

            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            if ep % log_every == 0:
                rate = success_count / log_every * 100
                print(f'  Episode {ep:>7}/{episodes} | '
                      f'epsilon={self.epsilon:.4f} | success rate={rate:.1f}%')
                success_count = 0

        self.train_time = time.time() - t0
        print(f'[Q-Learning] Done in {self.train_time:.2f}s | '
              f'Q-table states={len(self.q_table)}\n')

        self.metrics = {
            'episode_rewards':   episode_rewards,
            'episode_successes': episode_successes,
            'episode_lengths':   episode_lengths,
        }
        return self.metrics

    def run(self, max_steps=None):
        if max_steps is None:
            max_steps = recommended_max_steps(self.env.rows, self.env.cols)
        state, _ = self.env.reset()
        path      = [tuple(state)]
        for _ in range(max_steps):
            action = int(np.argmax(self._get_q(state)))
            next_state, _, terminated, _, _ = self.env.step(action)
            path.append(tuple(next_state))
            state = next_state
            if terminated:
                return True, path
        return False, path


# ═════════════════════════════════════════════════════════════════════════════
# Cell 4 — MonteCarloAgent
# ═════════════════════════════════════════════════════════════════════════════

class MonteCarloAgent:
    """
    First-Visit Monte Carlo Control with epsilon-greedy exploration.
    Waits until episode ends, then computes actual discounted return G
    for each first-visit (state, action) pair.
    Q(s,a) <- incremental mean of all G seen from (s,a).
    """
    name = 'Monte Carlo'

    def __init__(self, env, gamma=0.95,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.env           = env
        self.gamma         = gamma
        self.epsilon       = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end   = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.q_table       = defaultdict(lambda: np.zeros(env.action_space.n))
        self.returns_n     = defaultdict(lambda: np.zeros(env.action_space.n))
        self.train_time    = 0.0
        self.metrics       = {}

    def get_greedy_action(self, state):
        return int(np.argmax(self.q_table[tuple(state)]))

    def _policy(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        return int(np.argmax(self.q_table[tuple(state)]))

    def _generate_episode(self, max_steps):
        episode  = []
        state, _ = self.env.reset()
        for _ in range(max_steps):
            action = self._policy(state)
            next_state, reward, terminated, _, _ = self.env.step(action)
            episode.append((tuple(state), action, reward, terminated))
            state = next_state
            if terminated:
                break
        return episode

    def train(self, episodes=None, max_steps=None):
        rows, cols = self.env.rows, self.env.cols
        if episodes  is None: episodes  = recommended_episodes(rows, cols)
        if max_steps is None: max_steps = recommended_max_steps(rows, cols)

        self.q_table   = defaultdict(lambda: np.zeros(self.env.action_space.n))
        self.returns_n = defaultdict(lambda: np.zeros(self.env.action_space.n))
        self.epsilon   = self.epsilon_start

        # Monte Carlo needs complete trajectories to learn — slow epsilon decay so
        # exploration stays alive long enough for the Q-table to converge.
        # Target: epsilon reaches epsilon_end at 80% of training, not at ~20%.
        auto_decay = (self.epsilon_end / self.epsilon_start) ** (
            1.0 / max(1, int(episodes * 0.8))
        )
        effective_decay = max(self.epsilon_decay, auto_decay)  # higher = slower decay

        # Give each training episode a larger step budget so random exploration can
        # occasionally produce complete pickup→dropoff trajectories from the start.
        episode_max_steps = max(max_steps * 3, rows * cols * 15)

        print(f'[Monte Carlo] Training {episodes} episodes | '
              f'max_steps={episode_max_steps} | epsilon_decay={effective_decay:.6f}')
        success_count    = 0
        log_every        = max(1, episodes // 10)
        t0               = time.time()

        episode_rewards   = []
        episode_successes = []
        episode_lengths   = []

        for ep in range(1, episodes + 1):
            episode = self._generate_episode(episode_max_steps)

            total_reward = sum(r for _, _, r, _ in episode)
            success      = episode[-1][3] if episode else False
            if success:
                success_count += 1

            episode_rewards.append(total_reward)
            episode_successes.append(1 if success else 0)
            episode_lengths.append(len(episode))

            G       = 0.0
            visited = set()
            for state, action, reward, _ in reversed(episode):
                G  = reward + self.gamma * G
                sa = (state, action)
                if sa not in visited:
                    visited.add(sa)
                    self.returns_n[state][action] += 1
                    n = self.returns_n[state][action]
                    self.q_table[state][action]   += (G - self.q_table[state][action]) / n

            self.epsilon = max(self.epsilon_end, self.epsilon * effective_decay)
            if ep % log_every == 0:
                rate = success_count / log_every * 100
                print(f'  Episode {ep:>7}/{episodes} | '
                      f'epsilon={self.epsilon:.4f} | success rate={rate:.1f}%')
                success_count = 0

        self.train_time = time.time() - t0
        print(f'[Monte Carlo] Done in {self.train_time:.2f}s | '
              f'Q-table states={len(self.q_table)}\n')

        self.metrics = {
            'episode_rewards':   episode_rewards,
            'episode_successes': episode_successes,
            'episode_lengths':   episode_lengths,
        }
        return self.metrics

    def run(self, max_steps=None):
        if max_steps is None:
            max_steps = recommended_max_steps(self.env.rows, self.env.cols) * 3
        state, _ = self.env.reset()
        path      = [tuple(state)]
        for _ in range(max_steps):
            action = int(np.argmax(self.q_table[tuple(state)]))
            next_state, _, terminated, _, _ = self.env.step(action)
            path.append(tuple(next_state))
            state = next_state
            if terminated:
                return True, path
        return False, path
