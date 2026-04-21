import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";
import type { RLState, TrainResponse } from "../services/api";

// ── Types ─────────────────────────────────────────────────────────────────────

export interface StepInfo {
  stepIndex: number;
  totalSteps: number;
  currentState: RLState;
  action: number;
  actionName: string;
  reward: number;
  cumulativeReward: number;
  terminated: boolean;
}

interface InfoPanelProps {
  stepInfo: StepInfo | null;
  trainMetrics: TrainResponse | null;
  bfsReachable: { leg1: boolean; leg2: boolean } | null;
  bfsPathLengths: { leg1: number; leg2: number } | null;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

const ACTION_LABELS: Record<number, string> = {
  0: "↓ Down",
  1: "↑ Up",
  2: "→ Right",
  3: "← Left",
  4: "⬆ Pickup",
  5: "⬇ Dropoff",
};

const STATE_LABELS = ["Taxi R", "Taxi C", "Pass R", "Pass C", "Dest R", "Dest C", "In Taxi"];

const STATE_COLORS = [
  "bg-yellow-100 text-yellow-800 border-yellow-300",
  "bg-yellow-100 text-yellow-800 border-yellow-300",
  "bg-green-100 text-green-800 border-green-300",
  "bg-green-100 text-green-800 border-green-300",
  "bg-red-100 text-red-800 border-red-300",
  "bg-red-100 text-red-800 border-red-300",
  "bg-purple-100 text-purple-800 border-purple-300",
];

/** Downsample array to at most `maxPoints` evenly-spaced values */
function downsample(arr: number[], maxPoints = 200): { episode: number; value: number }[] {
  if (arr.length <= maxPoints) {
    return arr.map((v, i) => ({ episode: i + 1, value: v }));
  }
  const step = arr.length / maxPoints;
  return Array.from({ length: maxPoints }, (_, i) => {
    const idx = Math.min(Math.floor(i * step), arr.length - 1);
    return { episode: idx + 1, value: arr[idx] };
  });
}

/** Rolling average smoother */
function smooth(arr: number[], window = 50): number[] {
  return arr.map((_, i) => {
    const start = Math.max(0, i - window + 1);
    const slice = arr.slice(start, i + 1);
    return slice.reduce((a, b) => a + b, 0) / slice.length;
  });
}

// ── Sub-components ────────────────────────────────────────────────────────────

function StateBadges({ state }: { state: RLState }) {
  return (
    <div className="flex flex-wrap gap-1">
      {state.map((val, i) => (
        <div
          key={i}
          className={`flex flex-col items-center px-2 py-1 rounded border text-xs font-mono ${STATE_COLORS[i]}`}
        >
          <span className="text-[10px] opacity-60 font-sans">{STATE_LABELS[i]}</span>
          <span className="font-bold text-sm">{val}</span>
        </div>
      ))}
    </div>
  );
}

function RewardBadge({ reward }: { reward: number }) {
  const isPositive = reward > 0;
  const isNegative = reward < 0;
  return (
    <span
      className={`px-3 py-1 rounded-full font-bold text-sm ${
        isPositive
          ? "bg-green-100 text-green-700 border border-green-300"
          : isNegative
          ? "bg-red-100 text-red-700 border border-red-300"
          : "bg-gray-100 text-gray-700 border border-gray-300"
      }`}
    >
      {isPositive ? "+" : ""}
      {reward}
    </span>
  );
}

// ── Main Component ────────────────────────────────────────────────────────────

export function InfoPanel({ stepInfo, trainMetrics, bfsReachable, bfsPathLengths }: InfoPanelProps) {
  const hasStepInfo = stepInfo !== null;
  const hasMetrics = trainMetrics !== null;
  const hasBfs = bfsReachable !== null;

  const inTaxi = stepInfo ? stepInfo.currentState[6] === 1 : false;
  const phase = inTaxi ? "Dropoff Phase" : "Pickup Phase";
  const phaseColor = inTaxi
    ? "bg-purple-100 text-purple-700 border border-purple-300"
    : "bg-blue-100 text-blue-700 border border-blue-300";

  // Prepare chart data
  const chartData = hasMetrics
    ? downsample(smooth(trainMetrics.episode_rewards, 50), 200)
    : [];

  // final_success_rate comes from backend as a fraction 0-1; multiply to get %
  const rawRate = hasMetrics ? trainMetrics.final_success_rate : 0;
  const successRate = Math.min(100, Math.max(0, Math.round(rawRate * 100)));

  if (!hasStepInfo && !hasMetrics && !hasBfs) {
    return (
      <div className="bg-white rounded-lg shadow-lg p-6 w-full">
        <h2 className="font-semibold text-gray-700 mb-3">Info Panel</h2>
        <p className="text-sm text-gray-400 italic">
          Initialize the environment to get started.
        </p>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-lg p-4 w-full space-y-4">
      <h2 className="font-semibold text-gray-700 text-base">Info Panel</h2>

      {/* ── BFS Summary ── */}
      {hasBfs && (
        <div className="border border-gray-200 rounded-lg p-3 space-y-2">
          <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wide">
            BFS Results
          </h3>
          <div className="space-y-1 text-sm">
            <div className="flex items-center justify-between">
              <span className="text-gray-600">Taxi → Passenger</span>
              <div className="flex items-center gap-2">
                {bfsPathLengths && (
                  <span className="text-xs text-gray-400">{bfsPathLengths.leg1} steps</span>
                )}
                <span
                  className={`px-2 py-0.5 rounded-full text-xs font-semibold ${
                    bfsReachable.leg1
                      ? "bg-green-100 text-green-700"
                      : "bg-red-100 text-red-700"
                  }`}
                >
                  {bfsReachable.leg1 ? "Reachable" : "Blocked"}
                </span>
              </div>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-gray-600">Passenger → Destination</span>
              <div className="flex items-center gap-2">
                {bfsPathLengths && (
                  <span className="text-xs text-gray-400">{bfsPathLengths.leg2} steps</span>
                )}
                <span
                  className={`px-2 py-0.5 rounded-full text-xs font-semibold ${
                    bfsReachable.leg2
                      ? "bg-green-100 text-green-700"
                      : "bg-red-100 text-red-700"
                  }`}
                >
                  {bfsReachable.leg2 ? "Reachable" : "Blocked"}
                </span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Training Metrics ── */}
      {hasMetrics && (
        <div className="border border-gray-200 rounded-lg p-3 space-y-3">
          <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wide">
            Training Results
          </h3>

          {/* Algorithm + time */}
          <div className="flex items-center gap-2 flex-wrap">
            <span className="px-2 py-0.5 rounded bg-indigo-100 text-indigo-700 text-xs font-semibold capitalize">
              {trainMetrics.algorithm === "qlearning" ? "Q-Learning" : "Monte Carlo"}
            </span>
            <span className="text-xs text-gray-400">
              {trainMetrics.episodes.toLocaleString()} episodes
            </span>
            <span className="text-xs text-gray-400">
              {trainMetrics.train_time_seconds.toFixed(1)}s
            </span>
          </div>

          {/* Success rate bar */}
          <div>
            <div className="flex justify-between items-center mb-1">
              <span className="text-xs text-gray-600">Final success rate</span>
              <span
                className={`text-sm font-bold ${
                  successRate >= 70
                    ? "text-green-600"
                    : successRate >= 40
                    ? "text-yellow-600"
                    : "text-red-600"
                }`}
              >
                {successRate}%
              </span>
            </div>
            <div className="w-full bg-gray-100 rounded-full h-2">
              <div
                className={`h-2 rounded-full transition-all duration-500 ${
                  successRate >= 70
                    ? "bg-green-500"
                    : successRate >= 40
                    ? "bg-yellow-500"
                    : "bg-red-500"
                }`}
                style={{ width: `${successRate}%` }}
              />
            </div>
          </div>

          {/* Rewards chart */}
          <div>
            <p className="text-xs text-gray-500 mb-1">Episode rewards (smoothed)</p>
            <ResponsiveContainer width="100%" height={110}>
              {/* key forces a full remount so recharts picks up new data on retrain */}
              <LineChart
                key={`${trainMetrics.algorithm}-${trainMetrics.episodes}-${trainMetrics.train_time_seconds}`}
                data={chartData}
                margin={{ top: 4, right: 4, left: -20, bottom: 0 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis
                  dataKey="episode"
                  tick={{ fontSize: 9 }}
                  tickFormatter={(v) => (v >= 1000 ? `${(v / 1000).toFixed(1)}k` : v)}
                />
                <YAxis tick={{ fontSize: 9 }} />
                <Tooltip
                  contentStyle={{ fontSize: 11 }}
                  formatter={(v: number) => [v.toFixed(1), "Reward"]}
                  labelFormatter={(l) => `Episode ${l}`}
                />
                <Line
                  type="monotone"
                  dataKey="value"
                  stroke="#6366f1"
                  dot={false}
                  strokeWidth={1.5}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* ── Step Info ── */}
      {hasStepInfo && (
        <div className="border border-gray-200 rounded-lg p-3 space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wide">
              Simulation
            </h3>
            <span className={`px-2 py-0.5 rounded-full text-xs font-semibold ${phaseColor}`}>
              {phase}
            </span>
          </div>

          {/* State tuple */}
          <div>
            <p className="text-xs text-gray-400 mb-1">Current state</p>
            <StateBadges state={stepInfo.currentState} />
          </div>

          {/* Action */}
          <div className="flex items-center gap-3">
            <div>
              <p className="text-xs text-gray-400 mb-1">Action</p>
              <span className="px-3 py-1 bg-gray-800 text-white text-sm font-mono rounded">
                {ACTION_LABELS[stepInfo.action] ?? stepInfo.actionName}
              </span>
            </div>
            <div>
              <p className="text-xs text-gray-400 mb-1">Reward</p>
              <RewardBadge reward={stepInfo.reward} />
            </div>
          </div>

          {/* Progress + cumulative */}
          <div className="grid grid-cols-2 gap-3 text-sm">
            <div className="bg-gray-50 rounded p-2 text-center">
              <p className="text-xs text-gray-400">Step</p>
              <p className="font-bold text-gray-800">
                {stepInfo.stepIndex + 1}
                <span className="font-normal text-gray-400"> / {stepInfo.totalSteps}</span>
              </p>
            </div>
            <div className="bg-gray-50 rounded p-2 text-center">
              <p className="text-xs text-gray-400">Total reward</p>
              <p
                className={`font-bold ${
                  stepInfo.cumulativeReward >= 0 ? "text-green-700" : "text-red-700"
                }`}
              >
                {stepInfo.cumulativeReward >= 0 ? "+" : ""}
                {stepInfo.cumulativeReward}
              </p>
            </div>
          </div>

          {/* Terminal banner */}
          {stepInfo.terminated && (
            <div className="bg-green-50 border border-green-200 rounded p-2 text-center">
              <p className="text-green-700 font-semibold text-sm">🎉 Episode Complete!</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
