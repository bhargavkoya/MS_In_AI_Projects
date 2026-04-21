import {
  Car, MapPin, Flag, X, Play, RotateCcw, Trash2,
  Zap, Network, ChevronRight, Square, Loader2,
} from "lucide-react";
import type { Algorithm, TrainResponse } from "../services/api";

// ── Types ─────────────────────────────────────────────────────────────────────

type PlacementMode = "taxi" | "start" | "dest" | "obstacle";

interface ControlPanelProps {
  // Grid config
  rows: number;
  cols: number;
  placementMode: PlacementMode;
  onPlacementModeChange: (mode: PlacementMode) => void;
  onGridSizeChange: (rows: number, cols: number) => void;
  obstacleCount: number;
  onClearObstacles: () => void;

  // Initialization
  onInitialize: () => void;
  isInitialized: boolean;
  isLoading: boolean;

  // BFS
  onBfsCheck: () => void;

  // Algorithm & training
  algorithm: Algorithm;
  onAlgorithmChange: (alg: Algorithm) => void;
  episodeCount: number;
  onEpisodeCountChange: (n: number) => void;
  recommendedEpisodes: number;
  onTrain: () => void;
  isTrained: boolean;
  isTraining: boolean;
  trainMetrics: TrainResponse | null;

  // Simulation
  onStep: () => void;
  onAutoRun: () => void;
  onAutoRunStop: () => void;
  isAutoRunning: boolean;
  isSimulating: boolean;
  simDone: boolean;

  // Reset
  onReset: () => void;
}

// ── Section wrapper ───────────────────────────────────────────────────────────

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <p className="text-xs font-semibold text-gray-400 uppercase tracking-widest mb-2">
        {title}
      </p>
      {children}
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────

export function ControlPanel({
  rows, cols, placementMode, onPlacementModeChange,
  onGridSizeChange, obstacleCount, onClearObstacles,
  onInitialize, isInitialized, isLoading,
  onBfsCheck,
  algorithm, onAlgorithmChange, episodeCount, onEpisodeCountChange,
  recommendedEpisodes, onTrain, isTrained, isTraining, trainMetrics,
  onStep, onAutoRun, onAutoRunStop, isAutoRunning, isSimulating, simDone,
  onReset,
}: ControlPanelProps) {
  const busy = isLoading || isTraining || isSimulating || isAutoRunning;
  const canInteractWithGrid = !isInitialized && !busy;

  return (
    <div className="bg-white rounded-lg shadow-lg p-5 space-y-5 select-none">

      {/* ── 1. Grid Size ── */}
      <Section title="Grid Size">
        <div className="flex gap-3">
          <div className="flex-1">
            <label className="block text-xs text-gray-500 mb-1">Rows</label>
            <input
              type="number" min="2" max="20"
              value={rows}
              onChange={(e) => onGridSizeChange(parseInt(e.target.value) || 2, cols)}
              disabled={isInitialized || busy}
              className="w-full px-2 py-1.5 text-sm border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-400 disabled:bg-gray-50 disabled:text-gray-400"
            />
          </div>
          <div className="flex-1">
            <label className="block text-xs text-gray-500 mb-1">Cols</label>
            <input
              type="number" min="2" max="20"
              value={cols}
              onChange={(e) => onGridSizeChange(rows, parseInt(e.target.value) || 2)}
              disabled={isInitialized || busy}
              className="w-full px-2 py-1.5 text-sm border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-400 disabled:bg-gray-50 disabled:text-gray-400"
            />
          </div>
        </div>
      </Section>

      {/* ── 2. Placement Mode ── */}
      <Section title="Placement Mode">
        <div className="space-y-1.5">
          {([
            { mode: "taxi"     as PlacementMode, label: "Taxi Start",    Icon: Car,    active: "bg-yellow-500 text-white", hint: "yellow" },
            { mode: "start"    as PlacementMode, label: "Passenger",     Icon: MapPin, active: "bg-green-500 text-white",  hint: "green"  },
            { mode: "dest"     as PlacementMode, label: "Destination",   Icon: Flag,   active: "bg-red-500 text-white",    hint: "red"    },
            { mode: "obstacle" as PlacementMode, label: `Obstacles (${obstacleCount})`, Icon: X, active: "bg-gray-800 text-white", hint: "dark" },
          ] as const).map(({ mode, label, Icon, active }) => (
            <button
              key={mode}
              onClick={() => onPlacementModeChange(mode)}
              disabled={!canInteractWithGrid}
              className={`
                w-full px-3 py-2.5 rounded flex items-center gap-2.5 text-sm transition-all
                ${placementMode === mode ? active : "bg-gray-100 text-gray-700 hover:bg-gray-200"}
                ${!canInteractWithGrid ? "opacity-40 cursor-not-allowed" : ""}
              `}
            >
              <Icon className="w-4 h-4 flex-shrink-0" />
              <span>{label}</span>
            </button>
          ))}
        </div>
        {obstacleCount > 0 && canInteractWithGrid && (
          <button
            onClick={onClearObstacles}
            className="mt-1.5 w-full px-3 py-2 rounded flex items-center gap-2 text-sm bg-orange-50 text-orange-700 hover:bg-orange-100 border border-orange-200 transition-all"
          >
            <Trash2 className="w-4 h-4" />
            Clear obstacles
          </button>
        )}
      </Section>

      <div className="border-t border-gray-100" />

      {/* ── 3. Initialize ── */}
      <Section title="Step 1 — Initialize">
        <button
          onClick={onInitialize}
          disabled={busy}
          className={`
            w-full px-3 py-2.5 rounded flex items-center justify-center gap-2 text-sm font-semibold transition-all
            ${isInitialized
              ? "bg-emerald-50 text-emerald-700 border border-emerald-200"
              : "bg-blue-600 text-white hover:bg-blue-700 shadow-sm"}
            ${busy ? "opacity-50 cursor-not-allowed" : ""}
          `}
        >
          {isInitialized ? (
            <><span className="text-emerald-500">✓</span> Environment Ready</>
          ) : (
            <><Zap className="w-4 h-4" /> Initialize Environment</>
          )}
        </button>
        {isInitialized && (
          <p className="text-xs text-gray-400 mt-1 text-center">
            Click Reset to reconfigure
          </p>
        )}
      </Section>

      {/* ── 4. BFS Check ── */}
      <Section title="Step 2 — BFS Check">
        <button
          onClick={onBfsCheck}
          disabled={!isInitialized || busy}
          className={`
            w-full px-3 py-2.5 rounded flex items-center justify-center gap-2 text-sm font-semibold transition-all
            ${isInitialized && !busy
              ? "bg-amber-500 text-white hover:bg-amber-600 shadow-sm"
              : "bg-gray-100 text-gray-400 cursor-not-allowed"}
          `}
        >
          <Network className="w-4 h-4" />
          Run BFS Check
        </button>
        <p className="text-xs text-gray-400 mt-1">
          Visualises reachability paths on the grid
        </p>
      </Section>

      <div className="border-t border-gray-100" />

      {/* ── 5. Algorithm + Training ── */}
      <Section title="Step 3 — Train Agent">
        {/* Algorithm selector */}
        <div className="flex gap-2 mb-3">
          {(["qlearning", "montecarlo"] as Algorithm[]).map((alg) => (
            <button
              key={alg}
              onClick={() => onAlgorithmChange(alg)}
              disabled={!isInitialized || busy}
              className={`
                flex-1 py-2 rounded text-xs font-semibold border transition-all
                ${algorithm === alg
                  ? "bg-indigo-600 text-white border-indigo-600"
                  : "bg-white text-gray-600 border-gray-300 hover:border-indigo-400"}
                ${!isInitialized || busy ? "opacity-40 cursor-not-allowed" : ""}
              `}
            >
              {alg === "qlearning" ? "Q-Learning" : "Monte Carlo"}
            </button>
          ))}
        </div>

        {/* Episode count slider */}
        <div className="mb-3">
          <div className="flex justify-between items-center mb-1">
            <label className="text-xs text-gray-500">Episodes</label>
            <span className="text-xs font-mono text-gray-700">
              {episodeCount.toLocaleString()}
              {episodeCount === recommendedEpisodes && (
                <span className="ml-1 text-blue-500">(rec.)</span>
              )}
            </span>
          </div>
          <input
            type="range"
            min={500}
            max={10000}
            step={500}
            value={episodeCount}
            onChange={(e) => onEpisodeCountChange(Number(e.target.value))}
            disabled={!isInitialized || busy}
            className="w-full accent-indigo-600 disabled:opacity-40"
          />
          <div className="flex justify-between text-[10px] text-gray-400 mt-0.5">
            <span>500</span>
            <span>10,000</span>
          </div>
        </div>

        {/* Train button */}
        <button
          onClick={onTrain}
          disabled={!isInitialized || busy}
          className={`
            w-full px-3 py-2.5 rounded flex items-center justify-center gap-2 text-sm font-semibold transition-all
            ${isInitialized && !busy
              ? "bg-indigo-600 text-white hover:bg-indigo-700 shadow-sm"
              : "bg-gray-100 text-gray-400 cursor-not-allowed"}
          `}
        >
          {isTraining ? (
            <><Loader2 className="w-4 h-4 animate-spin" /> Training…</>
          ) : isTrained ? (
            <><span className="text-green-300">✓</span> Re-Train</>
          ) : (
            <><Play className="w-4 h-4" /> Train</>
          )}
        </button>

        {/* Training result summary */}
        {trainMetrics && (
          <div className="mt-2 flex items-center justify-between text-xs text-gray-500 bg-gray-50 rounded px-2 py-1.5">
            <span>Success rate</span>
            <span className={`font-bold ${
              trainMetrics.final_success_rate >= 0.7
                ? "text-green-600"
                : trainMetrics.final_success_rate >= 0.4
                ? "text-yellow-600"
                : "text-red-600"
            }`}>
              {Math.round(trainMetrics.final_success_rate * 100)}%
            </span>
          </div>
        )}
      </Section>

      <div className="border-t border-gray-100" />

      {/* ── 6. Simulation Controls ── */}
      <Section title="Step 4 — Simulate">
        <div className="space-y-1.5">
          {/* Step button */}
          <button
            onClick={onStep}
            disabled={!isTrained || busy || simDone}
            className={`
              w-full px-3 py-2.5 rounded flex items-center justify-center gap-2 text-sm font-semibold transition-all
              ${isTrained && !busy && !simDone
                ? "bg-blue-600 text-white hover:bg-blue-700 shadow-sm"
                : "bg-gray-100 text-gray-400 cursor-not-allowed"}
            `}
          >
            <ChevronRight className="w-4 h-4" />
            Step
          </button>

          {/* Auto-run / Stop */}
          {isAutoRunning ? (
            <button
              onClick={onAutoRunStop}
              className="w-full px-3 py-2.5 rounded flex items-center justify-center gap-2 text-sm font-semibold bg-red-500 text-white hover:bg-red-600 shadow-sm transition-all"
            >
              <Square className="w-4 h-4" />
              Stop
            </button>
          ) : (
            <button
              onClick={onAutoRun}
              disabled={!isTrained || busy || simDone}
              className={`
                w-full px-3 py-2.5 rounded flex items-center justify-center gap-2 text-sm font-semibold transition-all
                ${isTrained && !busy && !simDone
                  ? "bg-emerald-600 text-white hover:bg-emerald-700 shadow-sm"
                  : "bg-gray-100 text-gray-400 cursor-not-allowed"}
              `}
            >
              <Play className="w-4 h-4" />
              Auto-Run
            </button>
          )}

          {simDone && (
            <button
              onClick={onStep}
              disabled={true}
              className="w-full px-2 py-1.5 rounded text-xs text-gray-400 bg-gray-50 border border-gray-200 text-center"
            >
              Episode finished — click Reset or Re-Train to replay
            </button>
          )}
        </div>
      </Section>

      <div className="border-t border-gray-100" />

      {/* ── 7. Reset ── */}
      <button
        onClick={onReset}
        disabled={isTraining || isAutoRunning}
        className={`
          w-full px-3 py-2.5 rounded flex items-center justify-center gap-2 text-sm font-semibold transition-all
          bg-gray-600 text-white hover:bg-gray-700
          ${isTraining || isAutoRunning ? "opacity-40 cursor-not-allowed" : ""}
        `}
      >
        <RotateCcw className="w-4 h-4" />
        Reset
      </button>

      {/* ── Legend ── */}
      <div className="pt-2 border-t border-gray-100">
        <p className="text-xs font-semibold text-gray-400 uppercase tracking-widest mb-2">
          Legend
        </p>
        <div className="space-y-1 text-xs text-gray-600">
          {[
            { el: <Car className="w-3.5 h-3.5 text-yellow-600" />,                                                      label: "Taxi (start)" },
            { el: <MapPin className="w-3.5 h-3.5 text-green-600" />,                                                    label: "Passenger (pickup)" },
            { el: <Flag className="w-3.5 h-3.5 text-red-600" />,                                                        label: "Destination (dropoff)" },
            { el: <div className="w-3.5 h-3.5 bg-green-400 rounded-sm" />,                                              label: "Taxi in motion" },
            { el: <div className="w-3.5 h-3.5 bg-blue-200 border border-blue-400 rounded-sm" />,                        label: "RL path (pickup leg)" },
            { el: <div className="w-3.5 h-3.5 bg-purple-200 border border-purple-400 rounded-sm" />,                    label: "RL path (dropoff leg)" },
            { el: <div className="w-3.5 h-3.5 bg-orange-300 border border-orange-500 rounded-sm" />,                    label: "BFS shortest path" },
            { el: <div className="w-3.5 h-3.5 bg-yellow-100 border border-yellow-300 rounded-sm" />,                    label: "BFS explored" },
            { el: <div className="w-3.5 h-3.5 bg-gray-800 rounded-sm flex items-center justify-center"><X className="w-2.5 h-2.5 text-white" strokeWidth={3} /></div>, label: "Obstacle" },
          ].map(({ el, label }, i) => (
            <div key={i} className="flex items-center gap-2">
              <div className="flex-shrink-0">{el}</div>
              <span>{label}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
