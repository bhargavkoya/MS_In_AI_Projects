import { Car, MapPin, Flag, X } from "lucide-react";

// ── Types ─────────────────────────────────────────────────────────────────────

interface Position {
  row: number;
  col: number;
}

interface TaxiGridProps {
  // Grid dimensions
  rows: number;
  cols: number;

  // Setup-mode positions (used before simulation starts)
  taxiPos: Position | null;
  startPos: Position | null;    // passenger pickup
  destPos: Position | null;     // destination
  obstacles: Position[];

  // Cell click handler (disabled during simulation)
  onCellClick: (row: number, col: number) => void;
  isInteractive: boolean;       // false locks all clicks

  // BFS visualization (animated reveal)
  bfsExplored1: [number, number][];   // leg 1 explored nodes (yellow)
  bfsPath1: [number, number][];       // leg 1 optimal path  (orange)
  bfsExplored2: [number, number][];   // leg 2 explored nodes (yellow)
  bfsPath2: [number, number][];       // leg 2 optimal path  (orange)

  // RL simulation state
  rlCurrentPos: Position | null;      // live taxi position during playback
  rlPath: Position[];                 // cells visited so far
  rlPhase: "pickup" | "dropoff" | null;  // determines path color
  inTaxi: boolean;                    // true = passenger is aboard

  // Warning overlay (auto-clears from parent)
  warningMessage: string | null;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function posKey(r: number, c: number) {
  return `${r},${c}`;
}

function posSetFrom(arr: [number, number][]) {
  return new Set(arr.map(([r, c]) => posKey(r, c)));
}

function posMatch(pos: Position | null, row: number, col: number) {
  return pos !== null && pos.row === row && pos.col === col;
}

// ── Component ─────────────────────────────────────────────────────────────────

export function TaxiGrid({
  rows, cols,
  taxiPos, startPos, destPos, obstacles,
  onCellClick, isInteractive,
  bfsExplored1, bfsPath1, bfsExplored2, bfsPath2,
  rlCurrentPos, rlPath, rlPhase, inTaxi,
  warningMessage,
}: TaxiGridProps) {
  // Pre-build look-up sets for O(1) cell classification
  const obstacleSet  = new Set(obstacles.map((o) => posKey(o.row, o.col)));
  const rlPathSet    = new Set(rlPath.map((p) => posKey(p.row, p.col)));
  const bfsExp1Set   = posSetFrom(bfsExplored1);
  const bfsPath1Set  = posSetFrom(bfsPath1);
  const bfsExp2Set   = posSetFrom(bfsExplored2);
  const bfsPath2Set  = posSetFrom(bfsPath2);

  return (
    <div className="relative inline-block">
      {/* Grid */}
      <div
        className="inline-grid gap-0.5 p-3 bg-white rounded-xl shadow-lg border border-gray-100"
        style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }}
      >
        {Array.from({ length: rows * cols }).map((_, index) => {
          const row = Math.floor(index / cols);
          const col = index % cols;
          const key = posKey(row, col);

          // Booleans for this cell
          const isObstacle    = obstacleSet.has(key);
          const isCurrent     = posMatch(rlCurrentPos, row, col);
          const isRlPath      = rlPathSet.has(key) && !isCurrent;
          const isBfsPath1    = bfsPath1Set.has(key);
          const isBfsPath2    = bfsPath2Set.has(key);
          const isBfsExp      = (bfsExp1Set.has(key) || bfsExp2Set.has(key)) && !isBfsPath1 && !isBfsPath2;
          const isTaxiSetup   = posMatch(taxiPos, row, col) && !isCurrent;
          const isPassenger   = posMatch(startPos, row, col);
          const isDest        = posMatch(destPos, row, col);

          // ── Cell background via inline style (bypasses Tailwind CSS scanning) ──
          // Priority: obstacle > current > rl path > bfs path > bfs explored > default
          const cellStyle: React.CSSProperties = { backgroundColor: "#f9fafb", borderColor: "#e5e7eb" };

          if (isObstacle) {
            cellStyle.backgroundColor = "#1f2937"; // gray-800
            cellStyle.borderColor     = "#111827"; // gray-900
          } else if (isCurrent) {
            cellStyle.backgroundColor = "#4ade80"; // green-400
            cellStyle.borderColor     = "#22c55e"; // green-500
          } else if (isRlPath) {
            if (rlPhase === "dropoff") {
              cellStyle.backgroundColor = "#e9d5ff"; // purple-200
              cellStyle.borderColor     = "#c084fc"; // purple-400
            } else {
              cellStyle.backgroundColor = "#bfdbfe"; // blue-200
              cellStyle.borderColor     = "#60a5fa"; // blue-400
            }
          } else if (isBfsPath1 || isBfsPath2) {
            cellStyle.backgroundColor = "#fdba74"; // orange-300
            cellStyle.borderColor     = "#f97316"; // orange-500
          } else if (isBfsExp) {
            cellStyle.backgroundColor = "#fef9c3"; // yellow-100
            cellStyle.borderColor     = "#fde047"; // yellow-300
          }

          // ── Icon rendering (priority order) ───────────────────────────────
          let icon: React.ReactNode = null;

          if (isObstacle && !isCurrent) {
            icon = <X className="w-5 h-5 text-white opacity-80" strokeWidth={3} />;
          } else if (isCurrent) {
            icon = <Car className="w-6 h-6 text-white drop-shadow animate-pulse" />;
          } else if (isTaxiSetup) {
            icon = <Car className="w-6 h-6 text-yellow-600" />;
          } else if (isPassenger && !inTaxi) {
            icon = <MapPin className="w-5 h-5 text-green-600" />;
          } else if (isDest) {
            icon = <Flag className="w-5 h-5 text-red-600" />;
          }

          return (
            <button
              key={index}
              onClick={() => isInteractive && !isObstacle && onCellClick(row, col)}
              disabled={!isInteractive}
              title={`(${row}, ${col})`}
              style={cellStyle}
              className={`
                w-12 h-12 border rounded flex items-center justify-center
                relative transition-all duration-150
                ${isInteractive && !isObstacle ? "hover:brightness-95 cursor-pointer" : "cursor-default"}
                ${isCurrent ? "scale-110 z-10 shadow-md" : ""}
              `}
            >
              {icon}
              {/* Coordinate label */}
              <span className="absolute bottom-0.5 right-0.5 text-[8px] text-gray-400 leading-none font-mono pointer-events-none">
                {row},{col}
              </span>
            </button>
          );
        })}
      </div>

      {/* ── Warning overlay — blends into the grid ── */}
      {warningMessage && (
        <div className="absolute inset-0 flex items-end justify-center pb-3 pointer-events-none z-20">
          <div className="bg-black/65 backdrop-blur-sm text-white text-xs px-4 py-2.5 rounded-lg max-w-[90%] text-center shadow-lg leading-snug">
            {warningMessage}
          </div>
        </div>
      )}
    </div>
  );
}
