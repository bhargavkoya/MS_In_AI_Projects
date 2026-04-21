import { useState, useRef, useCallback, useEffect } from "react";
import { TaxiGrid } from "./components/TaxiGrid";
import { ControlPanel } from "./components/ControlPanel";
import { InfoPanel, type StepInfo } from "./components/InfoPanel";
import { api, type Algorithm, type RLState, type BfsCheckResponse, type TrainResponse } from "./services/api";

// ── Types ─────────────────────────────────────────────────────────────────────

type PlacementMode = "taxi" | "start" | "dest" | "obstacle";

interface Position {
  row: number;
  col: number;
}

// ── Constants ─────────────────────────────────────────────────────────────────

const AUTO_RUN_INTERVAL_MS = 450;
const WARNING_DURATION_MS  = 4000;
const BFS_EXPLORE_INTERVAL = 12; // ms per explored-cell reveal

// ── App ───────────────────────────────────────────────────────────────────────

export default function App() {
  // ── Grid setup state ────────────────────────────────────────────────────────
  const [rows, setRows] = useState(7);
  const [cols, setCols] = useState(7);
  const [placementMode, setPlacementMode] = useState<PlacementMode>("taxi");
  const [taxiPos,  setTaxiPos]  = useState<Position | null>(null);
  const [startPos, setStartPos] = useState<Position | null>(null);
  const [destPos,  setDestPos]  = useState<Position | null>(null);
  const [obstacles, setObstacles] = useState<Position[]>([]);

  // ── Backend / phase state ───────────────────────────────────────────────────
  const [isInitialized, setIsInitialized] = useState(false);
  const [isLoading,     setIsLoading]     = useState(false);
  const [isTraining,    setIsTraining]    = useState(false);
  const [warningMsg,    setWarningMsg]    = useState<string | null>(null);

  // ── Backend-resolved positions (may differ from setup if auto-placed) ───────
  const [resolvedTaxi, setResolvedTaxi]   = useState<Position | null>(null);
  const [resolvedPass, setResolvedPass]   = useState<Position | null>(null);
  const [resolvedDest, setResolvedDest]   = useState<Position | null>(null);
  const [recEpisodes,  setRecEpisodes]    = useState(5000);

  // ── BFS state ───────────────────────────────────────────────────────────────
  const [bfsResult,    setBfsResult]    = useState<BfsCheckResponse | null>(null);
  const [bfsAnimIndex, setBfsAnimIndex] = useState(0);

  // ── Algorithm & training ────────────────────────────────────────────────────
  const [algorithm,    setAlgorithm]    = useState<Algorithm>("qlearning");
  const [episodeCount, setEpisodeCount] = useState(5000);
  const [isTrained,    setIsTrained]    = useState(false);
  const [trainMetrics, setTrainMetrics] = useState<TrainResponse | null>(null);

  // ── Simulation (step / auto-run) ────────────────────────────────────────────
  const [simPath,       setSimPath]       = useState<RLState[] | null>(null);
  const [simIndex,      setSimIndex]      = useState(0);
  const [stepInfo,      setStepInfo]      = useState<StepInfo | null>(null);
  const [isAutoRunning, setIsAutoRunning] = useState(false);
  const [simDone,       setSimDone]       = useState(false);
  const [cumReward,     setCumReward]     = useState(0);

  // Ref for auto-run interval
  const autoRunRef  = useRef<ReturnType<typeof setInterval> | null>(null);
  // Ref for warning auto-clear
  const warnTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Ref for BFS animation interval
  const bfsTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // ── Warning helper ──────────────────────────────────────────────────────────
  const showWarning = useCallback((msg: string) => {
    setWarningMsg(msg);
    if (warnTimerRef.current) clearTimeout(warnTimerRef.current);
    warnTimerRef.current = setTimeout(() => setWarningMsg(null), WARNING_DURATION_MS);
  }, []);

  // ── Derived BFS slice (animated reveal) ────────────────────────────────────
  const leg1ExpLen = bfsResult?.leg1.explored.length ?? 0;
  const leg2ExpLen = bfsResult?.leg2.explored.length ?? 0;
  const totalExplored = leg1ExpLen + leg2ExpLen;

  const bfsExplored1 = bfsResult
    ? (bfsResult.leg1.explored.slice(0, Math.min(bfsAnimIndex, leg1ExpLen)) as [number,number][])
    : [];
  const bfsExplored2 = bfsResult
    ? (bfsResult.leg2.explored.slice(0, Math.max(0, bfsAnimIndex - leg1ExpLen)) as [number,number][])
    : [];
  const bfsPath1 = (bfsResult && bfsAnimIndex >= leg1ExpLen)
    ? (bfsResult.leg1.path as [number,number][])
    : [];
  const bfsPath2 = (bfsResult && bfsAnimIndex >= totalExplored)
    ? (bfsResult.leg2.path as [number,number][])
    : [];

  // ── Derived RL display values ────────────────────────────────────────────────
  const currentRLState: RLState | null = simPath ? simPath[simIndex] : null;
  const rlCurrentPos: Position | null = currentRLState
    ? { row: currentRLState[0], col: currentRLState[1] }
    : null;

  const rlPathPositions: Position[] = simPath
    ? simPath.slice(0, simIndex + 1).map((s) => ({ row: s[0], col: s[1] }))
    : [];

  const rlPhase: "pickup" | "dropoff" | null = currentRLState
    ? (currentRLState[6] === 1 ? "dropoff" : "pickup")
    : null;

  const inTaxi = currentRLState ? currentRLState[6] === 1 : false;

  // Passenger position shown on grid (follows taxi when in_taxi=1)
  const displayPassPos: Position | null = currentRLState
    ? (currentRLState[6] === 1
        ? { row: currentRLState[0], col: currentRLState[1] }
        : { row: currentRLState[2], col: currentRLState[3] })
    : resolvedPass;

  const displayDestPos: Position | null = currentRLState
    ? { row: currentRLState[4], col: currentRLState[5] }
    : resolvedDest;

  // ── Cell click (setup mode) ─────────────────────────────────────────────────
  const handleCellClick = (row: number, col: number) => {
    const newPos = { row, col };
    const isObs  = obstacles.some((o) => o.row === row && o.col === col);

    switch (placementMode) {
      case "taxi":
        setObstacles((prev) => prev.filter((o) => !(o.row === row && o.col === col)));
        setTaxiPos(newPos);
        break;
      case "start":
        setObstacles((prev) => prev.filter((o) => !(o.row === row && o.col === col)));
        setStartPos(newPos);
        break;
      case "dest":
        setObstacles((prev) => prev.filter((o) => !(o.row === row && o.col === col)));
        setDestPos(newPos);
        break;
      case "obstacle": {
        const isTaxi  = taxiPos  && taxiPos.row  === row && taxiPos.col  === col;
        const isStart = startPos && startPos.row === row && startPos.col === col;
        const isDest  = destPos  && destPos.row  === row && destPos.col  === col;
        if (isTaxi || isStart || isDest) return; // guard key positions
        if (isObs) {
          setObstacles((prev) => prev.filter((o) => !(o.row === row && o.col === col)));
        } else {
          setObstacles((prev) => [...prev, newPos]);
        }
        break;
      }
    }
  };

  // ── Grid resize ─────────────────────────────────────────────────────────────
  const handleGridSizeChange = (newRows: number, newCols: number) => {
    const r = Math.max(2, Math.min(20, newRows));
    const c = Math.max(2, Math.min(20, newCols));
    setRows(r); setCols(c);
    if (taxiPos  && (taxiPos.row  >= r || taxiPos.col  >= c)) setTaxiPos(null);
    if (startPos && (startPos.row >= r || startPos.col >= c)) setStartPos(null);
    if (destPos  && (destPos.row  >= r || destPos.col  >= c)) setDestPos(null);
    setObstacles((prev) => prev.filter((o) => o.row < r && o.col < c));
  };

  // ── Initialize ──────────────────────────────────────────────────────────────
  const handleInitialize = async () => {
    setIsLoading(true);
    setWarningMsg(null);
    try {
      const res = await api.initialize({
        rows, cols,
        obstacles: obstacles.map((o) => [o.row, o.col] as [number, number]),
        taxi:        taxiPos  ? [taxiPos.row,  taxiPos.col]  : null,
        passenger:   startPos ? [startPos.row, startPos.col] : null,
        destination: destPos  ? [destPos.row,  destPos.col]  : null,
      });

      setResolvedTaxi({ row: res.taxi[0],        col: res.taxi[1] });
      setResolvedPass({ row: res.passenger[0],   col: res.passenger[1] });
      setResolvedDest({ row: res.destination[0], col: res.destination[1] });
      // Sync displayed positions back (important when positions were auto-placed)
      setTaxiPos({ row: res.taxi[0],        col: res.taxi[1] });
      setStartPos({ row: res.passenger[0],  col: res.passenger[1] });
      setDestPos({ row: res.destination[0], col: res.destination[1] });

      const rec = Math.min(res.recommended_episodes, 10000);
      setRecEpisodes(rec);
      setEpisodeCount(rec);
      setIsInitialized(true);
      setIsTrained(false);
      setTrainMetrics(null);
      setBfsResult(null);
      setBfsAnimIndex(0);
      resetSim();
    } catch (e: unknown) {
      showWarning((e as Error).message ?? "Initialization failed.");
    } finally {
      setIsLoading(false);
    }
  };

  // ── BFS Check ───────────────────────────────────────────────────────────────
  const handleBfsCheck = async () => {
    setIsLoading(true);
    try {
      const res = await api.bfsCheck();
      setBfsResult(res);
      setBfsAnimIndex(0);

      // Animate explored cells
      if (bfsTimerRef.current) clearInterval(bfsTimerRef.current);
      let i = 0;
      const total = (res.leg1.explored.length) + (res.leg2.explored.length);
      bfsTimerRef.current = setInterval(() => {
        i += 2; // reveal 2 cells per tick for snappier animation
        setBfsAnimIndex(i);
        if (i >= total) {
          clearInterval(bfsTimerRef.current!);
          setBfsAnimIndex(total + 1); // ensure paths are shown
        }
      }, BFS_EXPLORE_INTERVAL);

      if (!res.both_reachable) {
        const msg = [
          !res.leg1.reachable ? "Taxi cannot reach the passenger!" : null,
          !res.leg2.reachable ? "Passenger cannot reach the destination!" : null,
        ].filter(Boolean).join(" ");
        showWarning(msg);
      }
    } catch (e: unknown) {
      showWarning((e as Error).message ?? "BFS check failed.");
    } finally {
      setIsLoading(false);
    }
  };

  // ── Train ────────────────────────────────────────────────────────────────────
  const handleTrain = async () => {
    setIsTraining(true);
    setIsTrained(false);
    setTrainMetrics(null);
    resetSim();
    try {
      const res = await api.train({ algorithm, episodes: episodeCount });
      setTrainMetrics(res);
      setIsTrained(true);
    } catch (e: unknown) {
      showWarning((e as Error).message ?? "Training failed.");
    } finally {
      setIsTraining(false);
    }
  };

  // ── Simulation helpers ───────────────────────────────────────────────────────
  const resetSim = () => {
    if (autoRunRef.current) clearInterval(autoRunRef.current);
    setSimPath(null);
    setSimIndex(0);
    setStepInfo(null);
    setIsAutoRunning(false);
    setSimDone(false);
    setCumReward(0);
  };

  const applyStep = useCallback(
    (
      currentState: RLState,
      action: number,
      actionName: string,
      nextState: RLState,
      reward: number,
      terminated: boolean,
      stepIdx: number,
      total: number,
      running_reward: number,
    ) => {
      const newCum = running_reward + reward;
      setCumReward(newCum);
      setStepInfo({
        stepIndex: stepIdx,
        totalSteps: total,
        currentState,
        action,
        actionName,
        reward,
        cumulativeReward: newCum,
        terminated,
      });
      if (terminated) {
        setSimDone(true);
      }
    },
    [],
  );

  // ── Step (one step at a time) ────────────────────────────────────────────────
  const handleStep = async () => {
    // Clear BFS overlay on first step so it doesn't interfere with path colours
    if (!simPath) {
      setBfsResult(null);
      setBfsAnimIndex(0);
    }
    const currentState: RLState = simPath
      ? simPath[simIndex]
      : [
          resolvedTaxi!.row, resolvedTaxi!.col,
          resolvedPass!.row, resolvedPass!.col,
          resolvedDest!.row, resolvedDest!.col,
          0,
        ];

    try {
      const res = await api.step({ algorithm, state: currentState });

      // Build / extend the path
      setSimPath((prev) => {
        if (!prev) return [currentState, res.next_state];
        return [...prev, res.next_state];
      });
      setSimIndex((prev) => prev + 1);

      applyStep(
        currentState,
        res.action,
        res.action_name,
        res.next_state,
        res.reward,
        res.terminated,
        simIndex,              // will be incremented above
        simIndex + 1,          // we don't know total in step mode
        cumReward,
      );
    } catch (e: unknown) {
      showWarning((e as Error).message ?? "Step failed.");
    }
  };

  // ── Auto-Run ─────────────────────────────────────────────────────────────────
  const handleAutoRun = async () => {
    setIsLoading(true);
    resetSim();
    // Clear BFS overlay so it doesn't bleed through the RL path trail
    setBfsResult(null);
    setBfsAnimIndex(0);
    try {
      const res = await api.simulate({ algorithm });
      const path = res.path;
      setSimPath(path);
      setSimIndex(0);
      setIsAutoRunning(true);

      let idx = 0;
      let runningReward = 0;

      autoRunRef.current = setInterval(() => {
        if (idx >= path.length - 1) {
          clearInterval(autoRunRef.current!);
          setIsAutoRunning(false);
          return;
        }

        const curState  = path[idx];
        const nxtState  = path[idx + 1];
        // Reconstruct reward from state transition (action names not in path)
        // We infer reward: +20 on final step (done), +10 on pickup, -10 on bad actions, -1 otherwise
        const done      = idx === path.length - 2 && res.success;
        // Use a simplified reward heuristic for display (actual values differ slightly)
        const inTaxiBefore = curState[6] === 0 && nxtState[6] === 1;
        const reward = done ? 20 : inTaxiBefore ? 10 : -1;
        runningReward += reward;

        setSimIndex(idx + 1);
        setStepInfo({
          stepIndex: idx,
          totalSteps: path.length - 1,
          currentState: curState,
          action: 0,           // action not available from /simulate
          actionName: "—",
          reward,
          cumulativeReward: runningReward,
          terminated: done,
        });

        if (done) setSimDone(true);
        idx++;
      }, AUTO_RUN_INTERVAL_MS);

      if (!res.success) {
        showWarning("Agent could not complete the trip within the step limit.");
      }
    } catch (e: unknown) {
      showWarning((e as Error).message ?? "Simulation failed.");
      setIsAutoRunning(false);
    } finally {
      setIsLoading(false);
    }
  };

  const handleAutoRunStop = () => {
    if (autoRunRef.current) clearInterval(autoRunRef.current);
    setIsAutoRunning(false);
  };

  // ── Full Reset ───────────────────────────────────────────────────────────────
  const handleReset = async () => {
    handleAutoRunStop();
    if (bfsTimerRef.current) clearInterval(bfsTimerRef.current);
    if (warnTimerRef.current) clearTimeout(warnTimerRef.current);
    resetSim();
    try { await api.reset(); } catch { /* ignore */ }
    // Grid config — back to blank 5×5
    setRows(5);
    setCols(5);
    setTaxiPos(null);
    setStartPos(null);
    setDestPos(null);
    setObstacles([]);
    setPlacementMode("taxi");
    // Backend / training state
    setIsInitialized(false);
    setIsTrained(false);
    setTrainMetrics(null);
    setBfsResult(null);
    setBfsAnimIndex(0);
    setWarningMsg(null);
    setResolvedTaxi(null);
    setResolvedPass(null);
    setResolvedDest(null);
  };

  // ── Cleanup on unmount ───────────────────────────────────────────────────────
  useEffect(() => {
    return () => {
      if (autoRunRef.current)  clearInterval(autoRunRef.current);
      if (bfsTimerRef.current) clearInterval(bfsTimerRef.current);
      if (warnTimerRef.current) clearTimeout(warnTimerRef.current);
    };
  }, []);

  // ── Display positions for the grid ──────────────────────────────────────────
  // During simulation, use resolved / RL positions; during setup, use user-placed
  const gridTaxiPos  = isInitialized ? resolvedTaxi  : taxiPos;
  const gridStartPos = isInitialized ? displayPassPos : startPos;
  const gridDestPos  = isInitialized ? displayDestPos : destPos;

  const isGridInteractive = !isInitialized && !isLoading && !isTraining;

  // ── BFS summary for InfoPanel ────────────────────────────────────────────────
  const bfsReachable = bfsResult
    ? { leg1: bfsResult.leg1.reachable, leg2: bfsResult.leg2.reachable }
    : null;
  const bfsPathLengths = bfsResult
    ? { leg1: bfsResult.leg1.path.length, leg2: bfsResult.leg2.path.length }
    : null;

  // ── Render ───────────────────────────────────────────────────────────────────
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-100 to-gray-200 overflow-auto">
      <div className="max-w-[1400px] mx-auto px-6 py-8">

        {/* Header */}
        <div className="mb-8 text-center">
          <h1 className="text-4xl font-bold text-gray-800 mb-1 tracking-tight">
            Taxi RL Simulation
          </h1>
          <p className="text-gray-500 text-sm">
            BFS · Q-Learning · Monte Carlo — configure, train, and visualise live
          </p>
        </div>

        {/* Main layout: grid centre + sidebar right */}
        <div className="flex flex-col xl:flex-row gap-6 items-start justify-center">

          {/* Left: Grid + Info Panel stacked */}
          <div className="flex flex-col gap-4 items-center flex-shrink-0">
            <TaxiGrid
              rows={rows}
              cols={cols}
              taxiPos={gridTaxiPos}
              startPos={gridStartPos}
              destPos={gridDestPos}
              obstacles={obstacles}
              onCellClick={handleCellClick}
              isInteractive={isGridInteractive}
              bfsExplored1={bfsExplored1}
              bfsPath1={bfsPath1}
              bfsExplored2={bfsExplored2}
              bfsPath2={bfsPath2}
              rlCurrentPos={rlCurrentPos}
              rlPath={rlPathPositions}
              rlPhase={rlPhase}
              inTaxi={inTaxi}
              warningMessage={warningMsg}
            />

            {/* Info panel below the grid */}
            <div className="w-full max-w-[576px]">
              <InfoPanel
                stepInfo={stepInfo}
                trainMetrics={trainMetrics}
                bfsReachable={bfsReachable}
                bfsPathLengths={bfsPathLengths}
              />
            </div>
          </div>

          {/* Right: Control Panel */}
          <div className="w-full xl:w-80 flex-shrink-0">
            <ControlPanel
              rows={rows}
              cols={cols}
              placementMode={placementMode}
              onPlacementModeChange={setPlacementMode}
              onGridSizeChange={handleGridSizeChange}
              obstacleCount={obstacles.length}
              onClearObstacles={() => setObstacles([])}
              onInitialize={handleInitialize}
              isInitialized={isInitialized}
              isLoading={isLoading}
              onBfsCheck={handleBfsCheck}
              algorithm={algorithm}
              onAlgorithmChange={(alg) => {
                setAlgorithm(alg);
                setIsTrained(false);
                setTrainMetrics(null);
                resetSim();
              }}
              episodeCount={episodeCount}
              onEpisodeCountChange={setEpisodeCount}
              recommendedEpisodes={recEpisodes}
              onTrain={handleTrain}
              isTrained={isTrained}
              isTraining={isTraining}
              trainMetrics={trainMetrics}
              onStep={handleStep}
              onAutoRun={handleAutoRun}
              onAutoRunStop={handleAutoRunStop}
              isAutoRunning={isAutoRunning}
              isSimulating={isLoading}
              simDone={simDone}
              onReset={handleReset}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
