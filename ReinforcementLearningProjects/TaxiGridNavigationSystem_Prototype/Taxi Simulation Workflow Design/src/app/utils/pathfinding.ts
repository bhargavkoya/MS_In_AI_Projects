// Simple A* pathfinding algorithm for demonstration
// You can replace this with your RL model's output later

interface Position {
  row: number;
  col: number;
}

interface Node extends Position {
  g: number; // cost from start
  h: number; // heuristic cost to goal
  f: number; // total cost
  parent: Node | null;
}

function heuristic(a: Position, b: Position): number {
  // Manhattan distance
  return Math.abs(a.row - b.row) + Math.abs(a.col - b.col);
}

function getNeighbors(
  node: Position,
  rows: number,
  cols: number,
  obstacles: Position[]
): Position[] {
  const neighbors: Position[] = [];
  const directions = [
    { row: -1, col: 0 }, // up
    { row: 1, col: 0 },  // down
    { row: 0, col: -1 }, // left
    { row: 0, col: 1 },  // right
  ];

  for (const dir of directions) {
    const newRow = node.row + dir.row;
    const newCol = node.col + dir.col;

    // Check bounds
    if (newRow >= 0 && newRow < rows && newCol >= 0 && newCol < cols) {
      // Check if not an obstacle
      const isObstacle = obstacles.some(
        (o) => o.row === newRow && o.col === newCol
      );
      if (!isObstacle) {
        neighbors.push({ row: newRow, col: newCol });
      }
    }
  }

  return neighbors;
}

function positionsEqual(a: Position, b: Position): boolean {
  return a.row === b.row && a.col === b.col;
}

interface PathResult {
  path: Position[];
  complete: boolean;
  message?: string;
}

export function findPath(
  start: Position,
  goal: Position,
  rows: number,
  cols: number,
  obstacles: Position[]
): PathResult {
  const openSet: Node[] = [];
  const closedSet: Set<string> = new Set();
  let bestNode: Node | null = null;

  const startNode: Node = {
    ...start,
    g: 0,
    h: heuristic(start, goal),
    f: heuristic(start, goal),
    parent: null,
  };

  openSet.push(startNode);
  bestNode = startNode;

  while (openSet.length > 0) {
    // Find node with lowest f score
    openSet.sort((a, b) => a.f - b.f);
    const current = openSet.shift()!;

    // Track the node closest to the goal
    if (!bestNode || current.h < bestNode.h) {
      bestNode = current;
    }

    // Check if we reached the goal
    if (positionsEqual(current, goal)) {
      // Reconstruct path
      const path: Position[] = [];
      let node: Node | null = current;
      while (node) {
        path.unshift({ row: node.row, col: node.col });
        node = node.parent;
      }
      return { path, complete: true };
    }

    closedSet.add(`${current.row},${current.col}`);

    // Check all neighbors
    const neighbors = getNeighbors(current, rows, cols, obstacles);
    for (const neighborPos of neighbors) {
      const key = `${neighborPos.row},${neighborPos.col}`;
      if (closedSet.has(key)) continue;

      const g = current.g + 1;
      const h = heuristic(neighborPos, goal);
      const f = g + h;

      // Check if neighbor is already in open set
      const existingIndex = openSet.findIndex((n) =>
        positionsEqual(n, neighborPos)
      );

      if (existingIndex === -1) {
        // Add new node
        openSet.push({
          ...neighborPos,
          g,
          h,
          f,
          parent: current,
        });
      } else if (g < openSet[existingIndex].g) {
        // Update existing node if this path is better
        openSet[existingIndex].g = g;
        openSet[existingIndex].f = f;
        openSet[existingIndex].parent = current;
      }
    }
  }

  // No complete path found - return partial path to closest point
  if (bestNode && !positionsEqual(bestNode, start)) {
    const partialPath: Position[] = [];
    let node: Node | null = bestNode;
    while (node) {
      partialPath.unshift({ row: node.row, col: node.col });
      node = node.parent;
    }
    return {
      path: partialPath,
      complete: false,
      message: `Cannot reach destination. Stopped at closest point (${bestNode.row}, ${bestNode.col})`
    };
  }

  // Completely blocked
  return {
    path: [start],
    complete: false,
    message: 'Taxi is completely blocked by obstacles!'
  };
}

// Generate complete path including taxi to start, then start to destination
export function generateCompletePath(
  taxiPos: Position,
  startPos: Position,
  destPos: Position,
  rows: number,
  cols: number,
  obstacles: Position[]
): { path: Position[]; complete: boolean; message?: string } {
  // First: taxi goes to start (pick-up)
  const pathToStartResult = findPath(taxiPos, startPos, rows, cols, obstacles);

  if (!pathToStartResult.complete) {
    return {
      path: pathToStartResult.path,
      complete: false,
      message: `Taxi cannot reach passenger! ${pathToStartResult.message}`
    };
  }

  // Then: taxi goes from start to destination (drop-off)
  const pathToDestResult = findPath(startPos, destPos, rows, cols, obstacles);

  if (!pathToDestResult.complete) {
    return {
      path: [...pathToStartResult.path, ...pathToDestResult.path.slice(1)],
      complete: false,
      message: `Passenger cannot reach destination! ${pathToDestResult.message}`
    };
  }

  // Combine paths, removing duplicate start position
  const combinedPath = [...pathToStartResult.path, ...pathToDestResult.path.slice(1)];

  return {
    path: combinedPath,
    complete: true
  };
}
