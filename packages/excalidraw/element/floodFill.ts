import { pointFrom, type LocalPoint } from "@excalidraw/math";

import { exportToCanvas } from "../scene/export";
import { getRootElements } from "../frame";

import { getCommonBounds } from "./bounds";

import type { NonDeletedExcalidrawElement } from "./types";
import type { AppState, BinaryFiles } from "../types";

/**
 * Paint-bucket flood fill.
 *
 * The tool rasterizes the current strokes, morphologically DILATES the ink to
 * bridge the small gaps between separate pencil strokes, flood-fills the region
 * under the click, and traces the result into a closed polygon. The polygon is
 * inserted as a `line` element with a solid background — the one shape
 * Excalidraw exports as a solid `<path fill-rule="evenodd">` (see
 * `scene/Shape.ts` `isPathALoop` + `renderer/staticSvgScene.ts`), so it renders
 * live AND survives the app's SVG flatten.
 *
 * The pure core (`maskToFillContour`, `dilateBinary`) is DOM-free and unit
 * tested in `floodFill.test.ts`. The async wrapper (`computeFloodFillContour`)
 * owns the raster + coordinate mapping.
 */

// --- tuning knobs (raster pixels unless noted) ---
const RASTER_CAP = 2048; // max raster dimension; caps the O(n) passes at ~4.2M px
// ponytail: r=6 is THE knob. Bridges gaps up to ~2r px between disconnected
// strokes (the whole point of the tool). Bigger = fills sloppier loops but can
// seal a real opening to the outside; smaller = stricter. Tune after real use.
const DILATION_RADIUS = 6;
const ALPHA_THRESHOLD = 16; // ink vs. empty on the transparent-background raster
const RDP_EPSILON = 1.5; // contour simplification tolerance
const MIN_FILL_PIXELS = 64; // ignore noise pockets

const clampInt = (v: number, lo: number, hi: number) =>
  v < lo ? lo : v > hi ? hi : v;

/**
 * Separable O(n) dilation by a `(2r+1)×(2r+1)` square (Chebyshev radius `r`).
 * Horizontal distance-threshold pass, then vertical — the square structuring
 * element is the Minkowski sum of the two 1-D segments. Exported for testing.
 */
export const dilateBinary = (
  src: Uint8Array,
  w: number,
  h: number,
  r: number,
): Uint8Array => {
  const INF = w + h + 1;
  const tmp = new Uint8Array(w * h);
  const distL = new Int32Array(w);

  for (let y = 0; y < h; y++) {
    const base = y * w;
    let d = INF;
    for (let x = 0; x < w; x++) {
      d = src[base + x] ? 0 : d + 1;
      distL[x] = d;
    }
    d = INF;
    for (let x = w - 1; x >= 0; x--) {
      d = src[base + x] ? 0 : d + 1;
      tmp[base + x] = Math.min(distL[x], d) <= r ? 1 : 0;
    }
  }

  const out = new Uint8Array(w * h);
  const distU = new Int32Array(h);
  for (let x = 0; x < w; x++) {
    let d = INF;
    for (let y = 0; y < h; y++) {
      const i = y * w + x;
      d = tmp[i] ? 0 : d + 1;
      distU[y] = d;
    }
    d = INF;
    for (let y = h - 1; y >= 0; y--) {
      const i = y * w + x;
      d = tmp[i] ? 0 : d + 1;
      out[i] = Math.min(distU[y], d) <= r ? 1 : 0;
    }
  }
  return out;
};

/**
 * 4-connected stack flood fill. `reachedBorder` doubles as the enclosure test:
 * the caller pads the raster so a truly enclosed region can never touch the
 * edge, hence "reached border" ⇔ "open to the outside".
 */
const floodRegion = (
  w: number,
  h: number,
  sx: number,
  sy: number,
  passable: (i: number) => boolean,
): { mask: Uint8Array; reachedBorder: boolean; count: number } => {
  const mask = new Uint8Array(w * h);
  const stack = new Int32Array(w * h);
  let sp = 0;
  const start = sy * w + sx;
  if (!passable(start)) {
    return { mask, reachedBorder: false, count: 0 };
  }
  stack[sp++] = start;
  mask[start] = 1;
  let count = 0;
  let reachedBorder = false;
  while (sp > 0) {
    const i = stack[--sp];
    count++;
    const x = i % w;
    const y = (i / w) | 0;
    if (x === 0 || y === 0 || x === w - 1 || y === h - 1) {
      reachedBorder = true;
    }
    if (x > 0 && !mask[i - 1] && passable(i - 1)) {
      mask[i - 1] = 1;
      stack[sp++] = i - 1;
    }
    if (x < w - 1 && !mask[i + 1] && passable(i + 1)) {
      mask[i + 1] = 1;
      stack[sp++] = i + 1;
    }
    if (y > 0 && !mask[i - w] && passable(i - w)) {
      mask[i - w] = 1;
      stack[sp++] = i - w;
    }
    if (y < h - 1 && !mask[i + w] && passable(i + w)) {
      mask[i + w] = 1;
      stack[sp++] = i + w;
    }
  }
  return { mask, reachedBorder, count };
};

// 8-neighbour offsets, clockwise from East (screen coords, y down).
const NX = [1, 1, 0, -1, -1, -1, 0, 1];
const NY = [0, 1, 1, 1, 0, -1, -1, -1];

/**
 * Moore-neighbour boundary trace of a single filled blob, with Jacob's
 * stopping criterion. Returns an ordered ring `[x0,y0,x1,y1,…]` in raster px.
 */
const traceContour = (region: Uint8Array, w: number, h: number): number[] => {
  let start = -1;
  for (let i = 0; i < w * h; i++) {
    if (region[i]) {
      start = i;
      break;
    }
  }
  if (start < 0) {
    return [];
  }
  const sx = start % w;
  const sy = (start / w) | 0;

  const contour: number[] = [];
  let cx = sx;
  let cy = sy;
  let backtrack = 4; // we arrived at the topmost-leftmost pixel from the West
  let firstMoveX = -1;
  let firstMoveY = -1;
  const cap = 8 * w * h + 16;

  for (let step = 0; step < cap; step++) {
    contour.push(cx, cy);
    let nd = -1;
    for (let k = 1; k <= 8; k++) {
      const d = (backtrack + k) % 8;
      const nx = cx + NX[d];
      const ny = cy + NY[d];
      if (nx >= 0 && ny >= 0 && nx < w && ny < h && region[ny * w + nx]) {
        nd = d;
        break;
      }
    }
    if (nd < 0) {
      break; // isolated pixel
    }
    const nx = cx + NX[nd];
    const ny = cy + NY[nd];
    if (
      cx === sx &&
      cy === sy &&
      firstMoveX >= 0 &&
      nx === firstMoveX &&
      ny === firstMoveY
    ) {
      contour.pop(); // drop the duplicate start we just pushed
      contour.pop();
      break;
    }
    if (firstMoveX < 0) {
      firstMoveX = nx;
      firstMoveY = ny;
    }
    backtrack = (nd + 4) % 8;
    cx = nx;
    cy = ny;
  }
  return contour;
};

/** Ramer–Douglas–Peucker on a flat `[x,y,…]` array (perpendicular distance). */
const rdp = (pts: number[], eps: number): number[] => {
  const n = pts.length / 2;
  if (n < 3) {
    return pts.slice();
  }
  const keep = new Uint8Array(n);
  keep[0] = 1;
  keep[n - 1] = 1;
  const eps2 = eps * eps;
  const stack: number[] = [0, n - 1];
  while (stack.length) {
    const hi = stack.pop()!;
    const lo = stack.pop()!;
    const ax = pts[lo * 2];
    const ay = pts[lo * 2 + 1];
    const bx = pts[hi * 2];
    const by = pts[hi * 2 + 1];
    const vx = bx - ax;
    const vy = by - ay;
    const len2 = vx * vx + vy * vy || 1;
    let maxD = -1;
    let idx = -1;
    for (let i = lo + 1; i < hi; i++) {
      const px = pts[i * 2];
      const py = pts[i * 2 + 1];
      const t = ((px - ax) * vx + (py - ay) * vy) / len2;
      const projx = ax + t * vx;
      const projy = ay + t * vy;
      const ddx = px - projx;
      const ddy = py - projy;
      const d = ddx * ddx + ddy * ddy;
      if (d > maxD) {
        maxD = d;
        idx = i;
      }
    }
    if (idx >= 0 && maxD > eps2) {
      keep[idx] = 1;
      stack.push(lo, idx, idx, hi);
    }
  }
  const out: number[] = [];
  for (let i = 0; i < n; i++) {
    if (keep[i]) {
      out.push(pts[i * 2], pts[i * 2 + 1]);
    }
  }
  return out;
};

const findEmptyNear = (
  dil: Uint8Array,
  w: number,
  h: number,
  cx: number,
  cy: number,
  radius: number,
): [number, number] | null => {
  if (!dil[cy * w + cx]) {
    return [cx, cy];
  }
  for (let ring = 1; ring <= radius; ring++) {
    for (let dy = -ring; dy <= ring; dy++) {
      for (let dx = -ring; dx <= ring; dx++) {
        if (Math.max(Math.abs(dx), Math.abs(dy)) !== ring) {
          continue; // outer ring only, nearest-first
        }
        const nx = cx + dx;
        const ny = cy + dy;
        if (nx >= 0 && ny >= 0 && nx < w && ny < h && !dil[ny * w + nx]) {
          return [nx, ny];
        }
      }
    }
  }
  return null;
};

/**
 * PURE CORE. Given a binary ink `barrier` and a click pixel, returns a
 * simplified contour `[x,y,…]` in raster px for the enclosed region, or `null`
 * if the click is not inside a closed area. DOM-free — unit tested directly.
 */
export const maskToFillContour = (
  barrier: Uint8Array,
  w: number,
  h: number,
  clickPx: number,
  clickPy: number,
  opts?: { r?: number; eps?: number; minPixels?: number },
): number[] | null => {
  const r = opts?.r ?? DILATION_RADIUS;
  const eps = opts?.eps ?? RDP_EPSILON;
  const minPixels = opts?.minPixels ?? MIN_FILL_PIXELS;

  const dil = dilateBinary(barrier, w, h, r);

  let cx = clampInt(clickPx, 0, w - 1);
  let cy = clampInt(clickPy, 0, h - 1);
  if (dil[cy * w + cx]) {
    const near = findEmptyNear(dil, w, h, cx, cy, 2 * r);
    if (!near) {
      return null; // click buried in ink
    }
    [cx, cy] = near;
  }

  const interior = floodRegion(w, h, cx, cy, (i) => dil[i] === 0);
  if (interior.reachedBorder || interior.count < minPixels) {
    return null; // open to the outside, or too small
  }

  // Grow the fill back by r and clip to the ORIGINAL ink, so the fill meets the
  // strokes with no r-px seam (dilation had eaten into the interior).
  const grown = dilateBinary(interior.mask, w, h, r);
  const fill = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) {
    fill[i] = grown[i] && !barrier[i] ? 1 : 0;
  }
  const region = floodRegion(w, h, cx, cy, (i) => fill[i] === 1).mask;

  const contour = rdp(traceContour(region, w, h), eps);
  if (contour.length < 6) {
    return null; // fewer than 3 vertices → degenerate
  }
  return contour;
};

/**
 * Async wrapper: rasterize the scene, build the barrier, run the pure core, and
 * map the contour back to scene coordinates ready for `newLinearElement`.
 * Returns element-ready data, or `null` when there is nothing to fill.
 */
export const computeFloodFillContour = async ({
  elements,
  appState,
  files,
  sceneX,
  sceneY,
}: {
  elements: readonly NonDeletedExcalidrawElement[];
  appState: AppState;
  files: BinaryFiles;
  sceneX: number;
  sceneY: number;
}): Promise<{
  x: number;
  y: number;
  width: number;
  height: number;
  points: LocalPoint[];
} | null> => {
  const roots = getRootElements(elements);
  if (roots.length === 0) {
    return null;
  }

  const [minX, minY, maxX, maxY] = getCommonBounds(roots);
  const sceneW = Math.max(maxX - minX, 1);
  const sceneH = Math.max(maxY - minY, 1);
  const r = DILATION_RADIUS;

  const scale = Math.min(1, RASTER_CAP / Math.max(sceneW, sceneH));
  // Clear border ring ≥ r+1 device px so an enclosed region can't touch the edge.
  const padScene = Math.ceil((r + 2) / scale);

  const canvas = await exportToCanvas(
    elements,
    {
      ...appState,
      exportScale: scale,
      // Disable frame-name labels so exportToCanvas's bounds match `roots` above.
      frameRendering: { ...appState.frameRendering, name: false },
    },
    files,
    {
      exportBackground: false,
      viewBackgroundColor: "transparent",
      exportPadding: padScene,
    },
  );

  const w = canvas.width;
  const h = canvas.height;
  const ctx = canvas.getContext("2d");
  if (!ctx || w === 0 || h === 0) {
    return null;
  }
  const data = ctx.getImageData(0, 0, w, h).data;
  const barrier = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) {
    barrier[i] = data[i * 4 + 3] > ALPHA_THRESHOLD ? 1 : 0;
  }

  const clickPx = Math.round((sceneX - minX + padScene) * scale);
  const clickPy = Math.round((sceneY - minY + padScene) * scale);

  const contour = maskToFillContour(barrier, w, h, clickPx, clickPy);
  if (!contour) {
    return null;
  }

  // raster px → scene coords
  const scenePts: number[] = [];
  for (let i = 0; i < contour.length; i += 2) {
    scenePts.push(contour[i] / scale + minX - padScene);
    scenePts.push(contour[i + 1] / scale + minY - padScene);
  }

  let pMinX = Infinity;
  let pMinY = Infinity;
  let pMaxX = -Infinity;
  let pMaxY = -Infinity;
  for (let i = 0; i < scenePts.length; i += 2) {
    pMinX = Math.min(pMinX, scenePts[i]);
    pMaxX = Math.max(pMaxX, scenePts[i]);
    pMinY = Math.min(pMinY, scenePts[i + 1]);
    pMaxY = Math.max(pMaxY, scenePts[i + 1]);
  }

  // Excalidraw requires linear elements normalized so points[0] === (0,0):
  // anchor x,y at the FIRST contour point, not the bbox min corner.
  const originX = scenePts[0];
  const originY = scenePts[1];
  const points: LocalPoint[] = [];
  for (let i = 0; i < scenePts.length; i += 2) {
    points.push(
      pointFrom<LocalPoint>(scenePts[i] - originX, scenePts[i + 1] - originY),
    );
  }
  // close the loop exactly so isPathALoop() is satisfied
  const first = points[0];
  const last = points[points.length - 1];
  if (first[0] !== last[0] || first[1] !== last[1]) {
    points.push(pointFrom<LocalPoint>(first[0], first[1]));
  }
  if (points.length < 4) {
    return null;
  }

  return {
    x: originX,
    y: originY,
    width: pMaxX - pMinX, // bbox extent (invariant under the anchor shift)
    height: pMaxY - pMinY,
    points,
  };
};
