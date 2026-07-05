import { describe, it, expect } from "vitest";

import { maskToFillContour, dilateBinary } from "./floodFill";

const W = 256;
const H = 256;

// A square barrier ring in [40..216] with a gap of width `g` on the top edge.
const ringWithGap = (g: number): Uint8Array => {
  const b = new Uint8Array(W * H);
  const lo = 40;
  const hi = 216;
  const set = (x: number, y: number) => {
    b[y * W + x] = 1;
  };
  const gapLo = 128 - Math.floor(g / 2);
  const gapHi = gapLo + g;
  for (let x = lo; x <= hi; x++) {
    if (x < gapLo || x >= gapHi) {
      set(x, lo); // top edge (with the gap)
    }
    set(x, hi); // bottom edge
  }
  for (let y = lo; y <= hi; y++) {
    set(lo, y); // left edge
    set(hi, y); // right edge
  }
  return b;
};

describe("floodFill pure core", () => {
  const opts = { r: 6, eps: 1.5, minPixels: 64 };

  it("fills a solid closed ring", () => {
    expect(maskToFillContour(ringWithGap(0), W, H, 128, 128, opts)).not.toBeNull();
  });

  it("bridges a small gap (< 2r) and fills", () => {
    expect(maskToFillContour(ringWithGap(8), W, H, 128, 128, opts)).not.toBeNull();
  });

  it("does not fill an open region (gap > 2r)", () => {
    expect(maskToFillContour(ringWithGap(20), W, H, 128, 128, opts)).toBeNull();
  });

  it("does not fill an open field (no barrier)", () => {
    expect(
      maskToFillContour(new Uint8Array(W * H), W, H, 128, 128, opts),
    ).toBeNull();
  });

  it("returns a closed-ish polygon (>= 3 vertices) for a filled region", () => {
    const contour = maskToFillContour(ringWithGap(0), W, H, 128, 128, opts);
    expect(contour).not.toBeNull();
    expect((contour as number[]).length).toBeGreaterThanOrEqual(6);
  });
});

describe("dilateBinary", () => {
  it("grows a single pixel into a (2r+1) square (Chebyshev r)", () => {
    const src = new Uint8Array(W * H);
    src[128 * W + 128] = 1;
    const out = dilateBinary(src, W, H, 6);
    expect(out[128 * W + 128]).toBe(1);
    expect(out[(128 - 6) * W + 128]).toBe(1); // 6 up: inside
    expect(out[(128 + 6) * W + (128 + 6)]).toBe(1); // corner at Chebyshev 6: inside
    expect(out[(128 - 7) * W + 128]).toBe(0); // 7 up: outside
  });
});
