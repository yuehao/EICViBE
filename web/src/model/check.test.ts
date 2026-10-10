import { describe, expect, it } from "vitest";

import { agrees, decades, lineOf, tuneDifference, tuneText } from "./check";

describe("the check's arithmetic", () => {
  it("puts a tune's line where a spectrum shows it", () => {
    expect(lineOf(28.2002)).toBeCloseTo(0.2002, 6);
    expect(lineOf(27.8)).toBeCloseTo(0.2, 6); // 1 - q
    expect(lineOf(null)).toBeNull();
  });

  it("compares measured and model tunes", () => {
    const plane = { measured_full: 28.2525, model_tune: 28.2002 } as Parameters<typeof tuneDifference>[0];
    expect(tuneDifference(plane)).toBeCloseTo(0.0523, 6);
    expect(agrees(0.0005)).toBe(true);
    expect(agrees(0.0523)).toBe(false);
    expect(agrees(null)).toBe(false);
  });

  it("chooses decades for a log axis and writes tunes", () => {
    expect(decades([0.002, 3.5, null, 0])).toEqual([-3, 1]);
    expect(decades([])).toEqual([-3, 0]);
    expect(tuneText(0.25253, 28.25253)).toBe("0.2525 (28.2525)");
    expect(tuneText(null)).toBe("—");
  });
});
