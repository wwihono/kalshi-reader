import { describe, expect, it } from "vitest";
import {
  filterEvents,
  formatVolume,
  toCents,
  toProbabilityPercent,
  type KalshiEvent,
} from "./kalshi";

describe("toProbabilityPercent", () => {
  it("converts a dollar-string into an implied probability percentage", () => {
    expect(toProbabilityPercent("0.0900")).toBe("9%");
    expect(toProbabilityPercent("0.8000")).toBe("80%");
    expect(toProbabilityPercent("1.0000")).toBe("100%");
  });

  it("returns a placeholder for missing or invalid values", () => {
    expect(toProbabilityPercent(undefined)).toBe("—");
    expect(toProbabilityPercent(null)).toBe("—");
    expect(toProbabilityPercent("")).toBe("—");
    expect(toProbabilityPercent("abc")).toBe("—");
  });
});

describe("toCents", () => {
  it("formats a dollar-string as a cents price", () => {
    expect(toCents("0.0900")).toBe("9¢");
    expect(toCents("0.4600")).toBe("46¢");
  });

  it("returns a placeholder for missing values", () => {
    expect(toCents(undefined)).toBe("—");
  });
});

describe("formatVolume", () => {
  it("formats large volumes compactly", () => {
    expect(formatVolume(115072)).toBe("115.1K");
    expect(formatVolume(2_400_000)).toBe("2.4M");
    expect(formatVolume(42)).toBe("42");
  });

  it("handles missing volume", () => {
    expect(formatVolume(undefined)).toBe("—");
    expect(formatVolume(null)).toBe("—");
  });
});

describe("filterEvents", () => {
  const events: KalshiEvent[] = [
    { event_ticker: "A", title: "Who will the next Pope be?", category: "Elections", markets: [] },
    { event_ticker: "B", title: "Will it rain in NYC?", category: "Climate and Weather", markets: [] },
  ];

  it("returns all events for an empty query", () => {
    expect(filterEvents(events, "")).toHaveLength(2);
    expect(filterEvents(events, "   ")).toHaveLength(2);
  });

  it("matches on title case-insensitively", () => {
    const result = filterEvents(events, "pope");
    expect(result).toHaveLength(1);
    expect(result[0].event_ticker).toBe("A");
  });

  it("matches on category", () => {
    const result = filterEvents(events, "weather");
    expect(result).toHaveLength(1);
    expect(result[0].event_ticker).toBe("B");
  });
});
