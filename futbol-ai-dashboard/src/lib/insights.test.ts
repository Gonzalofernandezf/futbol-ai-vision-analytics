import { describe, it, expect } from "vitest";
import {
  countSprints,
  computeSpeedZones,
  computeHighIntensityPct,
  computeDropOff,
  zonePctsFromCounts,
  buildInsightCandidates,
  selectTopInsights,
  JOG_KMH,
} from "./insights";
import type { Player, PlayerDerived, TeamStats } from "@/types/match";

function makePlayer(speedOverTime: Array<number | null>, overrides: Partial<Player> = {}): Player {
  return {
    id: "1",
    team: 1,
    max_speed_kmh: 30,
    total_distance_m: 5000,
    max_acceleration_ms2: 5,
    speed_over_time: speedOverTime,
    position_history: [],
    ...overrides,
  };
}

/** Full derived block mirroring analytics/match_analytics.py output */
function makeDerived(overrides: Partial<PlayerDerived> = {}): PlayerDerived {
  return {
    active_time_sec: 5400,
    missing_data_pct: 2,
    sprints: { count: 18, longest_sec: 6, total_distance_m: 350 },
    speed_zones_sec: { walk: 1080, jog: 1620, run: 1620, sprint: 1080 },
    high_intensity_pct: 50,
    accelerations: { high_count: 12, max_ms2: 4.2 },
    halves: {
      first: { distance_m: 5500, avg_speed_kmh: 8.2, sprints: 10, high_intensity_pct: 55 },
      second: { distance_m: 4800, avg_speed_kmh: 7.1, sprints: 8, high_intensity_pct: 45 },
    },
    drop_off_pct: 18,
    peak_window: { start_sec: 3600, end_sec: 4500, avg_speed_kmh: 14.5 },
    position: {
      centroid_m: [50, 32],
      dispersion_m: 8.5,
      coverage_pct: 22,
      dominant_cell: [5, 5],
      lateral_band: "center",
    },
    percentiles: { sprints: 95, high_intensity_pct: 82, max_speed_kmh: 60 },
    ...overrides,
  };
}

// ---------------------------------------------------------------------------
// countSprints
// ---------------------------------------------------------------------------
describe("countSprints", () => {
  it("returns 0 when no sample exceeds sprint threshold", () => {
    expect(countSprints([5, 10, 15, 18])).toBe(0);
  });

  it("counts one sprint for a single sprint event", () => {
    // enters sprint zone once, exits once
    expect(countSprints([5, 22, 25, 15])).toBe(1);
  });

  it("counts multiple distinct sprint events", () => {
    // sprint → sub-sprint → sprint = 2 events
    expect(countSprints([25, 10, 22])).toBe(2);
  });

  it("counts one sprint when player stays in sprint zone continuously", () => {
    expect(countSprints([22, 25, 28, 24])).toBe(1);
  });

  it("ignores null values without resetting sprint state", () => {
    // sprint, null, still in sprint zone → 1 sprint; then drops, then new sprint → 2
    expect(countSprints([null, 22, null, 10, 22])).toBe(2);
  });

  it("returns 0 for an empty array", () => {
    expect(countSprints([])).toBe(0);
  });

  it("returns 0 for an all-null array", () => {
    expect(countSprints([null, null, null])).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// computeSpeedZones
// ---------------------------------------------------------------------------
describe("computeSpeedZones", () => {
  it("classifies four samples into equal zones", () => {
    // 1 walk (3), 1 jog (10), 1 run (18), 1 sprint (25)
    const z = computeSpeedZones([3, 10, 18, 25]);
    expect(z.walk_pct).toBeCloseTo(25);
    expect(z.jog_pct).toBeCloseTo(25);
    expect(z.run_pct).toBeCloseTo(25);
    expect(z.sprint_pct).toBeCloseTo(25);
  });

  it("returns all zeros for an empty array", () => {
    const z = computeSpeedZones([]);
    expect(z.walk_pct).toBe(0);
    expect(z.jog_pct).toBe(0);
    expect(z.run_pct).toBe(0);
    expect(z.sprint_pct).toBe(0);
  });

  it("ignores null values in percentage denominator", () => {
    // valid samples: walk (3), sprint (25); null ignored
    const z = computeSpeedZones([3, null, 25]);
    expect(z.walk_pct).toBeCloseTo(50);
    expect(z.sprint_pct).toBeCloseTo(50);
    expect(z.jog_pct).toBeCloseTo(0);
  });

  it("respects JOG_KMH boundary (value exactly at threshold is jog, not walk)", () => {
    // below JOG_KMH → walk; exactly JOG_KMH or above → jog
    const z = computeSpeedZones([JOG_KMH - 0.001, JOG_KMH, JOG_KMH + 0.001]);
    expect(z.walk_pct).toBeCloseTo(33.33, 1);
    expect(z.jog_pct).toBeCloseTo(66.67, 1);
  });

  it("percentages sum to 100 for mixed data", () => {
    const z = computeSpeedZones([2, 8, 17, 25, 3, 14, 22, 19]);
    const sum = z.walk_pct + z.jog_pct + z.run_pct + z.sprint_pct;
    expect(sum).toBeCloseTo(100);
  });
});

// ---------------------------------------------------------------------------
// computeHighIntensityPct
// ---------------------------------------------------------------------------
describe("computeHighIntensityPct", () => {
  it("sums run_pct and sprint_pct", () => {
    // 1 walk, 1 jog, 1 run, 1 sprint → run+sprint = 50%
    expect(computeHighIntensityPct([3, 10, 18, 25])).toBeCloseTo(50);
  });

  it("returns 0 when all movement is below run threshold", () => {
    expect(computeHighIntensityPct([2, 4, 6, 10, 14])).toBeCloseTo(0);
  });

  it("returns 100 when all samples are sprints", () => {
    expect(computeHighIntensityPct([22, 25, 30])).toBeCloseTo(100);
  });
});

// ---------------------------------------------------------------------------
// zonePctsFromCounts
// ---------------------------------------------------------------------------
describe("zonePctsFromCounts", () => {
  it("converts second counts into percentages summing 100", () => {
    const z = zonePctsFromCounts({ walk: 10, jog: 20, run: 30, sprint: 40 });
    expect(z.walk_pct).toBeCloseTo(10);
    expect(z.jog_pct).toBeCloseTo(20);
    expect(z.run_pct).toBeCloseTo(30);
    expect(z.sprint_pct).toBeCloseTo(40);
  });

  it("returns all zeros when total is 0", () => {
    const z = zonePctsFromCounts({ walk: 0, jog: 0, run: 0, sprint: 0 });
    expect(z.walk_pct + z.jog_pct + z.run_pct + z.sprint_pct).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// computeDropOff
// ---------------------------------------------------------------------------
describe("computeDropOff", () => {
  it("returns positive when second half intensity drops", () => {
    expect(computeDropOff(40, 20)).toBeCloseTo(50);
  });

  it("returns negative when second half intensity improves", () => {
    expect(computeDropOff(20, 40)).toBeCloseTo(-100);
  });

  it("returns 0 when first half HI is 0 (avoids division by zero)", () => {
    expect(computeDropOff(0, 20)).toBe(0);
  });

  it("returns 0 when both halves are equal", () => {
    expect(computeDropOff(30, 30)).toBeCloseTo(0);
  });
});

// ---------------------------------------------------------------------------
// selectTopInsights
// ---------------------------------------------------------------------------
describe("selectTopInsights", () => {
  const high = (key: string) => ({ key, severity: "high" as const, message: key });
  const low = (key: string) => ({ key, severity: "low" as const, message: key });

  it("always places high-severity candidates first", () => {
    const result = selectTopInsights([low("l1"), low("l2"), high("h1"), high("h2")], 4);
    expect(result[0].key).toBe("h1");
    expect(result[1].key).toBe("h2");
  });

  it("fills remaining slots with low-severity when not enough high ones", () => {
    const result = selectTopInsights([low("l1"), low("l2"), high("h1")], 3);
    expect(result).toHaveLength(3);
    expect(result.find((c) => c.key === "l1")).toBeDefined();
  });

  it("caps at n when more high-severity candidates than n exist", () => {
    const result = selectTopInsights([high("a"), high("b"), high("c"), high("d"), high("e")], 3);
    expect(result).toHaveLength(3);
    expect(result.every((c) => c.severity === "high")).toBe(true);
  });

  it("returns all candidates when total is less than n", () => {
    const result = selectTopInsights([high("h1"), low("l1")], 4);
    expect(result).toHaveLength(2);
  });
});

// ---------------------------------------------------------------------------
// buildInsightCandidates
// ---------------------------------------------------------------------------
describe("buildInsightCandidates", () => {
  it("recalculates from speed_over_time when derived is absent (v1 fallback)", () => {
    // 1 sprint event (enters >21 once)
    const player = makePlayer([5, 25, 25, 10], {
      position_history: [[50, 32]] as Array<[number, number] | null>,
    });
    const candidates = buildInsightCandidates(player, null, [0, 1], 1);
    const sprint = candidates.find((c) => c.key === "sprints");
    expect(sprint).toBeDefined();
    expect(sprint?.message).toContain("1");
  });

  it("uses precomputed sprints from derived.sprints.count for full-match range", () => {
    const player = makePlayer([], {
      derived: makeDerived({ sprints: { count: 18, longest_sec: 6, total_distance_m: 350 } }),
      position_history: [],
    });
    const candidates = buildInsightCandidates(player, null, [0, 90], 90);
    const sprintInsight = candidates.find(
      (c) => c.key === "sprints_leader" || c.key === "sprints" || c.key === "sprints_low",
    );
    expect(sprintInsight).toBeDefined();
    expect(sprintInsight?.message).toContain("18");
  });

  it("renders peak_window in minutes from start_sec/end_sec for full-match range", () => {
    const player = makePlayer([], {
      derived: makeDerived({
        peak_window: { start_sec: 3600, end_sec: 4500, avg_speed_kmh: 14.5 },
      }),
      position_history: [],
    });
    const candidates = buildInsightCandidates(player, null, [0, 90], 90);
    const peak = candidates.find((c) => c.key === "peak_window");
    expect(peak).toBeDefined();
    expect(peak?.message).toContain("60–75");
    expect(peak?.message).toContain("14.5");
  });

  it("reports decline when drop_off_pct is positive (backend convention)", () => {
    const player = makePlayer([], {
      derived: makeDerived({ drop_off_pct: 20 }),
      position_history: [],
    });
    const candidates = buildInsightCandidates(player, null, [0, 90], 90);
    const drop = candidates.find((c) => c.key === "drop_off");
    expect(drop).toBeDefined();
    expect(drop?.message).toContain("Bajó");
  });

  it("uses derived.halves.first when range matches first half", () => {
    const player = makePlayer(new Array<null>(90).fill(null), {
      derived: makeDerived({
        sprints: { count: 20, longest_sec: 6, total_distance_m: 400 },
        halves: {
          first: { distance_m: 5500, avg_speed_kmh: 8.2, sprints: 12, high_intensity_pct: 55 },
          second: { distance_m: 4800, avg_speed_kmh: 7.1, sprints: 8, high_intensity_pct: 45 },
        },
        percentiles: { sprints: 70, high_intensity_pct: 50 },
        missing_data_pct: 5,
      }),
      position_history: [],
    });
    const candidates = buildInsightCandidates(player, null, [0, 45], 90);
    const sprintInsight = candidates.find(
      (c) => c.key === "sprints_leader" || c.key === "sprints" || c.key === "sprints_low",
    );
    expect(sprintInsight?.message).toContain("12");
  });

  it("omits peak_window and drop_off insights when missing_data_pct > 30", () => {
    const player = makePlayer([], {
      derived: makeDerived({
        sprints: { count: 5, longest_sec: 3, total_distance_m: 90 },
        high_intensity_pct: 30,
        halves: {
          first: { distance_m: 3000, avg_speed_kmh: 7.5, sprints: 3, high_intensity_pct: 35 },
          second: { distance_m: 2000, avg_speed_kmh: 5.4, sprints: 2, high_intensity_pct: 25 },
        },
        drop_off_pct: 28,
        percentiles: { sprints: 50, high_intensity_pct: 50 },
        missing_data_pct: 35,
      }),
      position_history: [],
    });
    const candidates = buildInsightCandidates(player, null, [0, 90], 90);
    expect(candidates.find((c) => c.key === "peak_window")).toBeUndefined();
    expect(candidates.find((c) => c.key === "drop_off")).toBeUndefined();
  });

  it("marks sprint_leader as high severity when player is team sprint leader", () => {
    const player = makePlayer([], {
      id: "7",
      derived: makeDerived({
        sprints: { count: 22, longest_sec: 7, total_distance_m: 460 },
        drop_off_pct: 10,
        percentiles: { sprints: 98 },
        missing_data_pct: 3,
      }),
      position_history: [],
    });
    const teamStats: TeamStats = {
      players_count: 11,
      percentiles: {},
      leaders: { sprints_count: { player_id: "7", value: 22 } },
    };
    const candidates = buildInsightCandidates(player, teamStats, [0, 90], 90);
    const leader = candidates.find((c) => c.key === "sprints_leader");
    expect(leader).toBeDefined();
    expect(leader?.severity).toBe("high");
    expect(leader?.message).toContain("22");
  });
});
