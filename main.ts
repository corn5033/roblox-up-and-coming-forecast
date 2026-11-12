// Roblox Up-and-Coming top-10 forecaster (Node 22, standard library only, no network, deterministic).
//
// Idea: the scored window averages over many hourly reads, so the forecast should be the chart's *horizon-averaged*
// transition law. From the history itself we learn, for a game sitting at rank r on an earlier read, how often it
// holds place k on reads that come `off .. off+window` hours later (off = gap between the last read and the window's
// start). Rank is bucketed, the counts are shrunk toward a smooth geometric prior around the current rank, and rows are
// then calibrated and capped so every place sums to at most 1. The unnamed rest is spread by the grader over the chart.
//
//   MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Tunables (fitted on rolling-origin backtests with eval.mjs).
const PRIOR_STRENGTH = 25;
const PRIOR_SCALE = 3;
const PRIOR_MASS = 0.3;
const CALIB = 0.75;
const MAX_RANK_NAMED = 40;
const HOUR = 3600_000;

interface Row { rank: number; universe_id: number; playing: number | null }
interface Snap { t: number; ranks: Map<number, number>; chart: number[] }

const task = JSON.parse(readFileSync(join(INPUT_DIR, 'task.json'), 'utf8')) as {
  places: number; output: string; window: { from: string; slots: number; slot_minutes: number };
};
const PLACES = task.places;
const snaps: Snap[] = [];
for (const line of readFileSync(join(INPUT_DIR, 'history.ndjson'), 'utf8').split('\n')) {
  if (line === '') continue;
  const rec = JSON.parse(line) as { at: string; ok?: boolean; up_and_coming?: Row[] | null };
  if (rec.ok === false || !rec.up_and_coming || rec.up_and_coming.length === 0) continue;
  const rows = [...rec.up_and_coming].sort((a, b) => a.rank - b.rank);
  const ranks = new Map<number, number>();
  rows.forEach((r, i) => ranks.set(r.universe_id, i + 1));
  snaps.push({ t: Date.parse(rec.at), ranks, chart: rows.map((r) => r.universe_id) });
}
snaps.sort((a, b) => a.t - b.t);

const last = snaps[snaps.length - 1]!;
const winFrom = Date.parse(task.window.from);
const winLen = task.window.slots * task.window.slot_minutes * 60_000;
const off = Math.max(0, winFrom - last.t);

/** Rank bucket: 1..15 alone, then wider bands. */
function bucket(r: number): number {
  if (r <= 15) return r - 1;
  if (r <= 20) return 15;
  if (r <= 25) return 16;
  if (r <= 30) return 17;
  if (r <= 40) return 18;
  if (r <= 55) return 19;
  return 20;
}
const NB = 21;
const cnt: number[][] = Array.from({ length: NB }, () => new Array<number>(PLACES).fill(0));
const tot: number[] = new Array<number>(NB).fill(0);

// Pairs (origin i, target j) with horizon in [off, off+winLen]; weight each origin's targets so every hour-of-horizon
// bin counts equally (history reads can be hourly or every few hours).
for (let i = 0; i < snaps.length; i++) {
  const o = snaps[i]!;
  const bins = new Map<number, number[]>();
  for (let j = i + 1; j < snaps.length; j++) {
    const dt = snaps[j]!.t - o.t;
    if (dt < off - 1 || dt > off + winLen) continue;
    const b = Math.floor((dt - off) / HOUR);
    const a = bins.get(b);
    if (a) a.push(j); else bins.set(b, [j]);
  }
  for (const js of bins.values()) {
    const w = 1 / js.length;
    for (const j of js) {
      const tg = snaps[j]!;
      for (const [id, r] of o.ranks) {
        const b = bucket(r);
        tot[b]! += w;
        const k = tg.ranks.get(id);
        if (k !== undefined && k <= PLACES) cnt[b]![k - 1]! += w;
      }
    }
  }
}

/** P(game now at rank r holds place k, averaged over the window). */
function prob(r: number, k: number): number {
  const b = bucket(r);
  // geometric prior around the current rank, total mass PRIOR_MASS spread over places (renormalised on 1..PLACES)
  let z = 0;
  for (let q = 1; q <= PLACES; q++) z += Math.exp(-Math.abs(q - r) / PRIOR_SCALE);
  const full = 1 + 2 * (Math.exp(-1 / PRIOR_SCALE) / (1 - Math.exp(-1 / PRIOR_SCALE)));
  const mass = r <= PLACES ? PRIOR_MASS : PRIOR_MASS * Math.min(1, z / full) * Math.exp(-(r - PLACES) / 12);
  const prior = (mass * Math.exp(-Math.abs(k - r) / PRIOR_SCALE)) / (z || 1);
  const n = tot[b]!;
  return (cnt[b]![k - 1]! + PRIOR_STRENGTH * prior) / (n + PRIOR_STRENGTH);
}

const probabilities: Array<{ universe_id: number; place: number; p: number }> = [];
for (let k = 1; k <= PLACES; k++) {
  const rows: Array<{ id: number; p: number }> = [];
  for (let idx = 0; idx < last.chart.length && idx < MAX_RANK_NAMED; idx++) {
    const p = prob(idx + 1, k) * CALIB;
    rows.push({ id: last.chart[idx]!, p });
  }
  const sum = rows.reduce((s, x) => s + x.p, 0);
  const scale = sum > 0.98 ? 0.98 / sum : 1;
  const unnamed = Math.max(1, last.chart.length - rows.length);
  const left = 1 - Math.min(sum * scale, 0.98);
  const share = left / unnamed;
  for (const x of rows) {
    const p = Math.round(x.p * scale * 1e6) / 1e6;
    // naming a game below the share it would get unnamed only hurts
    if (p > share * 1.05 && p >= 0.0005) probabilities.push({ universe_id: x.id, place: k, p });
  }
}

// top10: the game with the highest expected place; greedy assignment place by place on the probabilities.
const best: number[] = [];
const used = new Set<number>();
for (let k = 1; k <= PLACES; k++) {
  let bid = -1, bp = -1;
  for (const row of probabilities) {
    if (row.place !== k || used.has(row.universe_id)) continue;
    if (row.p > bp || (row.p === bp && row.universe_id < bid)) { bp = row.p; bid = row.universe_id; }
  }
  if (bid === -1) { for (const id of last.chart) if (!used.has(id)) { bid = id; break; } }
  used.add(bid);
  best.push(bid);
}

writeFileSync(join(OUTPUT_DIR, task.output), `${JSON.stringify({ probabilities, top10: best })}\n`);
