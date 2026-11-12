// Rolling-origin backtest of main.ts (and the persistence baseline) on a history.ndjson.
//   node eval.mjs <input dir> [main.ts path]
// For each origin read i (needs >= MIN_TRAIN earlier reads), the history is cut after read i, the window is the 72 h
// from the next whole hour, and the forecast is scored (rules: log of per-place probability, floor 1e-4, unnamed share
// spread over unnamed games) on every later read inside that window. Prints the mean over origins.
import { readFileSync, writeFileSync, mkdirSync, mkdtempSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';

const inDir = resolve(process.argv[2] ?? 'input');
const mainPath = resolve(process.argv[3] ?? 'main.ts');
const MIN_TRAIN = Number(process.env.MIN_TRAIN ?? 12);
const HOUR = 3600_000;
const lines = readFileSync(join(inDir, 'history.ndjson'), 'utf8').split('\n').filter(Boolean);
const recs = lines.map((l) => JSON.parse(l));
const task0 = JSON.parse(readFileSync(join(inDir, 'task.json'), 'utf8'));

function chartOf(r) { return [...r.up_and_coming].sort((a, b) => a.rank - b.rank).map((x) => x.universe_id); }
function scoreRead(fc, chart, P) {
  let s = 0, hits = 0;
  for (let k = 1; k <= P; k++) {
    const rows = fc.probabilities.filter((x) => x.place === k);
    const named = new Map(rows.map((x) => [x.universe_id, x.p]));
    const sum = rows.reduce((a, x) => a + x.p, 0);
    const unnamedN = chart.filter((id) => !named.has(id)).length;
    const id = chart[k - 1];
    const p = named.has(id) ? named.get(id) : (1 - sum) / Math.max(1, unnamedN);
    s += Math.log(Math.max(p, 1e-4));
  }
  for (const id of fc.top10) if (chart.slice(0, P).includes(id)) hits++;
  return { s, hits };
}

const results = [];
for (let i = MIN_TRAIN; i < recs.length - 1; i++) {
  const t = Date.parse(recs[i].at);
  const from = Math.ceil((t + 1) / HOUR) * HOUR;
  const until = from + 72 * HOUR;
  const targets = recs.slice(i + 1).filter((r) => { const x = Date.parse(r.at); return x >= from && x < until; });
  if (!targets.length) continue;
  const tmp = mkdtempSync(join(tmpdir(), 'ev-'));
  mkdirSync(join(tmp, 'in')); mkdirSync(join(tmp, 'out'));
  writeFileSync(join(tmp, 'in', 'history.ndjson'), lines.slice(0, i + 1).join('\n') + '\n');
  const task = { ...task0, history_until: new Date(from).toISOString(), window: { ...task0.window, from: new Date(from).toISOString(), until: new Date(until).toISOString() }, snapshots: i + 1 };
  writeFileSync(join(tmp, 'in', 'task.json'), JSON.stringify(task));
  const env = { ...process.env, MAKEFUN_INPUT_DIR: join(tmp, 'in'), MAKEFUN_OUTPUT_DIR: join(tmp, 'out') };
  execFileSync('node', [mainPath], { env });
  const fc = JSON.parse(readFileSync(join(tmp, 'out', task.output), 'utf8'));
  let tot = 0, h = 0;
  for (const r of targets) { const sc = scoreRead(fc, chartOf(r), task.places); tot += sc.s; h += sc.hits; }
  results.push({ at: recs[i].at, n: targets.length, score: tot / targets.length, hits: h / targets.length });
}
const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
if (process.env.VERBOSE) for (const r of results) console.log(r.at, r.n, r.score.toFixed(2), r.hits.toFixed(1));
console.log(`origins ${results.length}  mean log score ${mean(results.map((r) => r.score)).toFixed(3)}  hits ${mean(results.map((r) => r.hits)).toFixed(2)}`);
