#!/usr/bin/env node
// Score the classifier on the held-out playback stream (data/swat_playback.csv),
// the same windows the demo replays and never part of the library.
//
// Each non-overlapping window is scaled, embedded with Omega 1.5 (one session
// query per stage) and classified per stage by KNN against data/knn-library.json,
// exactly as /api/classify does. A window counts as flagged when any stage
// says ATTACK; its true label is the majority label of its rows.
//
// Reports: false alarms on normal windows, flagged attack windows, how many rows
// after the normal→attack transition the first flag fires, and per-stage counts.
//
// Usage: node scripts/eval-playback.js [--every=1] [--k=3]
//   --every=N  score every Nth window only (a quicker, smaller sample)

import { readFileSync } from 'fs';
import { resolve } from 'path';
import { OmegaSession, apiBase } from '../src/lib/inference-client.js';

const STAGE_COLUMNS = {
	P1: ['FIT101', 'LIT101', 'MV101', 'P101'],
	P2: ['AIT201', 'AIT202', 'AIT203', 'FIT201', 'MV201', 'P203', 'P205'],
	P3: ['DPIT301', 'FIT301', 'LIT301', 'MV301', 'MV302', 'MV303', 'MV304', 'P301', 'P302'],
	P4: ['AIT401', 'AIT402', 'FIT401', 'LIT401', 'P402', 'UV401'],
	P5: [
		'AIT501',
		'AIT502',
		'AIT503',
		'AIT504',
		'FIT501',
		'FIT502',
		'FIT503',
		'FIT504',
		'P501',
		'PIT501',
		'PIT502',
		'PIT503'
	],
	P6: ['FIT601', 'P602']
};
const STAGE_IDS = Object.keys(STAGE_COLUMNS);

function loadEnv() {
	const env = {};
	for (const line of readFileSync('.env', 'utf-8').split('\n')) {
		const m = line.match(/^([A-Z_][A-Z0-9_]*)=(.*)$/);
		if (m) env[m[1]] = m[2].trim();
	}
	return env;
}

function parseArgs() {
	const args = { every: 1, k: 3 };
	for (const a of process.argv.slice(2)) {
		const m = a.match(/^--(\w+)=(\d+)$/);
		if (m) args[m[1]] = parseInt(m[2]);
	}
	return args;
}

const scaler = JSON.parse(readFileSync(resolve('data/scaler.json'), 'utf-8'));
function scaledWindow(rows, headerIdx, columns, start, size) {
	return columns.map((col, c) => {
		const m = scaler.mean[col] ?? 0;
		const s = scaler.std[col] ?? 1;
		const out = new Array(size);
		for (let r = 0; r < size; r++) {
			const v = parseFloat(rows[start + r][headerIdx[c]]);
			out[r] = ((isNaN(v) ? 0 : v) - m) / s;
		}
		return out;
	});
}

// Same KNN as src/lib/server/newton.js: squared euclidean, top k, majority vote,
// ties going to the label seen first among the nearest.
function knnLabel(library, vec, k) {
	const dists = library.map((e) => {
		let d = 0;
		for (let i = 0; i < vec.length; i++) {
			const x = e.vec[i] - vec[i];
			d += x * x;
		}
		return { d, label: e.label };
	});
	dists.sort((a, b) => a.d - b.d);
	const votes = {};
	for (const t of dists.slice(0, k)) votes[t.label] = (votes[t.label] || 0) + 1;
	let winner = null;
	let max = -1;
	for (const [label, n] of Object.entries(votes)) {
		if (n > max) {
			max = n;
			winner = label;
		}
	}
	return winner;
}

let session = null;

async function main() {
	const env = loadEnv();
	const args = parseArgs();
	const lib = JSON.parse(readFileSync(resolve('data/knn-library.json'), 'utf-8'));
	const size = lib.config.windowSize;
	console.log(
		`Endpoint: ${apiBase(env.ATAI_API_ENDPOINT)}  library: ${lib.config.model}, window ${size}`
	);

	const lines = readFileSync(resolve('data/swat_playback.csv'), 'utf-8')
		.split(/\r?\n/)
		.filter((l) => l.trim());
	const headers = lines[0].split(',').map((h) => h.trim());
	const rows = lines.slice(1).map((l) => l.split(','));
	const labelIdx = headers.indexOf('label');
	const transition = rows.findIndex((r) => r[labelIdx].trim().toLowerCase() === 'attack');

	const starts = [];
	for (let s = 0, i = 0; s + size <= rows.length; s += size, i++)
		if (i % args.every === 0) starts.push(s);

	session = new OmegaSession({
		endpoint: env.ATAI_API_ENDPOINT,
		apiKey: env.ATAI_API_KEY,
		windowSize: size,
		maxInFlight: 6
	});
	const t0 = Date.now();
	const results = await Promise.all(
		starts.map(async (start) => {
			let attack = 0;
			for (let r = start; r < start + size; r++)
				if (rows[r][labelIdx].trim().toLowerCase() === 'attack') attack++;
			const truth = attack > size / 2 ? 'ATTACK' : 'NORMAL';
			const stages = {};
			await Promise.all(
				STAGE_IDS.map(async (id) => {
					const cols = STAGE_COLUMNS[id];
					const idx = cols.map((c) => headers.indexOf(c));
					const vec = await session.embed(scaledWindow(rows, idx, cols, start, size));
					stages[id] = knnLabel(lib.stages[id].embeddings, vec, args.k);
				})
			);
			return { start, truth, stages, flagged: Object.values(stages).includes('ATTACK') };
		})
	);
	await session.close();

	const normal = results.filter((r) => r.truth === 'NORMAL');
	const attack = results.filter((r) => r.truth === 'ATTACK');
	const firstFlag = attack.find((r) => r.flagged);
	console.log(
		`\n${results.length} windows (every ${args.every}), ${starts.length * STAGE_IDS.length} Omega queries in ${((Date.now() - t0) / 1000).toFixed(0)}s`
	);
	console.log(
		`normal windows flagged (false alarms): ${normal.filter((r) => r.flagged).length}/${normal.length}`
	);
	console.log(
		`attack windows flagged:                ${attack.filter((r) => r.flagged).length}/${attack.length}`
	);
	console.log(
		`first flag after the transition:       ${firstFlag ? `${firstFlag.start - transition} rows (window at row ${firstFlag.start})` : 'never'}`
	);
	console.log('per stage, attack windows flagged / normal windows flagged:');
	for (const id of STAGE_IDS) {
		const a = attack.filter((r) => r.stages[id] === 'ATTACK').length;
		const n = normal.filter((r) => r.stages[id] === 'ATTACK').length;
		console.log(`  ${id}: ${a}/${attack.length}  ${n}/${normal.length}`);
	}
}

main().catch(async (err) => {
	console.error('FATAL:', err);
	await session?.close();
	process.exit(1);
});
