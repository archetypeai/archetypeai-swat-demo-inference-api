#!/usr/bin/env node
// Build the n-shot KNN library used by the classifier.
//
// For each of the 6 SWaT stages, this slides a window across swat_normal.csv
// and swat_attack.csv, embeds each window (all of the stage's channels in one
// query) with Newton Omega 1.5 through an Inference API session whose
// window_size is the window length, and stores the embeddings labeled
// NORMAL / ATTACK.
//
// Output: data/knn-library.json
//   {
//     config: { windowSize, stepSize, model, builtAt },
//     stages: {
//       P1: {
//         columns: [...],
//         embeddings: [{ label: 'NORMAL'|'ATTACK', vec: number[] }, ...]
//       },
//       ...
//     }
//   }
//
// Each `vec` is a flattened [num_columns * 768] vector.
//
// Usage: node scripts/build-knn-library.js [--window=128] [--step=128]

import { readFileSync, writeFileSync, mkdirSync, existsSync } from 'fs';
import { resolve, dirname } from 'path';
import { OmegaSession, OMEGA_MODEL, apiBase } from '../src/lib/inference-client.js';

// Global per-channel scaler, shared with the server's classify path, so the
// library and live windows share one reference frame.
const SCALER_PATH = resolve('data/scaler.json');
const SCALER = existsSync(SCALER_PATH) ? JSON.parse(readFileSync(SCALER_PATH, 'utf-8')) : null;
if (!SCALER) {
	console.error('Missing data/scaler.json — run `node scripts/build-scaler.js` first.');
	process.exit(1);
}
function applyScaler(channelFirstWindow, columns) {
	const out = new Array(columns.length);
	for (let c = 0; c < columns.length; c++) {
		const col = columns[c];
		const m = SCALER.mean[col] ?? 0;
		const s = SCALER.std[col] ?? 1;
		const src = channelFirstWindow[c];
		const dst = new Array(src.length);
		for (let i = 0; i < src.length; i++) dst[i] = (src[i] - m) / s;
		out[c] = dst;
	}
	return out;
}

const STAGE_COLUMNS = {
	P1: ['FIT101', 'LIT101', 'MV101', 'P101'],
	P2: ['AIT201', 'AIT202', 'AIT203', 'FIT201', 'MV201', 'P203', 'P205'],
	P3: ['DPIT301', 'FIT301', 'LIT301', 'MV301', 'MV302', 'MV303', 'MV304', 'P301', 'P302'],
	P4: ['AIT401', 'AIT402', 'FIT401', 'LIT401', 'P402', 'UV401'],
	P5: [
		'AIT501', 'AIT502', 'AIT503', 'AIT504',
		'FIT501', 'FIT502', 'FIT503', 'FIT504',
		'P501', 'PIT501', 'PIT502', 'PIT503'
	],
	P6: ['FIT601', 'P602']
};
const STAGE_IDS = Object.keys(STAGE_COLUMNS);
const MODEL = OMEGA_MODEL;

function loadEnv() {
	const env = {};
	const raw = readFileSync('.env', 'utf-8');
	for (const line of raw.split('\n')) {
		const m = line.match(/^([A-Z_][A-Z0-9_]*)=(.*)$/);
		if (m) env[m[1]] = m[2].trim();
	}
	return env;
}

function parseArgs() {
	const args = { window: 128, step: 128 };
	for (const a of process.argv.slice(2)) {
		const m = a.match(/^--(\w+)=(\d+)$/);
		if (m) args[m[1]] = parseInt(m[2]);
	}
	return args;
}

function readCsv(filePath) {
	const text = readFileSync(filePath, 'utf-8');
	const lines = text.split(/\r?\n/).filter((l) => l.trim());
	const headers = lines[0].split(',').map((h) => h.trim());
	const rows = lines.slice(1).map((line) => line.split(','));
	return { headers, rows };
}

// Return channel-first 2D array [num_cols x window_size] for the given window.
function extractWindow(rows, headerIdx, startRow, windowSize) {
	const out = headerIdx.map(() => new Array(windowSize));
	for (let r = 0; r < windowSize; r++) {
		const row = rows[startRow + r];
		for (let c = 0; c < headerIdx.length; c++) {
			const v = parseFloat(row[headerIdx[c]]);
			out[c][r] = isNaN(v) ? 0 : v;
		}
	}
	return out;
}

async function buildForStage(stageId, csvSets, headerIdxByLabel, session, windowSize, stepSize) {
	const embeddings = [];
	const stageColumns = STAGE_COLUMNS[stageId];
	for (const { label, csv } of csvSets) {
		const headerIdx = headerIdxByLabel[label];
		const rows = csv.rows;
		const starts = [];
		for (let start = 0; start + windowSize <= rows.length; start += stepSize) starts.push(start);
		// The session's pool caps how many queries run at once; retries live in the client.
		const vecs = await Promise.all(
			starts.map((start) => session.embed(applyScaler(extractWindow(rows, headerIdx, start, windowSize), stageColumns)))
		);
		for (const vec of vecs) embeddings.push({ label, vec: Array.from(vec) });
		console.log(`  ${stageId} ${label}: ${vecs.length} windows`);
	}
	return embeddings;
}

// The Omega session, cancelled when the script ends or fails (an organization
// may hold 16 open sessions on the model).
let session = null;

async function main() {
	const env = loadEnv();
	const args = parseArgs();
	console.log(`Endpoint: ${apiBase(env.ATAI_API_ENDPOINT)}  model: ${MODEL}`);
	session = new OmegaSession({
		endpoint: env.ATAI_API_ENDPOINT,
		apiKey: env.ATAI_API_KEY,
		windowSize: args.window,
		maxInFlight: 6
	});
	console.log(`Window: ${args.window}, Step: ${args.step}`);

	const normal = readCsv(resolve('data/swat_normal.csv'));
	const attack = readCsv(resolve('data/swat_attack.csv'));
	console.log(`Loaded ${normal.rows.length} normal rows, ${attack.rows.length} attack rows`);

	const stages = {};
	const t0 = Date.now();
	for (const stageId of STAGE_IDS) {
		const cols = STAGE_COLUMNS[stageId];
		const normalIdx = cols.map((c) => normal.headers.indexOf(c));
		const attackIdx = cols.map((c) => attack.headers.indexOf(c));
		if (normalIdx.some((i) => i < 0) || attackIdx.some((i) => i < 0)) {
			throw new Error(`${stageId}: missing column in csv`);
		}
		console.log(`\n${stageId} (${cols.length} cols):`);
		const embeddings = await buildForStage(
			stageId,
			[
				{ label: 'NORMAL', csv: normal },
				{ label: 'ATTACK', csv: attack }
			],
			{ NORMAL: normalIdx, ATTACK: attackIdx },
			session,
			args.window,
			args.step
		);
		stages[stageId] = { columns: cols, embeddings };
	}

	await session.close();
	const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
	const lib = {
		config: {
			windowSize: args.window,
			stepSize: args.step,
			model: MODEL,
			builtAt: new Date().toISOString()
		},
		stages
	};

	const outPath = resolve('data/knn-library.json');
	mkdirSync(dirname(outPath), { recursive: true });
	writeFileSync(outPath, JSON.stringify(lib));
	const sizeMb = (readFileSync(outPath).length / 1024 / 1024).toFixed(2);
	console.log(`\nWrote ${outPath} (${sizeMb} MB) in ${elapsed}s`);
}

main().catch(async (err) => {
	console.error('FATAL:', err);
	await session?.close();
	process.exit(1);
});
