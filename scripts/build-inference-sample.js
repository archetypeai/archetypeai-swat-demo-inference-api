#!/usr/bin/env node
// Build a sampled-inference embedding pool for the embedding-viz panel.
//
// Reads a slice of swat_raw_labeled.csv around the demo's playback offset,
// embeds windows for each stage with Omega 1.5 (one session query per
// stage window), and writes
// data/inference-sample.json. The server projects these through the same
// PCA-2 / UMAP-2 fitted on the n-shot library, so the viz panel can show
// a dense "where real playback windows land" cloud in addition to the
// 30-point library.
//
// Usage: node scripts/build-inference-sample.js [--offset=701000] [--rows=6000] [--window=128] [--step=128]

import { readFileSync, writeFileSync, existsSync } from 'fs';
import { resolve } from 'path';
import { OmegaSession, OMEGA_MODEL, apiBase } from '../src/lib/inference-client.js';

// Global per-channel scaler shared with build-knn-library.js and the live
// /api/classify path, so every window shares one reference frame.
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
	const args = { offset: 701000, rows: 6000, window: 128, step: 128 };
	for (const a of process.argv.slice(2)) {
		const m = a.match(/^--(\w+)=(\d+)$/);
		if (m) args[m[1]] = parseInt(m[2]);
	}
	return args;
}

// Stream-parse only the rows we need; the labeled file is 1.4M rows / ~400 MB.
function readCsvSlice(filePath, offset, count) {
	const text = readFileSync(filePath, 'utf-8');
	const lines = text.split(/\r?\n/);
	const headers = lines[0].split(',').map((h) => h.trim());
	const slice = lines.slice(1 + offset, 1 + offset + count).filter((l) => l.length);
	const rows = slice.map((line) => line.split(','));
	return { headers, rows };
}

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

// Majority ground-truth label across the window's rows.
function majorityLabel(rows, labelIdx, startRow, windowSize) {
	let attack = 0;
	let normal = 0;
	for (let r = 0; r < windowSize; r++) {
		const lbl = (rows[startRow + r][labelIdx] || '').trim().toLowerCase();
		if (lbl === 'attack') attack++;
		else if (lbl === 'normal') normal++;
	}
	return attack > normal ? 'ATTACK' : 'NORMAL';
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
	console.log(`Offset ${args.offset}, slice ${args.rows} rows, window=${args.window}, step=${args.step}`);

	const { headers, rows } = readCsvSlice(
		resolve('data/swat_raw_labeled.csv'),
		args.offset,
		args.rows
	);
	const labelIdx = headers.indexOf('label');
	if (labelIdx < 0) throw new Error('label column missing');
	console.log(`Loaded ${rows.length} rows`);

	const stages = {};
	const t0 = Date.now();
	for (const stageId of STAGE_IDS) {
		const cols = STAGE_COLUMNS[stageId];
		const idx = cols.map((c) => headers.indexOf(c));
		if (idx.some((i) => i < 0)) throw new Error(`${stageId}: missing column`);

		const embeddings = [];
		const numWindows = Math.floor((rows.length - args.window) / args.step) + 1;
		console.log(`\n${stageId} (${cols.length} cols): ${numWindows} windows`);
		const windows = [];
		for (let i = 0; i < numWindows; i++) {
			const start = i * args.step;
			windows.push({ start, truth: majorityLabel(rows, labelIdx, start, args.window) });
		}
		const vecs = await Promise.all(
			windows.map(({ start }) => session.embed(applyScaler(extractWindow(rows, idx, start, args.window), cols)))
		);
		for (let i = 0; i < windows.length; i++) {
			embeddings.push({ vec: Array.from(vecs[i]), label: windows[i].truth, rowOffset: args.offset + windows[i].start });
		}
		stages[stageId] = { columns: cols, embeddings };
	}

	await session.close();

	const out = {
		config: {
			offset: args.offset,
			rows: args.rows,
			windowSize: args.window,
			stepSize: args.step,
			model: MODEL,
			builtAt: new Date().toISOString()
		},
		stages
	};
	const outPath = resolve('data/inference-sample.json');
	writeFileSync(outPath, JSON.stringify(out));
	const sizeMb = (readFileSync(outPath).length / 1024 / 1024).toFixed(2);
	console.log(`\nWrote ${outPath} (${sizeMb} MB) in ${((Date.now() - t0) / 1000).toFixed(0)}s`);
}

main().catch(async (err) => {
	console.error('FATAL:', err);
	await session?.close();
	process.exit(1);
});
