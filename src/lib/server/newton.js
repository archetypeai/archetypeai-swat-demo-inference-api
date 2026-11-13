import { ATAI_API_KEY, ATAI_API_ENDPOINT } from '$env/static/private';
import { readFileSync, existsSync } from 'fs';
import { resolve } from 'path';
import { projectEmbedding } from './projections.js';
import { OmegaSession, queryNewton as queryNewtonModel } from '../inference-client.js';

// SWaT stage → sensor column mapping. Identical to the original Lens-based demo
// so the same n-shot files and KNN library work without conversion.
export const STAGE_COLUMNS = {
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
export const STAGE_IDS = Object.keys(STAGE_COLUMNS);
export const MONITORED_STAGE_IDS = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6'];

export const DEFAULT_CONFIG = {
	windowSize: 128,
	stepSize: 128,
	nNeighbors: 3
};

// ──────────────────────────────────────────────────────────────────────
// Global per-channel StandardScaler (data/scaler.json). Loaded lazily.
// Omega 1.5 standardizes each window itself and passes the window's original
// scale to the encoder, but every window is still scaled with these fixed
// stats first, so the library and live windows share one reference frame.
// See scripts/build-scaler.js.
// ──────────────────────────────────────────────────────────────────────

let SCALER = null;
let SCALER_ERROR = null;
function ensureScaler() {
	if (SCALER || SCALER_ERROR) return;
	const path = resolve('data/scaler.json');
	if (!existsSync(path)) {
		SCALER_ERROR = new Error(
			'Missing data/scaler.json — run `node scripts/build-scaler.js` first.'
		);
		return;
	}
	SCALER = JSON.parse(readFileSync(path, 'utf-8'));
}

function applyScaler(channelFirstWindow, columns) {
	ensureScaler();
	if (SCALER_ERROR) throw SCALER_ERROR;
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

// ──────────────────────────────────────────────────────────────────────
// KNN library (loaded once at boot from data/knn-library.json)
// ──────────────────────────────────────────────────────────────────────

let LIBRARY = null;
let LIBRARY_ERROR = null;
function ensureLibrary() {
	if (LIBRARY || LIBRARY_ERROR) return;
	const path = resolve('data/knn-library.json');
	if (!existsSync(path)) {
		LIBRARY_ERROR = new Error(
			'Missing data/knn-library.json — run `node scripts/build-knn-library.js` first.'
		);
		return;
	}
	const raw = JSON.parse(readFileSync(path, 'utf-8'));
	for (const stageId of Object.keys(raw.stages)) {
		raw.stages[stageId].embeddings = raw.stages[stageId].embeddings.map((e) => ({
			label: e.label,
			vec: Float32Array.from(e.vec)
		}));
	}
	LIBRARY = raw;
}

export function getLibraryConfig() {
	ensureLibrary();
	if (LIBRARY_ERROR) throw LIBRARY_ERROR;
	return LIBRARY.config;
}

// ──────────────────────────────────────────────────────────────────────
// Omega 1.5 embedding through one Inference API session
// ──────────────────────────────────────────────────────────────────────

// One session for the whole server, at the demo's window length, kept on
// globalThis so Vite's hot reload doesn't open a new one on every edit. All
// stages share its in-flight cap. It is cancelled when the server stops; an
// organization may hold 16 open sessions on the model.
function getOmegaSession() {
	const key = '__swatOmegaSession';
	if (!globalThis[key]) {
		const session = new OmegaSession({
			endpoint: ATAI_API_ENDPOINT,
			apiKey: ATAI_API_KEY,
			windowSize: DEFAULT_CONFIG.windowSize,
			maxInFlight: 6
		});
		const stop = (signal) => {
			session.close().finally(() => process.exit(signal === 'SIGINT' ? 130 : 143));
		};
		process.once('SIGINT', () => stop('SIGINT'));
		process.once('SIGTERM', () => stop('SIGTERM'));
		globalThis[key] = session;
	}
	return globalThis[key];
}

// Send a [num_columns x window_size] channel-first window to Omega in one
// query and return the per-channel 768-d vectors concatenated in channel
// order, as a single Float32Array for KNN against the library.
export async function embedWindow(channelFirstWindow) {
	return getOmegaSession().embed(channelFirstWindow);
}

// ──────────────────────────────────────────────────────────────────────
// Local KNN classifier
// ──────────────────────────────────────────────────────────────────────

function euclideanSq(a, b) {
	let s = 0;
	for (let i = 0; i < a.length; i++) {
		const d = a[i] - b[i];
		s += d * d;
	}
	return s;
}

function classifyEmbedding(stageId, embedding, k = DEFAULT_CONFIG.nNeighbors) {
	ensureLibrary();
	if (LIBRARY_ERROR) throw LIBRARY_ERROR;
	const lib = LIBRARY.stages[stageId];
	if (!lib) throw new Error(`no library for stage ${stageId}`);
	const dists = new Array(lib.embeddings.length);
	for (let i = 0; i < lib.embeddings.length; i++) {
		dists[i] = { d: euclideanSq(lib.embeddings[i].vec, embedding), label: lib.embeddings[i].label };
	}
	dists.sort((a, b) => a.d - b.d);
	const top = dists.slice(0, k);
	const votes = {};
	for (const t of top) votes[t.label] = (votes[t.label] || 0) + 1;
	let winner = null;
	let max = -1;
	for (const [label, n] of Object.entries(votes)) {
		if (n > max) {
			max = n;
			winner = label;
		}
	}
	return { label: winner, neighbors: top.map((t) => ({ label: t.label, dist: Math.sqrt(t.d) })) };
}

function extractStageWindow(stageId, rows) {
	const cols = STAGE_COLUMNS[stageId];
	return cols.map((col) =>
		rows.map((row) => {
			const v = parseFloat(row[col]);
			return isNaN(v) ? 0 : v;
		})
	);
}

// Run Omega → KNN for one stage. Returns the label, neighbors,
// and the raw embedding (so the client can run PCA-2 projection for the
// embedding-viz panel without re-querying Omega).
export async function classifyStage(stageId, rows, { k = DEFAULT_CONFIG.nNeighbors } = {}) {
	const win = extractStageWindow(stageId, rows);
	const scaled = applyScaler(win, STAGE_COLUMNS[stageId]);
	const embedding = await embedWindow(scaled);
	const { label, neighbors } = classifyEmbedding(stageId, embedding, k);
	// Project to 2D for the embedding-viz panel. Cheap (~10-50ms total for
	// PCA + UMAP transform). If projection ever fails, the classification
	// itself still returns — viz coords are best-effort.
	let coords;
	try {
		coords = await projectEmbedding(stageId, embedding);
	} catch {
		coords = null;
	}
	return { stageId, label, neighbors, coords };
}

export async function classifyAllStages(rows, opts = {}) {
	const results = await Promise.allSettled(
		MONITORED_STAGE_IDS.map((stageId) => classifyStage(stageId, rows, opts))
	);
	const out = {};
	const errors = [];
	for (let i = 0; i < results.length; i++) {
		const stageId = MONITORED_STAGE_IDS[i];
		const r = results[i];
		if (r.status === 'fulfilled') {
			out[stageId] = r.value;
		} else {
			errors.push({ stageId, error: r.reason?.message || String(r.reason) });
		}
	}
	return { stages: out, errors };
}

// ──────────────────────────────────────────────────────────────────────
// Newton C 2.6 text reasoning (operator suggestions)
// ──────────────────────────────────────────────────────────────────────

export async function queryNewton({ query, systemPrompt = '', maxNewTokens = 1024 }) {
	return queryNewtonModel({
		endpoint: ATAI_API_ENDPOINT,
		apiKey: ATAI_API_KEY,
		query,
		instructionPrompt: systemPrompt,
		maxNewTokens
	});
}
