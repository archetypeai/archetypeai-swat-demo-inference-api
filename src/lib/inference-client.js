// Archetype AI Inference API client, shared by the SvelteKit server and the
// build scripts in scripts/. Plain ESM with no SvelteKit imports, so Node can
// run it directly.
//
// - Newton Omega 1.5 (`omega-embedding`) embeds sensor windows. The app opens
//   one session whose window_size is the demo's window, so each slice of
//   windowSize rows is exactly one window, and sends all of a stage's channels
//   in one query.
// - Newton C 2.6 (`newton-c-2.6`) writes the operator suggestions.

export const OMEGA_MODEL = 'omega-embedding';
export const NEWTON_MODEL = 'newton-c-2.6';
export const EMBED_DIM = 768;

const SESSION_ENDED = new Set(['session_not_running', 'not_found']);

// Accept ATAI_API_ENDPOINT as a bare root, a root ending in /api, or the old
// /api/lens form, and return the /api base every call is made under.
export function apiBase(endpoint) {
	const root = String(endpoint || '')
		.trim()
		.replace(/\/+$/, '')
		.replace(/\/api(\/lens)?$/, '');
	if (!root) throw new Error('ATAI_API_ENDPOINT is not set');
	return `${root}/api`;
}

export class ApiError extends Error {
	constructor(status, body) {
		const first = body?.errors?.[0];
		super(
			`${status} ${first?.code ?? ''} ${first?.message ?? JSON.stringify(body).slice(0, 300)}`.trim()
		);
		this.status = status;
		this.code = first?.code ?? null;
	}
}

async function post(base, apiKey, path, body, timeoutMs) {
	const controller = new AbortController();
	const timer = setTimeout(() => controller.abort(), timeoutMs);
	try {
		const res = await fetch(`${base}${path}`, {
			method: 'POST',
			headers: { Authorization: `Bearer ${apiKey}`, 'Content-Type': 'application/json' },
			body: body === undefined ? undefined : JSON.stringify(body),
			signal: controller.signal
		});
		const text = await res.text();
		let data;
		try {
			data = text ? JSON.parse(text) : {};
		} catch {
			data = { raw: text.slice(0, 300) };
		}
		if (!res.ok) throw new ApiError(res.status, data);
		return data;
	} finally {
		clearTimeout(timer);
	}
}

const isTransient = (err) =>
	err?.name === 'AbortError' ||
	err?.status === 429 ||
	err?.status >= 500 ||
	err?.cause?.code === 'ECONNRESET';

// One Omega session at a fixed window length. embed() sends one window (all
// channels, channel first) and returns the per-channel vectors concatenated in
// channel order. Queries share a bounded in-flight pool; a session that has
// ended is reopened once, and transient failures are retried.
export class OmegaSession {
	constructor({
		endpoint,
		apiKey,
		windowSize,
		stepSize = windowSize,
		maxInFlight = 6,
		retries = 3,
		timeoutMs = 30000
	}) {
		this.base = apiBase(endpoint);
		this.apiKey = apiKey;
		this.values = { window_size: windowSize, step_size: stepSize };
		this.windowSize = windowSize;
		this.maxInFlight = maxInFlight;
		this.retries = retries;
		this.timeoutMs = timeoutMs;
		this.id = null;
		this.opening = null;
		this.closing = null;
		this.inFlight = 0;
		this.queue = [];
	}

	async open(replace = null) {
		if (this.id && this.id !== replace) return this.id;
		if (!this.opening) {
			this.opening = post(
				this.base,
				this.apiKey,
				'/inference/sessions',
				{ model: OMEGA_MODEL, values: this.values },
				this.timeoutMs
			)
				.then((s) => (this.id = s.id))
				.finally(() => (this.opening = null));
		}
		return this.opening;
	}

	// Cancel the session. Callers that arrive while a cancel is in flight get
	// the same promise, so every shutdown path can wait for the request to land.
	close() {
		const id = this.id;
		this.id = null;
		if (id) {
			this.closing = post(
				this.base,
				this.apiKey,
				`/inference/sessions/${id}/cancel`,
				undefined,
				10000
			).catch(() => {
				// Best effort: a session that isn't cancelled ends on its own.
			});
		}
		return this.closing ?? Promise.resolve();
	}

	slot(fn) {
		return new Promise((resolve, reject) => {
			const run = () => {
				this.inFlight++;
				Promise.resolve()
					.then(fn)
					.then(resolve, reject)
					.finally(() => {
						this.inFlight--;
						this.queue.shift()?.();
					});
			};
			if (this.inFlight < this.maxInFlight) run();
			else this.queue.push(run);
		});
	}

	async embed(channelFirstWindow) {
		const channels = channelFirstWindow.length;
		if (channelFirstWindow.some((c) => c.length !== this.windowSize)) {
			throw new Error(`each channel needs exactly ${this.windowSize} samples`);
		}
		const body = {
			model: OMEGA_MODEL,
			query: '',
			events: [{ type: 'data.numeric_array', event_data: { contents: channelFirstWindow } }]
		};
		return this.slot(async () => {
			let lastErr;
			for (let attempt = 0; attempt < this.retries; attempt++) {
				const id = await this.open();
				try {
					const data = await post(
						this.base,
						this.apiKey,
						`/inference/sessions/${id}/query`,
						body,
						this.timeoutMs
					);
					const answer = data.response ?? {};
					if (!answer.success)
						throw new Error('Omega returned success=false (input shorter than the window?)');
					const row = answer.response?.[0];
					if (!Array.isArray(row) || row.length !== channels * EMBED_DIM) {
						throw new Error(`expected ${channels * EMBED_DIM} values, got ${row?.length}`);
					}
					return Float32Array.from(row);
				} catch (err) {
					lastErr = err;
					if (SESSION_ENDED.has(err.code)) {
						await this.open(id);
						continue;
					}
					if (!isTransient(err) || attempt === this.retries - 1) throw err;
					await new Promise((r) => setTimeout(r, 400 * (attempt + 1)));
				}
			}
			throw lastErr;
		});
	}
}

// Newton C 2.6 text query. Returns the answer string.
export async function queryNewton({
	endpoint,
	apiKey,
	query,
	instructionPrompt = '',
	maxNewTokens = 1024,
	timeoutMs = 120000
}) {
	const data = await post(
		apiBase(endpoint),
		apiKey,
		'/inference/query',
		{
			model: NEWTON_MODEL,
			query,
			instruction_prompt: instructionPrompt,
			max_new_tokens: maxNewTokens
		},
		timeoutMs
	);
	const out = data.response?.response;
	return Array.isArray(out) ? out[0] || '' : typeof out === 'string' ? out : '';
}
