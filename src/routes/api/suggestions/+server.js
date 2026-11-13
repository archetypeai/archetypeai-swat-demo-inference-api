import { json } from '@sveltejs/kit';
import { queryNewton } from '$lib/server/newton.js';
import { loadBaselines } from '$lib/server/baselines.js';
import { SYSTEM_PROMPT, buildQuery, parseSuggestions } from '$lib/server/suggestions.js';

// Operator suggestions for the stages currently classified as attack, from
// Newton C 2.6 on the Inference API. Answers are cached by the set of anomalous
// stages, so a repeated signature doesn't query Newton again.
const cache = new Map();

export async function POST({ request }) {
	try {
		const { stageStatuses, stageSensors = {} } = await request.json();
		if (!stageStatuses) return json({ error: 'Missing stageStatuses' }, { status: 400 });

		const anomalous = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6']
			.filter((id) => stageStatuses[id] === 'attack')
			.sort();
		const signature = anomalous.join(',') || 'none';

		if (anomalous.length === 0) {
			return json({ suggestions: [], source: 'newton', signature });
		}
		if (cache.has(signature)) {
			return json({ suggestions: cache.get(signature), source: 'newton-cached', signature });
		}

		const raw = await queryNewton({
			query: buildQuery(stageStatuses, stageSensors, loadBaselines()),
			systemPrompt: SYSTEM_PROMPT,
			maxNewTokens: 1500
		});
		const parsed = parseSuggestions(raw);
		if (!parsed || parsed.length === 0) {
			console.warn('[suggestions] unparseable Newton response:', raw);
			return json({ suggestions: [], source: 'error', signature, raw });
		}

		cache.set(signature, parsed);
		return json({ suggestions: parsed, source: 'newton', signature });
	} catch (err) {
		return json({ error: err.message, source: 'error' }, { status: 500 });
	}
}
