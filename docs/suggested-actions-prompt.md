# Suggested Actions — Newton Prompt Reference

This demo asks Newton C 2.6, on the Inference API, to turn per-stage anomaly flags from the Omega + KNN classifier into concrete upstream / local / downstream operator actions. The prompt was iterated on heavily to keep Newton from hallucinating sensor names, collapsing to generic guidance, or mis-routing cards across the plant topology.

Use this as a template for similar "reason over structured state" use cases in other apps.

## Endpoint

```
POST {ATAI_API_ENDPOINT}/api/inference/query
Authorization: Bearer <API_KEY>
Content-Type: application/json
```

The server makes this call (`/api/suggestions`), so the API key never reaches the browser. See `src/lib/server/suggestions.js` for the prompt, query builder and parser, and `src/lib/inference-client.js` for the request.

## Request body

```json
{
  "model": "newton-c-2.6",
  "query": "<plant state snapshot — see below>",
  "instruction_prompt": "<SYSTEM_PROMPT — see below>",
  "max_new_tokens": 1500
}
```

Notes on the params:

- **`model`** — Newton C 2.6 by its Inference API name.
- **`max_new_tokens: 1500`** — enough budget for every attack stage's direction cards. Smaller budgets (500, then 700) truncated the final card mid-sentence, which breaks `JSON.parse`.
- **`instruction_prompt`** — Newton C 2.6 applies `instruction_prompt` as the system turn; the prompt goes there.
- No files or attachments: the prompt is fully self-contained.

The answer is the model's text at `response.response[0]`.

## System prompt

```text
You are an operator assistant for the SWaT six-stage water treatment plant. Flow: P1 raw intake → P2 chemical dosing → P3 ultrafiltration → P4 UV dechlorination → P5 reverse osmosis → treated water. P6 backwash recycles P5 reject to clean P3 UF membranes.

For each stage marked ATTACK, suggest specific operator actions in three directions:
- upstream: reduce or hold feed from the stage immediately before the anomalous one
- local: check / isolate equipment on the anomalous stage itself
- downstream: alert or protect the stage immediately after the anomalous one from cascading effects

TOPOLOGY — the "target" field MUST EXACTLY match these pairings. Never substitute or reassign:
- P1 anomaly: upstream=none, local=P1, downstream=P2
- P2 anomaly: upstream=P1, local=P2, downstream=P3
- P3 anomaly: upstream=P2, local=P3, downstream=P4
- P4 anomaly: upstream=P3, local=P4, downstream=P5
- P5 anomaly: upstream=P4, local=P5, downstream=P6
- P6 anomaly: upstream=P5, local=P6, downstream=none

If a direction maps to "none" for a given origin, DO NOT emit that direction.

Return ONLY a JSON array. No prose, no markdown code fences, no explanation — just the JSON. Shape:
[{"origin":"Pn","target":"Pm","direction":"upstream|local|downstream","text":"..."}]

Rules:
- Only generate suggestions for stages marked ATTACK. Skip NORMAL, STANDBY, CLASSIFYING.
- For EVERY ATTACK stage emit all three direction cards (or two cards if topology says "none" for one direction). Do not skip a stage.
- Each "text" field is a full instruction with TWO parts joined by " — ":
    part 1: the EXACT sensor citation from that stage's "cite this sensor:" line (copy sensor name and value verbatim; drop the z annotation)
    part 2: an imperative operator action with a concrete verb (reduce, check, isolate, alert, hold, bypass)
- Example shape (do not copy sensor name — use the one from "cite this sensor:" for each stage):
    {"origin":"PX","target":"PX","direction":"local","text":"<sensor>=<value> — check valve and surrounding piping"}
    {"origin":"PX","target":"PY","direction":"upstream","text":"<sensor>=<value> — reduce feed from PY to stabilise intake"}
    {"origin":"PX","target":"PZ","direction":"downstream","text":"<sensor>=<value> — alert PZ of contamination risk"}
- CRITICAL: use the sensor name+value specified after "cite this sensor:" for that stage. Do NOT substitute a different sensor even if another looks more familiar.
- For upstream/downstream cards, the action text must name the TARGET stage by its Pn code.
- Keep each "text" field under 140 characters.
```

### Why the prompt is shaped this way

Each of these bullets came from a specific failure mode during development:

- **Explicit topology table.** Newton initially routed upstream cards to the wrong target (e.g. P3 → P1 instead of P3 → P2). Spelling out the mapping and validating server-side fixed it.
- **`"none"` directions omitted.** Without this, Newton emitted phantom upstream cards for P1 and downstream cards for P6 with invented targets.
- **"Return ONLY a JSON array."** Newton likes to wrap responses in ```` ```json ```` fences and prose. We strip fences in the parser, but the instruction cuts most of it at the source.
- **Two-part `"text"` with ` — ` separator.** Earlier outputs were either pure citation (`"LIT101=648.46"`) or pure prose with no numbers. Forcing both halves keeps cards operator-actionable.
- **"cite this sensor:" pre-selection.** Newton has a strong prior toward familiar sensor names (FIT101, LIT101) and would cite them even when the actual anomaly was on AIT402 or DPIT301. We now z-rank sensors server-side and hand Newton a single sensor per stage; the rule forbids substitution.
- **Fake example sensor names (`PX/PY/PZ`).** Concrete examples in the prompt get copied verbatim by Newton. Using placeholder names prevents contamination.
- **140-char cap.** UI cards wrap ugly beyond that.

## Query body (per-request)

The `query` field is built fresh for each anomaly-set change. Shape:

```text
Current plant state. Each attack stage specifies the exact sensor to cite — do not substitute:
- P1 raw intake: ATTACK
    cite this sensor: FIT101=0.00 z=84.3 (strong deviation)
- P2 chemical dosing: NORMAL
- P3 ultrafiltration: ATTACK
    cite this sensor: LIT301=815.40 z=0.5 (weak signal)
- P4 UV dechlorination: NORMAL
- P5 reverse osmosis: NORMAL
- P6 backwash: STANDBY

Generate suggestions for ATTACK stages only. Emit three direction cards per attack stage (or two if topology says "none" for one direction).
```

Key construction choices (see `buildQuery` and `pickTopDeviation` in `src/lib/server/suggestions.js`):

- **One sensor per stage, pre-picked by z-score.** We compute `z = |value - mean| / std` against baselines from `swat_normal.csv` and hand Newton only the top-z sensor. Multiple sensor choices caused Newton to default to familiar names instead of the most-deviating one.
- **Signal-strength annotation.** `(strong deviation)` / `(moderate)` / `(weak signal)` gives Newton a hint about confidence without letting it skip the card.
- **NORMAL / STANDBY stages listed but without a sensor.** Context helps Newton reason about cascading effects, but we don't want cards generated for them.

## Response shape

Newton returns:

```json
{
  "response": { "response": ["<string>"] }
}
```

The string should be a parseable JSON array. We strip code fences, extract the outermost `[ ... ]`, `JSON.parse` it, and **topology-validate every item** — dropping any card whose `target` doesn't match the required mapping for its `origin` + `direction`. See `parseSuggestions` in `src/lib/server/suggestions.js`.

Example parsed output (4 attack stages × 3 cards each, minus 1 topology-excluded):

```json
[
  { "origin": "P1", "target": "P1", "direction": "local",      "text": "FIT101=0.00 — check MV101 intake valve, likely shut" },
  { "origin": "P1", "target": "P2", "direction": "downstream", "text": "FIT101=0.00 — alert P2 dosing: no feed, hold chemical pumps" },
  { "origin": "P3", "target": "P2", "direction": "upstream",   "text": "DPIT301=20.10 — hold feed from P2 while UF pressure is checked" },
  { "origin": "P3", "target": "P3", "direction": "local",      "text": "DPIT301=20.10 — inspect UF membranes for fouling or blockage" },
  { "origin": "P3", "target": "P4", "direction": "downstream", "text": "DPIT301=20.10 — alert P4 of reduced clarified flow" }
]
```

## End-to-end latency

On production on 2026-11-13, one request for a single attack stage took about 10 s through `/api/suggestions`. Answers are cached by the set of anomalous stages, so a repeated set returns immediately.

Fast enough to debounce on anomaly-set change and refetch whenever the set changes. We use a 1.5s debounce and drop overlapping requests.

## Reusing this for your own app

To adapt:

1. Replace the topology and stage names in the system prompt with your own graph.
2. Replace `pickTopDeviation` with whatever ranking signal makes sense for your domain (z-score, rate-of-change, rule score, etc.).
3. Keep the "cite this sensor:" / pre-pick pattern if your model tends to drift toward familiar identifiers — it's the single biggest win for citation accuracy.
4. Keep topology validation on the parsed output; Newton will occasionally route cards incorrectly regardless of how forcefully the prompt states the mapping.
