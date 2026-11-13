# Migration: Direct Query API → Inference API

This project is the [SWaT water-treatment demo](https://github.com/archetypeai/archetypeai-swat-demo-direct-query) with its Newton calls moved from the Direct Query API (`POST /api/lens/query`) to the Inference API: Newton Omega 1.4 → **Newton Omega 1.5** for embeddings, and Newton C 2.6 → **Newton C 2.6** (same model, new endpoint) for operator suggestions. The UI, the scaler, the data split, the KNN and the suggestion prompt are unchanged.

All measurements below were taken on production on 2026-11-13.

## Summary

- **Endpoint and models:** the server opens one session on `omega-embedding` (Omega 1.5) with `window_size` 128 and sends **one query per stage**, all of the stage's channels together. Suggestions call `newton-c-2.6` on `POST /api/inference/query`.
- **Fewer calls:** 6 Omega queries per window instead of 40 (one per sensor). Building the n-shot library takes **5.4 minutes** for 1,128 queries, instead of 25–35 minutes for about 7,500.
- **The API key no longer reaches the browser.** The Direct Query version returned `ATAI_API_KEY` from `/api/baselines` so the browser could call Newton for suggestions. Suggestions now go through the server's `/api/suggestions`, and that route is gone.
- **New library:** Omega 1.5 embeddings aren't comparable with Omega 1.4 ones, so `data/knn-library.json` and `data/inference-sample.json` were rebuilt.
- **The SWaT data is no longer committed.** It belongs to iTrust; the README explains how to download it and the prep scripts rebuild the CSVs.

## Direct Query API vs Inference API

|                     | Direct Query API                                                     | Inference API                                                                                                                               |
| ------------------- | -------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| Omega endpoint      | `POST /api/lens/query`                                               | `POST /api/inference/sessions` once, then `POST /api/inference/sessions/{id}/query`                                                         |
| Omega model         | `OmegaEncoder::omega_embeddings_1_4`                                 | `omega-embedding` (Omega 1.5)                                                                                                               |
| Window length       | 16–1024 per request; shorter is padded and masked                    | Set once by the session: `values.window_size` 128                                                                                           |
| Requests per window | One per channel: 40 across the 6 stages                              | One per stage: 6                                                                                                                            |
| Answer              | Flat `[768]` per channel request                                     | `[1, channels × 768]` per stage: one row, channels in order                                                                                 |
| `normalize_input`   | `false`, with the shared scaler                                      | Not sent (it has no effect); the shared scaler stays                                                                                        |
| Newton model        | `Newton::c2_6_8b_fp8_260424d7a55d5e`                                 | `newton-c-2.6`                                                                                                                              |
| Newton endpoint     | `POST /api/lens/query`, called **from the browser** with the API key | `POST /api/inference/query`, called from the server                                                                                         |
| Session lifecycle   | None                                                                 | Opened on first use, reopened on `409 session_not_running`, cancelled when the server or a script stops (16 open sessions per organization) |

## What changed in the code

| File                                                                | Change                                                                                                                                                                                                                                                     |
| ------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/lib/inference-client.js`                                       | New. `OmegaSession` (open, one-window queries through a bounded pool, reopen on `session_not_running`, retries, cancel), `queryNewton`, and `apiBase`, which accepts the endpoint with or without `/api`. Plain ESM, shared by the server and the scripts. |
| `src/lib/server/newton.js`                                          | Omega through one server-wide session (kept on `globalThis` so hot reload doesn't open more); the per-channel fan-out, `normalize_input` and `/lens/query` are gone. Newton through `queryNewton`.                                                         |
| `src/lib/server/suggestions.js`                                     | The suggestion prompt, query builder and parser, moved from `src/lib/suggestions-direct.js` (which ran in the browser) to the server. This was the maintained copy, with the 1,500-token budget.                                                           |
| `src/lib/server/baselines.js`                                       | New: per-sensor baselines from `swat_normal.csv`, for the suggestions prompt.                                                                                                                                                                              |
| `src/routes/api/suggestions/+server.js`                             | Uses the shared prompt, baselines and parser; caches by the set of anomalous stages.                                                                                                                                                                       |
| `src/routes/api/baselines/+server.js`                               | Removed: it returned the API key to the browser.                                                                                                                                                                                                           |
| `src/routes/+page.svelte`                                           | Always calls `/api/suggestions`; labels say Inference API.                                                                                                                                                                                                 |
| `vite.config.js`                                                    | The project's `.env` wins over `ATAI_*` shell variables (a shell pointing at the free trial silently redirected every call), and the Omega session is cancelled during Vite's shutdown.                                                                    |
| `scripts/build-knn-library.js`, `scripts/build-inference-sample.js` | Embed through `OmegaSession`, one query per stage window, up to 6 in flight; cancel the session at the end or on failure.                                                                                                                                  |
| `scripts/eval-playback.js`                                          | New: scores the classifier on the held-out playback stream.                                                                                                                                                                                                |
| `scripts/generate_labels.py`                                        | Writes only the files the app reads (n-shot and playback); the unused 395 MB `swat_inference.csv` and the quick-test file are gone.                                                                                                                        |
| Removed                                                             | Seven Direct Query probe and model-comparison scripts, and the Lens / Direct Query skills in `.claude/skills/` (listed in `CLAUDE.md`), which pointed coding agents at the old APIs.                                                                       |
| `data/`                                                             | No CSVs committed (see the README's Data section). `scaler.json` stays.                                                                                                                                                                                    |
| `LICENSE`                                                           | Apache License 2.0, for the code.                                                                                                                                                                                                                          |

## Results

### Library separability

Leave-one-out KNN accuracy (k=3) within each stage's library, as shown in the embedding panel's LOO badges:

|           | P1   | P2   | P3   | P4   | P5   | P6  |
| --------- | ---- | ---- | ---- | ---- | ---- | --- |
| Omega 1.4 | 100% | 100% | 100% | 100% | 100% | 50% |
| Omega 1.5 | 100% | 100% | 100% | 100% | 100% | 50% |

### Held-out playback

`node scripts/eval-playback.js` over every non-overlapping 128-row window of `swat_playback.csv` (3,000 normal rows, then 30,000 attack rows). A window is flagged when any stage says ATTACK.

|                                 | Omega 1.5, all 257 windows | Omega 1.5, every 4th window | Omega 1.4, every 4th window |
| ------------------------------- | -------------------------- | --------------------------- | --------------------------- |
| Normal windows flagged          | 0/23                       | 0/6                         | _pending_                   |
| Attack windows flagged          | 149/234 (64%)              | 38/59                       | _pending_                   |
| First flag after the transition | 1,480 rows                 | 1,608 rows                  | _pending_                   |

### Speed

|                                              | Direct Query (Omega 1.4) | Inference API (Omega 1.5)                                 |
| -------------------------------------------- | ------------------------ | --------------------------------------------------------- |
| Library build (`--step=20`)                  | 25–35 min, ~7,500 calls  | **5.4 min**, 1,128 queries                                |
| Inference sample (6,000 rows)                | not recorded             | **81 s**, 276 queries                                     |
| One window, all six stages (`/api/classify`) | ~1.5–2 s                 | **~2.2 s** (6.8 s for the first, which opens the session) |

## Reproducing

```bash
node scripts/build-knn-library.js --step=20
node scripts/build-inference-sample.js --offset=1384098 --rows=6000
node scripts/eval-playback.js            # or --every=4 for the quick sample
npm run dev
```
