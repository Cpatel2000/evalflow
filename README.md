# gradetrail

[![CI](https://github.com/Cpatel2000/gradetrail/actions/workflows/ci.yml/badge.svg)](https://github.com/Cpatel2000/gradetrail/actions/workflows/ci.yml)

A distributed LLM evaluation harness that treats evals like tests: declarative, cached, reproducible.

![demo](docs/demo.gif)

> PyPI publication pending.

## Quickstart

```bash
pip install gradetrail  # pending; for now: pip install git+https://github.com/Cpatel2000/gradetrail
export ANTHROPIC_API_KEY=sk-ant-...
```

Define an eval:

```yaml
# gsm8k_subset.yaml
name: gsm8k-subset
dataset:
  path: data/gsm8k_subset.jsonl
prompt: |
  Solve the following math problem. End your response with the line
  "Answer: <number>".

  {{ question }}
model:
  provider: anthropic
  name: claude-sonnet-4-6
scorer:
  type: regex
  pattern: 'Answer:\s*{{ answer }}\s*$'
```

`data/gsm8k_subset.jsonl` is three JSONL rows, one sample per line:

```json
{"id": "1", "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?", "answer": "72"}
{"id": "2", "question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "answer": "10"}
{"id": "3", "question": "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?", "answer": "624"}
```

Run it:

```bash
gradetrail run gsm8k_subset.yaml
```

Output from an actual cold run of this exact spec:

```
Samples: 3 (scored=3, provider_error=0, judge_error=0)
Mean score: 1.0000
Tokens: 189 in / 213 out
Cost: $0.0038
Cache hits: 0/3
Wall time: 2.50s
```

Run it again and it completes in about 40ms at $0.00: every response is cached, keyed on the request, not the scorer. The run writes `results/<name>-<identity-hash>/results.jsonl` (per-sample scores and responses) and `manifest.json` (spec hash, dataset hash, git SHA, timings, cost, so you can tell later exactly what produced a given number).

## Why

- **Cached**: responses are keyed on (provider, model, base_url, resolved prompt, params); re-runs are free, and a prompt edit invalidates only the affected samples.
- **Reproducible**: every run writes a manifest (spec identity hash, dataset hash, judge file hash, requested vs served model, git SHA, gradetrail version).
- **Multi-provider**: Anthropic, OpenAI, and any OpenAI-compatible endpoint (vLLM, local inference servers), through one provider abstraction with a shared retry/backoff policy.
- **Versioned judges**: LLM-as-judge prompts are separate, hashed files, not strings inlined in the spec.
- **Distributed**: the same spec runs unchanged on a local asyncio backend or a single-machine Ray cluster; the backend is a CLI flag, not a spec field.

## Prior art

Established tools cover much of this space: [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for academic benchmarks, [Inspect](https://inspect.ai-safety-institute.org.uk/) for safety evaluations, [promptfoo](https://github.com/promptfoo/promptfoo) for application testing. gradetrail differs in three specific choices: the response cache is keyed independently of the scorer, so changing how you grade re-scores for free without re-calling the API; every run's identity is a hash over spec, dataset content, and judge file, so two result sets are comparable exactly when their hashes match; and the local and Ray backends share one spec format and one per-sample pipeline, so distribution is an execution detail rather than a rewrite.

## Benchmark

500 samples, GSM8K test split, `claude-sonnet-4-6`, temperature 0, `max_tokens: 512`. Roughly 46.6k input / 56k output tokens per cold run.

| Run | Backend | In-flight requests | Wall time | Cost | Accuracy |
|---|---|---|---|---|---|
| Cold | local, concurrency 8 | 8 | 176.6s | $0.98 | 97.2%* |
| Re-score after scorer fix | local, all cached | n/a | 0.65s | $0.00 | 97.8% |
| Cold | ray, 8 workers | 64 (8 workers x concurrency 8) | 32.6s | $0.98 | 97.6% |
| Warm | ray, 8 workers | n/a | 4.15s | $0.00 | 97.6% |

\* 97.2% was measured before a scorer bug fix (see the failure audit below); 97.8% and 97.6% are post-fix.

Three things worth being explicit about, since a benchmark table invites the wrong conclusions if read too quickly:

1. **The ray-vs-local wall-time difference is a concurrency comparison, not framework magic.** Ray ran 64 requests in flight (8 workers, each applying `concurrency: 8` independently) while local ran 8, both against the same API rate limit. Ray's actual value here isn't raw speed, it's that the identical spec ran on a different execution backend with a one-flag change; nothing in the spec or scoring logic had to know or care.
2. **The $0.00 re-score row is the reason the cache exists.** After fixing a scorer regex bug (below), re-measuring all 500 samples against the corrected pattern took 0.65s and zero API calls, because the cache is keyed on (provider, model, prompt, params), not on the scorer. Changing how you score never invalidates what you already paid to generate.
3. **Temperature 0 does not guarantee identical outputs across runs.** The local run measured 97.8% and the Ray run measured 97.6% on what should be the same 500 completions; the difference is one sample, consistent with ordinary API-level nondeterminism at temperature 0, not a bug in either backend.

### Failure audit

97.2% understated the model's true accuracy by roughly 0.6 points due to measurement error, not model error. All 14 zero-score samples from the cold local run were inspected by hand:

- 3 were scoring artifacts: the model wrote a trailing period after the answer ("Answer: 25."), which the regex didn't tolerate. Fixed.
- 1 was a dataset-convention mismatch: the model gave a more precise decimal answer than GSM8K's integer ground truth. Left as-is rather than loosened into fuzzy matching.
- 2 were truncations at the 512-token output ceiling, indistinguishable from a wrong answer without inspecting the raw response.
- 8 were genuine model reasoning errors.

Only the last 8 reflect the model actually being wrong.

## Roadmap

- [x] Local async runner with caching and cost tracking
- [x] Ray execution backend
- [ ] Surface `stop_reason` in per-sample results, so truncation at the token ceiling is distinguishable from a wrong answer (motivated by the failure audit above)
- [ ] Fail fast when the first N samples all die with an identical fatal error, e.g. a missing API key, instead of logging 500 copies of it
- [ ] Multi-turn and tool-use evals
- [ ] HuggingFace dataset loader (dataset-specific conversion scripts exist today; no first-class loader in the spec yet)

## Design

See [docs/design/eval-spec.md](docs/design/eval-spec.md) for the spec schema and semantics. A writeup of debugging an intermittent SQLite WAL race under Ray is in [docs/blog/](docs/blog/).

MIT license.
