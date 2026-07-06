# Design: Eval Spec Schema (v0.1)

Status: locked for v0.1. Changes require updating this doc first.
Location in repo: `docs/design/eval-spec.md`

## Goals

- A researcher can write a working spec in under 10 lines.
- Everything that affects results is IN the spec (or pinned by it), so spec hash + dataset hash = reproducible run.
- Deterministic and judge-based scoring are first-class, judges are versioned.
- The spec is execution-agnostic: nothing in it says local vs Ray.

Non-goals for v0.1: multi-turn conversations, tool use evals, image inputs, pass@k sampling strategies. Listed in README roadmap instead.

## The spec, by example

Minimal spec (the one that goes in the README quickstart):

```yaml
# gsm8k_subset.yaml
name: gsm8k-subset
dataset:
  path: data/gsm8k_subset.jsonl      # local JSONL, one sample per line
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

Full spec with every field:

```yaml
name: gsm8k-subset            # required, [a-z0-9-], used in run IDs and file names
description: >                # optional
  GSM8K 200-sample subset, numeric answer extraction.

dataset:
  path: data/gsm8k_subset.jsonl   # required. JSONL. Each line is one sample (a flat JSON object).
  id_field: id                    # optional, default "id". Falls back to line number if absent.
  limit: 200                      # optional, take first N samples after load
  shuffle_seed: 42                # optional, deterministic shuffle before limit

prompt: |                         # required. Jinja2 template rendered per sample.
  {{ question }}                  # any field from the sample JSON is available

model:
  provider: anthropic             # required: anthropic | openai | openai_compatible
  name: claude-sonnet-4-6         # required, passed through to provider
  base_url: null                  # required for openai_compatible, forbidden otherwise
  params:                         # optional, all sampling params live here
    max_tokens: 1024              # default 1024
    temperature: 0.0              # default 0.0
    system: null                  # optional system prompt (string or template)

scorer:
  # exactly one of the three types below

  # 1) exact string match against a sample field
  type: exact
  target_field: answer            # sample field holding the expected value
  normalize: [strip, lower]       # optional list: strip, lower, collapse_whitespace

  # 2) regex, template-rendered per sample so it can embed expected values
  type: regex
  pattern: 'Answer:\s*{{ answer }}\s*$'
  flags: [MULTILINE]              # optional: IGNORECASE, MULTILINE, DOTALL

  # 3) LLM-as-judge
  type: judge
  judge_prompt: judges/correctness_v2.yaml   # required, path to versioned judge file
  model:                                     # judge model, same shape as top-level model
    provider: anthropic
    name: claude-sonnet-4-6
    params:
      temperature: 0.0
  samples: 1                      # optional, judge calls per response, score = mean

run:                              # optional block, execution knobs (not part of results identity)
  concurrency: 8                  # default 8, max in-flight requests
  max_retries: 5                  # default 5, exponential backoff with jitter on 429/5xx/timeouts
  timeout_s: 120                  # per-request timeout
```

## Judge prompt files

Judges are separate versioned YAML files so the same judge can be reused and its version recorded in the manifest:

```yaml
# judges/correctness_v2.yaml
version: 2
output: score_0_1        # v0.1 supports: score_0_1 | binary
prompt: |
  You are grading a model response for correctness.

  Question: {{ question }}
  Expected answer: {{ answer }}
  Model response: {{ response }}

  Reply with only a JSON object: {"score": <0 or 1>, "reason": "<one sentence>"}
```

Rules:
- The judge file's sha256 goes in the run manifest. Changing a judge means bumping `version` and ideally the filename.
- Judge output must be parseable JSON, the harness retries once with a "reply with only JSON" nudge, then marks the sample as `judge_error` (never silently scores 0).

## Semantics that must hold (write tests for these)

1. **Identity.** The results-relevant identity of a run is: sha256 of canonical JSON of (rendered spec minus `run` block, dataset content hash, judge file hash if present). The `run` block (concurrency, retries) never affects identity.
2. **Cache key** is per-sample: sha256 of canonical JSON of (provider, model name, base_url, resolved prompt string, params). Changing the prompt template invalidates exactly the affected samples, nothing else.
3. **Template strictness.** Referencing a missing sample field in `prompt` or `pattern` is a validation error at load time (fail fast on sample 0, not mid-run). This extends to `scorer.target_field` for the exact scorer, which must also exist in sample 0.
4. **Validation errors** name the field and the fix: `scorer.pattern: invalid regex (unbalanced parenthesis at position 12)`. Pydantic models with custom messages.
5. **Every sample terminates** in exactly one state: `scored`, `provider_error`, or `judge_error`. Failure states carry the error detail. Summary reports counts of each.
6. **Determinism caveat, documented not hidden:** temperature 0 does not guarantee identical outputs across API calls. The manifest records everything we control; the README says this out loud.

## Python shape (spec.py sketch)

```python
from __future__ import annotations
from pydantic import BaseModel

class DatasetSpec(BaseModel, frozen=True):
    path: str
    id_field: str = "id"
    limit: int | None = None
    shuffle_seed: int | None = None

class ModelParams(BaseModel, frozen=True):
    max_tokens: int = 1024
    temperature: float = 0.0
    system: str | None = None

class ModelSpec(BaseModel, frozen=True):
    provider: Literal["anthropic", "openai", "openai_compatible"]
    name: str
    base_url: str | None = None
    params: ModelParams = ModelParams()

# ScorerSpec is a discriminated union on "type": ExactScorer | RegexScorer | JudgeScorer

class RunSpec(BaseModel, frozen=True):
    concurrency: int = 8
    max_retries: int = 5
    timeout_s: float = 120.0

class EvalSpec(BaseModel, frozen=True):
    name: str
    description: str | None = None
    dataset: DatasetSpec
    prompt: str
    model: ModelSpec
    scorer: ExactScorer | RegexScorer | JudgeScorer
    run: RunSpec = RunSpec()

    def identity_hash(self) -> str: ...
```

## Open questions (decide before week 2, not blockers for week 1)

- Should `model` accept a list to sweep multiple models in one spec, or is that a CLI concern (`--model-override`)? Leaning CLI override to keep the spec single-purpose.
- JSONL only for v0.1, or also HuggingFace datasets by name? Leaning JSONL only, HF loader on the roadmap.
