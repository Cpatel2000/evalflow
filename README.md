# evalflow

A distributed LLM evaluation harness that treats evals like tests: declarative, cached, reproducible.

> Early development. The spec format below is stable; the runner lands next.

## Quickstart

```bash
pip install evalflow  # not yet published
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

Run it:

```bash
evalflow run gsm8k_subset.yaml
```

## Why

- **Cached**: responses keyed on (model, prompt, params); re-runs are free, prompt edits invalidate only affected samples.
- **Reproducible**: every run writes a manifest (spec hash, dataset hash, judge version, git SHA).
- **Multi-provider**: Anthropic, OpenAI, any OpenAI-compatible endpoint (vLLM).
- **Versioned judges**: LLM-as-judge prompts are hashed artifacts, not inline strings.
- **Distributed**: same spec runs locally or across Ray workers with a config flag.

## Roadmap

- [ ] Local async runner with caching and cost tracking
- [ ] Ray execution backend
- [ ] Multi-turn and tool-use evals
- [ ] HuggingFace dataset loader

## Design

See [docs/design/eval-spec.md](docs/design/eval-spec.md) for the spec schema and semantics.

MIT license.
