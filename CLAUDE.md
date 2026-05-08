---
title: CLAUDE
type: claude-config
permalink: vllm-mlx-fork/claude
---

# CLAUDE.md — vllm-mlx-fork

> Project-specific context only. Global settings live in `~/.claude/CLAUDE.md`.

> __Agent succession:__ If `internal/docs/BOOTSTRAP-SUCCESSION.md` exists, complete it before any other work. It will walk you through the entity transition.

---

## Project Overview

__Name__: vllm-mlx-fork  
__Type__: Forked inference server and Apple Silicon multimodal backend  
__Status__: Active  
__Primary language(s)__: Python, Markdown, shell scripts

This workspace is the working fork of `vllm-mlx` with custom backend patches, Apple Silicon validation, client compatibility evidence, and a large local-private documentation surface under `internal/docs/`.

---

## Project Layout

- `scripts/` — launchers, benchmarking utilities, and operator helpers
- `docs/` — public docs safe to push
- `internal/docs/` — private state, roadmap, benchmark exports, planning packets, and local-only notes
- `CODEX.md`, `CLAUDE.md`, `GEMINI.md` — runtime-specific project configs
- `AGENTS.md` — router file only

---

## Stack & Key Dependencies

- Python `3.11.x` via `mise`
- `uv` for Python environments and tool installs
- `mlx`, `mlx-vlm`, and this `vllm-mlx` fork for Apple Silicon multimodal serving
- Hugging Face Hub tooling for model publish and validation work
- local-private benchmark and planning artifacts under `internal/docs/`

---

## How to Run

__Session start__

1. Read `internal/docs/current-state.md`
2. Read `internal/docs/project-state.md`
3. Read `internal/docs/roadmap.md`
4. Read `internal/docs/decisions.md` if you need the decision trail

__Serve a multimodal model__

```bash
scripts/serve_profile.sh mllm-default <model>
```

__Serve for client/app interop__

```bash
scripts/serve_client_profile.sh generic-mllm <model>
```

__Benchmark tooling__

```bash
python scripts/gui_model_published_benchmark.py --help
```

__Environment / publish sanity__

```bash
hf auth whoami
```

Use targeted verification commands for the file or workflow you changed. There is no single always-run repo-wide test command.

---

## Architecture Decisions

Reference `internal/docs/decisions.md` for the structured log.

Key live decisions:

- canonical roadmap and planning entity is `vllm-mlx-fork-claude`
- roadmap uses schema v1.0 with `VLM-*` task IDs and `MIL-*` milestones
- public positioning is proof-first Apple Silicon multimodal backend work, not claim-first performance marketing

Related planning docs:

- `internal/docs/roadmap.md`
- `internal/docs/specs/apple-silicon-gui-and-vlm-positioning-strategy-2026-03-21.md`

---

## Project Conventions

- Conventional commits
- ISO 8601 dates everywhere
- underscores for emphasis, dashes for bullets, no asterisk-based emphasis
- public docs belong in `docs/`; sensitive and operator-facing context belongs in `internal/docs/`
- no AI attribution lines in commits or docs

---

## Known Gotchas

- `internal/docs/current-state.md` is the session-start file; do not look for `internal/docs/current.md`
- `internal/docs/` contains local-only paths and private coordination context; do not mirror that content into `docs/`
- multimodal packaging must keep root `processor_config.json` and a visible `tokenizer_config.json["chat_template"]`
- some older `mlx-community` repos are legacy or stale and should not be assumed to be authoritative roots
- `http://localhost:8012/v1` is intentionally reserved in this workspace; use parallel ports for experiments

---

## Agents for This Project

- __Primary working entity__: `vllm-mlx-fork-claude`
- __Maintained runtime configs__: `vllm-mlx-fork-claude`, `vllm-mlx-fork-codex`, `vllm-mlx-fork-gemini`
- __Primary runtime__: Claude
- __Operator__: `operator-swaylen`

For interagent memos, write to `~/dev/interagent/inbox/incoming/`.

---

## Out of Scope

- frontend application implementation, except backend compatibility support
- speculative benchmark claims without stored evidence
- writing sensitive coordination details into `docs/` or other public files

---

## Inter-Agent Communication

__Inter-agent memos__: When drafting memos intended for other agents, write to `~/dev/interagent/inbox/incoming/`. For standalone memos, create a flat file. For threads or memos with attachments, create a subfolder. Do not write directly to other agents' project directories.

---

## Public vs Internal Docs

`docs/` and root-level public docs are __public__.
The `internal/` directory is __private__ and local-only.

__Content rules__:

- __Public files__: must not contain agent or entity IDs, memo IDs, operator names, or interagent protocol text
- __Private files__: full context is allowed, but keep it inside `internal/docs/` or other ignored locations

__Update workflow__:

1. Write the full-context version in `internal/docs/`
2. Derive the public version in `docs/`
3. Never reverse that flow
4. Audit public docs before any push

---

## AI Attribution

No AI attribution in git commits or project docs. Do not add `Co-Authored-By` lines for any AI tool.

---

## Identity and Memory

- __Canonical entity ID__: `vllm-mlx-fork-claude`
- __Predecessor__: `vllm-mlx-fork-codex`
- __Operator ID__: `operator-swaylen`
- __Memory writes__: use the canonical entity ID for shared project memory and state writes
- __Legacy note__: older memo traffic and roadmap frontmatter may still reference `vllm-mlx-fork-codex`; treat as predecessor context, not the active entity

---

_Last updated: 2026-03-25_
