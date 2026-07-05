# GitHub labels & issue triage (maintainer checklist)

Phase 5 Task 5.2 — onboarding funnel setup. Run these steps once with a
maintainer-permissions account. Anything tagged "manual" needs the GitHub
web UI; everything else has a copy-pasteable `gh` command.

> All `gh` commands assume `gh auth login` has been run and the current
> working directory is the repo (so `--repo` is inferred).

## 1. Create labels (if they don't already exist)

| Label                | Color hex   | Description                          |
| -------------------- | ----------- | ------------------------------------ |
| `good first issue`   | `#7057ff`   | Good for newcomers                   |
| `help wanted`        | `#008672`   | Extra attention is needed            |
| `priority: critical` | `#b60205`   | Drop everything                      |
| `priority: high`     | `#d93f0b`   | High-priority follow-up              |
| `priority: medium`   | `#fbca04`   | Normal priority                      |
| `priority: low`      | `#0e8a16`   | Nice-to-have                         |

> The GitHub "good first issue" and "help wanted" colors above are the
> conventional ones documented at
> <https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests/encouraging-helpful-contributions-to-your-project-with-labels>.

Run:

```bash
gh label create "good first issue"   --color 7057ff --description "Good for newcomers" --force
gh label create "help wanted"        --color 008672 --description "Extra attention is needed" --force
gh label create "priority: critical" --color b60205 --description "Drop everything"         --force
gh label create "priority: high"     --color d93f0b --description "High-priority follow-up" --force
gh label create "priority: medium"   --color fbca04 --description "Normal priority"         --force
gh label create "priority: low"      --color 0e8a16 --description "Nice-to-have"            --force
```

The `--force` flag updates the label if it already exists (idempotent).

## 2. Tag existing issues `good first issue`

Spec called out #28, #30, #38. Before tagging, **verify each is still
open** — issue #30 (Fix PyPI logo) was already resolved in commit
`8b68ea4f`, so it should be skipped.

```bash
# Verify status first
gh issue view 28 --json state,title
gh issue view 30 --json state,title   # likely closed
gh issue view 38 --json state,title

# Then tag the still-open ones
gh issue edit 28 --add-label "good first issue"
gh issue edit 38 --add-label "good first issue"
# Skip 30 — already merged
```

## 3. Create new newcomer-friendly issues

The spec called for 5–7 new `good first issue` tickets. Below are
ready-to-create drafts. Run each `gh issue create` to file them, or
adapt the bodies in the web UI.

### Issue 1 — Add type hints to `multimind/models/openai_model.py`

```bash
gh issue create \
  --title "Add type hints to multimind/models/openai_model.py" \
  --label "good first issue,priority: low" \
  --body "$(cat <<'EOF'
**Background.** Most of `multimind/models/` is partially typed. As a first step
toward enabling mypy in pre-commit, we'd like every public method in
`openai_model.py` to have full type hints on parameters and return values.

**Acceptance criteria.**
- [ ] Every public method (no leading `_`) has type hints on all params + return.
- [ ] `mypy multimind/models/openai_model.py --ignore-missing-imports` is clean.
- [ ] `make test` still passes.

**Pointer.** See `multimind/_lazy.py` for the in-house style we're aiming for.
EOF
)"
```

### Issue 2 — Add docstrings to all public methods in `multimind/agents/`

```bash
gh issue create \
  --title "Add docstrings to all public methods in multimind/agents/" \
  --label "good first issue,priority: low" \
  --body "$(cat <<'EOF'
**Background.** `multimind/agents/agent.py`, `agent_loader.py`, and
`agent_registry.py` have several public methods without docstrings.

**Acceptance criteria.**
- [ ] Every public class + method has a Google-style docstring (see CONTRIBUTING.md).
- [ ] Docstrings include at least one usage example.
- [ ] `make lint` still passes.
EOF
)"
```

### Issue 3 — Add example: Using MultiMind with Ollama for local AI chat

```bash
gh issue create \
  --title "Add example: Using MultiMind with Ollama for local AI chat" \
  --label "good first issue,priority: medium" \
  --body "$(cat <<'EOF'
**Background.** Ollama works out of the box via HTTP (see README "Ollama users"
callout), but we don't have a dedicated example demonstrating it.

**Acceptance criteria.**
- [ ] New file: `examples/cli/ollama_chat.py` that demonstrates point-and-chat
      against a local Ollama instance.
- [ ] README snippet linking to the example.
- [ ] Works against the default `http://localhost:11434` endpoint.
EOF
)"
```

### Issue 4 — Add example: Basic RAG with PDF documents

```bash
gh issue create \
  --title "Add example: Basic RAG with PDF documents" \
  --label "good first issue,priority: medium" \
  --body "$(cat <<'EOF'
**Background.** `examples/rag/fluent_rag_example.py` covers the fluent API
with synthetic text. We need a companion that ingests real PDFs end-to-end.

**Acceptance criteria.**
- [ ] New file: `examples/rag/pdf_rag_example.py`.
- [ ] Loads PDFs via `multimind.document_loader` (the `[documents]` extra).
- [ ] Demonstrates chunking, embedding, retrieval, and answer generation.
- [ ] Includes a 3–5 line snippet in the README "Examples" section.
EOF
)"
```

### Issue 5 — Improve error messages when API keys are missing

```bash
gh issue create \
  --title "Improve error messages when API keys are missing" \
  --label "good first issue,priority: medium" \
  --body "$(cat <<'EOF'
**Background.** When a user instantiates `OpenAIModel(...)` without
`OPENAI_API_KEY` set, the error today is a low-level HTTP 401 or a
\`ValueError: \"No API key provided\"\` — neither of which tell newcomers
where to set the key.

**Acceptance criteria.**
- [ ] All `*Model` classes raise a friendly error referencing the env-var
      name and a doc link when the API key is missing.
- [ ] Test in `tests/test_models.py` (or equivalent) covering the new
      error message.
EOF
)"
```

### Issue 6 — Add `--version` flag to CLI

```bash
gh issue create \
  --title "Add --version flag to CLI" \
  --label "good first issue,priority: low" \
  --body "$(cat <<'EOF'
**Background.** Today \`multimind --version\` doesn't work. The CLI lives in
\`multimind/cli/__main__.py\`.

**Acceptance criteria.**
- [ ] \`multimind --version\` prints \`multimind-sdk X.Y.Z\` (read from
      \`multimind.__version__\`) and exits 0.
- [ ] Test added to \`tests/test_cli.py\` (create if missing).
EOF
)"
```

### Issue 7 — Add Python 3.13 to CI test matrix

> **NOTE:** This was already done in Phase 3. Verify in
> `.github/workflows/ci.yml` (`test-core.strategy.matrix.python-version`).
> If 3.13 is in the matrix and passing, **close this issue with a comment
> pointing at the relevant CI run**. If it's still missing, file the issue
> for real.

```bash
gh issue create \
  --title "Add Python 3.13 to CI test matrix" \
  --label "good first issue,priority: low" \
  --body "Python 3.13 was released October 2024. Add it to the matrix in .github/workflows/ci.yml and confirm the suite passes."
```

## 4. Wire labels into the issue templates (optional, recommended)

If you use issue templates (`.github/ISSUE_TEMPLATE/*.yml`), set sensible
defaults so reporters land on the right priority bucket:

- `bug.yml` → default `labels: [bug, priority: medium]`
- `feature.yml` → default `labels: [enhancement, priority: low]`
- `docs.yml` → default `labels: [documentation, priority: low]`

## Done — verify

```bash
gh label list | grep -E "(good first issue|help wanted|priority:)"
gh issue list --label "good first issue"
```

You should see all 6 labels and at least 5 newcomer-friendly issues.
