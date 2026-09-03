---
name: git-commit
description: 'Create one or multiple conventional commits by grouping changed files into domain-based groups (e.g. source code, tests, docs, config, infrastructure). Each group gets its own commit with an auto-detected type, scope, and message. Use when user asks to commit changes, create a git commit, or mentions "/commit".'
license: MIT
allowed-tools: Bash
---

# Git Commit — Domain-Based Multi-Commit

## Overview

Instead of a single commit, analyze **all working tree changes**, group them by **domain** (logical area/concern), and create one conventional commit per domain. Each commit gets its own type, scope, and summary based on its group's changes.

## Conventional Commit Format

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

## Commit Types

| Type       | Purpose                        |
| ---------- | ------------------------------ |
| `feat`     | New feature                    |
| `fix`      | Bug fix                        |
| `docs`     | Documentation only             |
| `style`    | Formatting/style (no logic)    |
| `refactor` | Code refactor (no feature/fix) |
| `perf`     | Performance improvement        |
| `test`     | Add/update tests               |
| `build`    | Build system/dependencies      |
| `ci`       | CI/config changes              |
| `chore`    | Maintenance/misc               |
| `revert`   | Revert commit                  |

## Breaking Changes

```
# Exclamation mark after type/scope
feat!: remove deprecated endpoint

# BREAKING CHANGE footer
feat: allow config to extend other configs

BREAKING CHANGE: `extends` key behavior changed
```

## Domain Classification Strategy

Detect the domain of each changed file using this priority-ordered heuristic:

### 1. By directory / file path

| Pattern                    | Domain         | Default type |
| -------------------------- | -------------- | ------------ |
| `src/`, `lib/`, `app/`     | `source`       | `feat`\|`fix`|
| `tests/`, `*test*`, `*spec*` | `tests`      | `test`       |
| `docs/`, `*.md`            | `docs`         | `docs`       |
| `infra/`, `ops/`, `deploy/` | `infra`       | `ci`         |
| `scripts/`, `Makefile`     | `build`        | `build`      |
| `config/`, `*.{yml,yaml,toml,ini,cfg}` | `config` | `chore` |
| `migrations/`              | `migrations`   | `feat`       |
| `styles/`, `*.css`         | `styles`       | `style`      |
| `.github/`, `.circleci/`   | `ci`           | `ci`         |
| `docker*`, `Dockerfile`    | `docker`       | `build`      |
| `package.json`, `requirements.*`, `go.mod` | `deps` | `build` |

### 2. Re-classify based on diff content

After initial path-based classification, inspect diffs to refine:

- If a `source` file is **only** adding/removing comments or whitespace → reclassify as `docs` or `style`
- If a `source` file is **only** renaming symbols or restructuring without behaviour change → reclassify as `refactor`
- If all changes are bug-oriented (fix, correct, handle edge case in diff) → use `fix`

### 3. Files to NEVER commit

Warn and skip: `.env`, `credentials.json`, `*.key`, `*.pem`, `.env.*`, `secrets.*`

## Workflow

### 1. Collect All Changes

```bash
# View full working tree state
git status --porcelain

# Get diffs for all unstaged/uncommitted files
git diff HEAD
```

### 2. Classify Into Domains

Group each changed file into a domain based on the strategy above.

Example groups for a typical change set:

```
source:  src/handlers/expenses.py, src/models/budget.py
tests:   tests/test_expenses.py, tests/test_budget.py
docs:    README.md, docs/api.md
config:  template.yaml, .env.example
```

### 3. Commit Per Domain (Ordered)

Commit domains in this order to maintain a clean history:

1. `config` / `build` / `ci` / `deps` / `docker` — foundations first
2. `refactor` — structural changes before new behaviour
3. `tests` — tests before the code they cover (shows red→green)
4. `source` — the actual feature/fix
5. `docs` / `styles` — polish last

**If there is only one domain**, produce a single commit as normal.

**If a domain has no changes**, skip it.

```bash
# Stage and commit one domain at a time
git add src/handlers/expenses.py src/models/budget.py
git commit -m "<type>(source): <description>"
```

**After each commit**, re-check with `git status --porcelain` to confirm only the right files were staged.

### 4. Generate Per-Domain Commit Message

For each domain, analyse its diff (`git diff --staged`) to determine:

- **Type** based on the domain's default type, refined by diff content
- **Scope** = the domain name (or sub-area within it, e.g. `expenses` for `src/handlers/expenses.py`)
- **Description**: imperative, <72 chars, focused on what this domain group accomplishes

### 5. Present Plan Before Committing

Before running any `git add`, show the user the proposed commit plan:

```
The following commits will be created:

1. test: add budget validation tests
   - tests/test_budget.py
   - tests/fixtures/budget.json

2. feat(expenses): add monthly expense summary endpoint
   - src/handlers/expenses.py
   - src/models/expense.py

3. docs: update README with new endpoint
   - README.md

Proceed? [Y/n]
```

If user declines, abort. If user provides feedback, adjust grouping accordingly.

## Git Safety Protocol

- NEVER update git config
- NEVER run destructive commands (--force, hard reset) without explicit request
- NEVER skip hooks (--no-verify) unless user asks
- NEVER force push to main/master
- If a commit fails due to hooks, fix the issue and create a NEW commit (don't amend)
- **If any commit in the sequence fails**, stop and report which commit failed. Do not continue the sequence until the issue is resolved.

## Best Practices

- One domain per commit (not one logical change — one domain's changes)
- If a single file touches multiple concerns, put it in the dominant domain (the one with >80% of the diff)
- Keep each commit's description under 72 characters
- Use `Refs #issue` in footers, never in the title line
- After the sequence completes, run `git log --oneline` to verify the result