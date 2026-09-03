# Authoring rules for agent plans

Calibration guide for writing the Agent Instructions section.
The mechanical rules (guardrail per step, eval before step, 1-3 checkpoints, error budget defaults) live in TEMPLATE.md; this file only calibrates judgment.
Grounded in "AI Engineering" (O'Reilly); the concepts themselves are assumed known.

## Evals

- Prefer the highest seam available: unit < integration < e2e < browser.
- Existing tests are the regression eval; every plan must run them.

## Guardrails

- Prefer commands the repo already has (typecheck, tests, lint) over new tooling.
- Chain order: types first, then implementation, then tests, then visual verification.
- For parallel waves, apply chain order per wave (types -> impl -> tests per lane), then a global barrier does integration and visual verification.

## Concurrency

- Waves are the unit of parallelism. A wave is a set of steps with no inter-dependency and no file conflict that can run concurrently.
- Leaf-first: start from tasks and files with zero `depends_on`, build the tree upward. If a dependency is uncertain, mark it as sequential (pessimistic).
- File-conflict rule: if two steps in the same wave touch the same file, they must be sequentialized into sub-waves or the file split. Detect with `grep` on the Files column.
- Degrees of freedom: narrow bridge (DB migration, schema, `graph`) = low freedom, exact script per worker; open field (UI copy, tests, `sequenceDiagram`) = high freedom, heuristic prompt. Match specificity to fragility.
- Chain order per wave: types -> implementation -> tests. Global barrier after the last wave does integration and visual verification.
- Error budget per wave: a failure in lane A isolates; lane B continues until the barrier, then the coordinator decides. Document `isolation` vs `fail-fast` per plan.
- Context scoping: each worker gets minimal context — its wave's files, key types, and one example test — not the full repo. Shared context is only merged at barriers.
- State is king: edges of the graph transport state. Keep the state schema (files, types, env) explicit per wave to avoid polluting shared context.

## Instructions vs context

- Instructions say WHAT; context (files to read, types, examples) says HOW.
- Compact instructions plus rich context beats long step-by-step prose, which is brittle.
- If a step needs more than two sentences to describe, split it.
- For parallel plans, scope context per wave and lane (its files + key types + one example test). Shared context merges only at barriers.

## Human-in-the-Loop

- Place checkpoints before structural changes and before merge.
- Every checkpoint must state exactly what the user is confirming.

## Error budget

- The defaults live in TEMPLATE.md; override them only with a reason stated in the plan.
