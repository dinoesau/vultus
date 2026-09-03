---
name: coding-guide
description: Behavioral coding guidelines that reduce common LLM mistakes — think before coding, keep it simple, make surgical changes, and define verifiable goals. Use when writing new code, editing existing code, fixing bugs, or whenever you need to avoid over-engineering, unnecessary refactoring, or vague execution plans.
---

# Coding Guide

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

State assumptions explicitly. If uncertain, ask. If multiple interpretations exist, present them — don't pick silently. If a simpler approach exists, say so. Push back when warranted.

## 2. Simplicity First

Minimum code that solves the problem. Nothing speculative. No features beyond what was asked. No abstractions for single-use code. No "flexibility" that wasn't requested. If 200 lines could be 50, rewrite it. Ask: "Would a senior engineer say this is overcomplicated?"

## 3. Surgical Changes

Touch only what you must. Clean up only your own mess.

- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.
- Remove imports/variables/functions that YOUR changes made unused.

**The test:** Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

Define success criteria. Loop until verified.

- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"

For multi-step tasks, state a brief plan:

```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
```

## Examples

See [EXAMPLES.md](EXAMPLES.md) for real-world before/after examples of each principle.