# PRD template

<!-- Template for the "Docs for Humans" section of a plan.
This file only defines the sections; process lives in SKILL.md. -->

## Diagrams

Use Mermaid diagrams (fenced `` ```mermaid `` code blocks) for structural or sequential concepts: architecture, data flows, control flow, state machines, entity relationships, sequences, decision trees.
Lead with the diagram and keep the accompanying text short.
One concept per diagram.
Fall back to plain prose when the concept does not map cleanly onto a diagram - a trivial feature may have no diagrams at all.
Diagrams live in Docs for Humans only; Agent Instructions stays prose and tables.

## Problem Statement

The problem that the user is facing, from the user's perspective.

## Solution

The solution to the problem, from the user's perspective.
If the proposed behavior is a flow or sequence, lead with a Mermaid flowchart or state diagram.

## User Stories

A LONG, numbered list of user stories. Each user story should be in the format of:

1. As an <actor>, I want a <feature>, so that <benefit>

<user-story-example>
1. As a mobile bank customer, I want to see balance on my accounts, so that I can make better informed decisions about my spending
</user-story-example>

This list of user stories should be extremely extensive and cover all aspects of the feature.

## Implementation Decisions

A list of implementation decisions that were made. This can include:

- The modules that will be built/modified
- The interfaces of those modules that will be modified
- Technical clarifications from the developer
- Architectural decisions
- Schema changes
- API contracts
- Specific interactions
- Dependency and concurrency: if parallel, include a Mermaid DAG of waves and a file-conflict analysis; if sequential, note why parallelism is not applicable

Express decisions as Mermaid diagrams when they map cleanly:

- Modules and architectural decisions: `graph`
- Schema changes: `erDiagram`
- API contracts and specific interactions: `sequenceDiagram`
- State machines: `stateDiagram-v2`

Do NOT include specific file paths or code snippets. They may end up being outdated very quickly.

Exception: if a prototype produced a snippet that encodes a decision more precisely than prose can (state machine, reducer, schema, type shape), inline it within the relevant decision and note briefly that it came from a prototype. Trim to the decision-rich parts - not a working demo, just the important bits.

## Testing Decisions

A list of testing decisions that were made. Include:

- A description of what makes a good test (only test external behavior, not implementation details)
- Which modules will be tested
- Prior art for the tests (i.e. similar types of tests in the codebase)

## Out of Scope

A description of the things that are out of scope for this PRD.

## Further Notes

Any further notes about the feature.
