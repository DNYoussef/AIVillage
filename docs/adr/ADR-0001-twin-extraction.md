# ADR-0001: Twin Extraction Strategy

## Status
Draft

## Context
The project envisions an optional "twin extraction" process that snapshots an agent's
state into a portable form for offline analysis. This mechanism does not yet exist
in the repository.

## Decision
We will design the twin extraction flow as a separate module that can serialize
an agent's configuration, knowledge base pointers and latest capabilities. The ADR
serves as a placeholder until an implementation is ready.

## Consequences
Documentation and interfaces can reference the extraction concept, but any code
that calls it should mark the functionality as experimental or raise
`NotImplementedError` until the module is implemented.
