# QVEX

QVEX is a modular quantitative trading system with a strict separation
between the trading Core and the external Control Plane.

## Architecture

```text
Telegram Bot / Mini App
        |
        v
QVEX Control Plane
        |
        | trading_control.json
        | control_plane_state.json
        v
QVEX Core
        |
        v
Hyperliquid
```

## Core

The Core is the single owner of Hyperliquid execution.

Production entrypoint:

```text
core/engine.py
```

Production engine:

```text
HyperliquidSwingBot
```

The Core owns:

- market and data processing
- strategy execution
- risk controls
- order execution
- position reconciliation
- panic handling
- model loading
- canonical telemetry

## Control Plane

The Control Plane provides the external operator interface.

It communicates with the Core through the canonical IPC contract:

```text
data/trading_control.json
data/control_plane_state.json
```

The Control Plane does not own the Hyperliquid private key.

## IPC

Canonical production IPC modules:

```text
core/ipc/control.py
core/ipc/schema.py
core/ipc/state.py
```

Canonical telemetry:

```text
data/control_plane_state.json
```

Canonical trading control:

```text
data/trading_control.json
```

## Model

The production meta-model is loaded from:

```text
data/meta_model.json
```

## Telegram

Telegram integration lives in the separate Control Plane repository:

```text
D:/Projects/qvex-control-plane
```

Telegram must not contain or own the Hyperliquid private key.

## Repository Boundary

This repository contains the QVEX Core.

Legacy execution stacks, historical snapshots, temporary backups and
obsolete AIQuant artifacts are intentionally excluded from the
production repository.
