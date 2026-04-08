---
title: Support Env
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# AI Customer Support Environment

This project simulates a real-world customer support system.

## Features
- Ticket classification
- Priority assignment
- Resolution suggestion
- Response generation

## Tasks
- Easy, Medium, Hard tasks
- Real-world scenarios

## Evaluation
- Custom grader with score (0–1)
- Partial scoring system

## Advanced Features

- Stateful environment with action history tracking
- Reward shaping with penalties and bonuses
- Multi-intent complex ticket scenarios
- Human-like response evaluation (tone + keywords)
- Real-world customer support workflow simulation

## Environment Design

- Implements core RL interface: `reset()`, `step()`, `state()`
- Each episode evaluates a full decision cycle
- Reward system provides partial scoring (0–1 range)
- Deterministic grading ensures reproducibility

## Task Complexity

- Easy: Basic classification
- Medium: Multi-field decision (priority + resolution)
- Hard: Multi-intent and ambiguous tickets

## Why This Matters

This environment simulates realistic customer support decision-making, enabling training and evaluation of AI agents in practical business scenarios.