# Email Triage OpenEnv Environment

This project implements a reinforcement learning environment using the OpenEnv specification.

## Description

The environment simulates an email triage workflow where an AI agent must:

- classify email category
- detect priority level
- identify spam messages

It includes three tasks with increasing difficulty:

1. Easy — classify category
2. Medium — classify category and priority
3. Hard — classify category, priority, and spam detection

## Observation Space

Agent receives:

- subject
- body
- sender_type

## Action Space

Agent predicts:

- category
- priority
- spam flag

## Reward Function

Reward ranges between:

0.0 to 1.0

Partial reward is given for correct predictions.

## Tasks

- task_easy
- task_medium
- task_hard

## Run baseline inference

```bash
python inference.py