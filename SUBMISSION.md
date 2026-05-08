# Consilium — Devpost Submission Materials

## Project Name

Consilium — Multi-Specialty Clinical Decision System

## Tagline

A2A clinical agents that reconcile cardiology, nephrology, and endocrinology recommendations with deterministic TOPSIS ranking.

## Short Description

Consilium helps clinicians manage complex chronic disease patients by combining specialist LLM agents with deterministic clinical scoring. It connects to Prompt Opinion as an A2A agent, accepts FHIR context, generates specialist recommendations, and returns an explainable ranked action plan.

## Inspiration

Patients with heart failure, chronic kidney disease, and diabetes often receive conflicting recommendations from different specialists. Primary care clinicians must manually reconcile guideline conflicts, medication safety thresholds, and patient-specific risks. Consilium was built to make that reconciliation structured, explainable, and safer.

## What It Does

- Connects to Prompt Opinion as a BYO A2A agent.
- Reads FHIR/SHARP context when available.
- Calls cardiology, nephrology, and endocrinology specialist agents.
- Validates specialist JSON outputs.
- Uses deterministic TOPSIS scoring to rank recommendations.
- Returns top pick, conflicts resolved, citations, and a safety disclaimer.
- Refuses to hallucinate when patient context is insufficient.

## How It Was Built

- Google ADK agents
- A2A JSON-RPC interface
- Prompt Opinion FHIR context extension
- DeepSeek V4 Flash via LiteLLM
- FastAPI/Starlette on Google Cloud Run
- React/Vite frontend
- Deterministic TOPSIS ranking engine

## Challenges

The hardest part was making the system clinically credible without pretending it is autonomous medical software. We separated LLM generation from deterministic ranking, added FHIR context handling, protected token logging, and implemented fallbacks for specialist failures and insufficient patient data.

## Accomplishments

- Working Cloud Run A2A backend
- Prompt Opinion FHIR extension support
- Three specialist agents with structured JSON contracts
- Deterministic scoring and safety overrides
- Frontend demo with real A2A backend integration
- Synthetic FHIR patient cases and regression tests

## What Is Next

- Multi-round specialist negotiation
- More guideline domains and specialty agents
- Broader synthetic FHIR case library
- Clinical workflow validation with physicians
- Streaming progress events from backend to frontend

## Built With

Google ADK, A2A, Prompt Opinion, FHIR R4, SMART scopes, DeepSeek V4 Flash, LiteLLM, FastAPI, Cloud Run, React, Vite, TOPSIS.

## Demo Links To Fill In

- Devpost video:
- GitHub repository:
- Cloud Run A2A endpoint: `https://consilium-1085209557278.us-central1.run.app`
- Prompt Opinion marketplace listing:

## Submission Checklist

- Video is under 3 minutes.
- Video shows the project functioning inside Prompt Opinion.
- Agent is published/discoverable in Prompt Opinion Marketplace.
- Demo uses only synthetic or de-identified data.
- GitHub repo does not include `.env`, API keys, node_modules, build artifacts, or local ADK state.
- README describes AI Factor, Potential Impact, and Feasibility clearly.
