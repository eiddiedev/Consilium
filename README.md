# Consilium

> **Multi-specialty clinical decision support for complex chronic disease patients.**
> Three specialist agents. One TOPSIS-ranked decision. Full clinical transparency.

**Built for [Agents Assemble: The Healthcare AI Endgame](https://agents-assemble.devpost.com/)**

---

## The Problem

A patient with **heart failure + type 2 diabetes + chronic kidney disease** sees three specialists. Each recommends treatments based on their own guidelines — but the recommendations often **contradict each other**.

- Cardiology says: increase diuretics
- Nephrology says: hold diuretics (eGFR too low)
- Endocrinology says: continue Metformin
- Nephrology says: **stop** Metformin (eGFR <30 is a hard contraindication)

The primary care physician is left to **reconcile these conflicts manually**, with no systematic tool and no explainable reasoning chain.

## The Solution

**Consilium** is a multi-agent clinical decision support system that:

1. **Routes** patient data to 3 specialist sub-agents (cardiology, nephrology, endocrinology)
2. **Collects** structured recommendations with evidence levels, risk flags, and guideline citations
3. **Scores** recommendations using **TOPSIS** (multi-criteria ranking) across 4 clinical dimensions
4. **Explains** the ranking with a clinician-friendly decision summary

```
Patient FHIR Data
       │
       ▼
┌──────────────────┐
│  ASM Orchestrator │  ← Single A2A endpoint
└──────┬───────────┘
       │ in-process (ADK AgentTool)
       ├──────────────┬──────────────┐
       ▼              ▼              ▼
┌──────────┐  ┌──────────┐  ┌──────────────┐
│Cardiology│  │Nephrology│  │Endocrinology │
│  Agent   │  │  Agent   │  │    Agent     │
│(ACC/AHA) │  │ (KDIGO)  │  │   (ADA)     │
└────┬─────┘  └────┬─────┘  └──────┬──────┘
     │             │               │
     └─────────────┼───────────────┘
                   ▼
          ┌────────────────┐
          │  TOPSIS Scorer │
          │  4 dimensions  │
          └───────┬────────┘
                  ▼
         ┌────────────────┐
         │ Explain Decision│
         │ (NL summary)    │
         └────────────────┘
```

## TOPSIS Scoring Dimensions

| Dimension | Weight | Direction | Rationale |
|-----------|--------|-----------|-----------|
| **Evidence Level** | 30% | Higher = better | ACC/AHA Class, KDIGO Grade, ADA Level |
| **Patient Match** | 30% | Higher = better | How well the rec fits this patient's specific metrics |
| **Drug Interaction Risk** | 20% | Lower = better | Potential for adverse interactions with existing meds |
| **Guideline Priority** | 20% | Higher = better | How strongly the guideline recommends this action |

Weights are dynamically adjustable based on patient state (e.g., eGFR <30 boosts drug interaction risk weight).

## Clinical Example

**Patient:** 68M, LVEF 32%, eGFR 28, HbA1c 8.2%
**Meds:** Lisinopril, Metformin, Furosemide, Aspirin, Glipizide

**TOPSIS Ranking:**

| Rank | Specialty | Score | Key Recommendation |
|:----:|:---------:|:-----:|:-------------------|
| 🥇 | Nephrology | 0.581 | Stop Metformin, start SGLT2i, continue Lisinopril |
| 🥈 | Endocrinology | 0.542 | Stop Metformin, start SGLT2i, consider GLP-1 RA |
| 🥉 | Cardiology | 0.398 | Add beta-blocker + SGLT2i, consider ARNI switch |

**Consensus:** All 3 specialists agree on stopping Metformin (eGFR <30 contraindication) and starting SGLT2i (triple benefit for HF + CKD + T2DM).

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Agent Framework | [Google ADK](https://google.github.io/adk-docs/) |
| Protocol | [A2A (Agent-to-Agent)](https://google.github.io/A2A/) v1 |
| Platform | [Prompt Opinion](https://promptopinion.ai) |
| LLM | DeepSeek V3 (via LiteLLM) |
| Decision Engine | TOPSIS (custom implementation) |
| Data Standard | FHIR R4 |
| Context Propagation | SHARP Extension Specs |

## Quick Start

```bash
# Clone
git clone https://github.com/eiddiedev/Consilium.git
cd Consilium

# Setup
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env: set DEEPSEEK_API_KEY (or GOOGLE_API_KEY for Gemini)

# Run locally (browser UI)
adk web .

# Run as A2A server (for Prompt Opinion)
uvicorn orchestrator.app:a2a_app --host 0.0.0.0 --port 8003

# Verify
curl http://localhost:8003/.well-known/agent-card.json
```

## Project Structure

```
Consilium/
├── orchestrator/          # Main A2A endpoint — routes to sub-agents
│   ├── agent.py           # Orchestrator logic + TOPSIS + explain tools
│   └── app.py             # A2A server config + agent card
├── cardiology_agent/      # HF specialist (ACC/AHA 2022)
├── nephrology_agent/      # CKD specialist (KDIGO 2024)
├── endocrinology_agent/   # T2DM specialist (ADA 2025)
├── shared/                # ADK infrastructure (middleware, FHIR hooks, tools)
│   ├── app_factory.py     # A2A app builder
│   ├── fhir_hook.py       # SHARP context extraction
│   ├── middleware.py       # API key enforcement
│   └── tools/fhir.py      # FHIR R4 query tools
├── tools/                 # Decision engine
│   ├── topsis.py          # TOPSIS multi-criteria scorer
│   ├── score_tool.py      # ADK tool wrapper
│   └── explain_tool.py    # Decision explanation generator
├── data/
│   ├── patient_hf_t2dm_ckd.json   # Mock FHIR Bundle
│   └── guideline_weights.json     # Clinical guideline weights
└── tests/
    └── test_topsis.py     # TOPSIS unit tests (10/10 passing)
```

## Why AI + TOPSIS (Not Pure Rules)?

| | Rule Engine | Consilium (AI + TOPSIS) |
|---|---|---|
| **Free-text guidelines** | Can't parse "if EF <40% and eGFR >30, consider..." | LLM extracts structured signals from natural language |
| **Dynamic weights** | Fixed rules | Weights shift based on patient state (eGFR, LVEF, HbA1c) |
| **Multi-criteria tradeoff** | Hard-coded priorities | TOPSIS mathematically ranks across 4 dimensions |
| **Explainability** | "Rule #47 triggered" | "Nephrology ranked #1 due to superior patient match (1.00) — eGFR 28 makes drug interaction risk the critical dimension" |
| **New guidelines** | Rewrite rules | Update prompt + weight table |

## Evidence Sources

- **ACC/AHA 2022** Heart Failure Guidelines ([JACC](https://www.jacc.org/doi/10.1016/j.jacc.2021.12.012))
- **KDIGO 2024** CKD Guidelines ([KDIGO](https://kdigo.org/guidelines/ckd-evaluation-and-management/))
- **ADA 2025** Standards of Care in Diabetes ([Diabetes Care](https://diabetesjournals.org/care/issue/48/Supplement_1))

## Disclaimer

Consilium is a **clinical decision support tool**, not a clinical decision maker. All outputs include evidence citations, risk flags, and confidence scores. Final treatment decisions rest with the treating physician.

## License

MIT

---

*Built for [Agents Assemble: The Healthcare AI Endgame](https://agents-assemble.devpost.com/) by Prompt Opinion / Darena Health.*
