# Consilium — 3-Minute Demo Script

## Segment 1: The Problem (0:00–0:30)

**[Screen: title card or patient scenario diagram]**

> "Meet Mr. Chen — 68 years old, with heart failure, type 2 diabetes, and chronic kidney disease. He sees three specialists. Cardiology says increase his diuretics. Nephrology says hold them — his kidneys can't handle it. Endocrinology says continue Metformin. Nephrology says stop it immediately — his eGFR is below 30.
>
> Who's right? Today, his primary care doctor has to figure this out manually, with no systematic tool and no explainable reasoning. This is the reality for millions of complex chronic disease patients."

## Segment 2: The Solution (0:30–1:00)

**[Screen: Consilium architecture diagram]**

> "Consilium is a multi-agent clinical decision support system built on the Prompt Opinion platform. It deploys three specialist AI agents — cardiology, nephrology, and endocrinology — each trained on their own clinical guidelines: ACC/AHA, KDIGO, and ADA.
>
> When a patient's data enters the system, all three specialists analyze it independently, then a TOPSIS engine ranks their recommendations across four clinical dimensions: evidence level, patient match, drug interaction risk, and guideline priority."

## Segment 3: Live Demo (1:00–2:10)

**[Screen: Prompt Opinion platform, running the orchestration]**

> "Let me show you. I'll enter Mr. Chen's data into the Prompt Opinion platform..."

**[Type the patient query, wait for response]**

> "In about 30 seconds, Consilium returns a complete decision. Let's look at the results."

**[Scroll through the output, highlight key sections]**

> "Nephrology ranks number one with a TOPSIS score of 0.945. The key finding: Metformin must be stopped immediately — his eGFR of 28 is below the absolute contraindication threshold. All three specialists unanimously agree on this.
>
> The system recommends starting an SGLT2 inhibitor — the only drug class endorsed by all three specialties for its triple benefit across heart failure, kidney disease, and diabetes.
>
> Notice the unified action plan — prioritized by urgency. And the key conflicts resolved section shows exactly where specialists disagreed and how the system reconciled it."

## Segment 4: Why AI + TOPSIS (2:10–2:40)

**[Screen: comparison table or scoring breakdown]**

> "Why not just use rule-based software? Because clinical guidelines are written in natural language with conditional logic that rules engines can't parse. The LLM extracts structured signals from free-text guidelines, then TOPSIS — a deterministic algorithm — handles the multi-criteria tradeoff mathematically. AI and rules working together, not替代.
>
> Every recommendation comes with evidence citations, confidence scores, and risk flags. This isn't a black box — it's an auditable decision chain."

## Segment 5: Impact (2:40–3:00)

**[Screen: impact summary]**

> "Consilium can reduce medication conflict risk for complex chronic disease patients, save primary care physicians hours of manual reconciliation per case, and scale to any number of specialties.
>
> All recommendations are advisory — final decisions rest with the treating physician. Consilium doesn't replace clinical judgment. It makes clinical judgment faster, safer, and more transparent.
>
> Thank you."

---

## Recording Tips

- Use OBS or Loom for screen recording
- Keep the Prompt Opinion platform visible during the demo segment
- Pre-type the patient query in a text editor so you can paste it quickly
- Have the output already scrolled and ready to show (record a second take if needed)
- Total time: aim for 2:50–2:55 to leave margin
