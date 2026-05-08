# Consilium — 3-Minute Demo Script

## Segment 1: Problem (0:00-0:25)

**Screen:** Opening image or simple slide: one complex chronic disease patient receives conflicting medication recommendations from two specialists.

> Complex chronic disease care often breaks down at the handoff between specialists. A heart failure doctor may optimize cardiac therapy. A nephrologist may worry about kidney safety. An endocrinologist may focus on glucose control. For a patient with heart failure, CKD, and diabetes, those recommendations can directly conflict. Today, the primary care clinician has to reconcile that manually.

**Opening image prompt:**

> A clean medical technology illustration, one older patient in the center, two physicians on opposite sides handing conflicting prescriptions, one prescription says "Continue Metformin", the other says "Stop Metformin", subtle warning lines between them, white and teal clinical style, premium healthcare UI aesthetic, no logos, no real hospital branding.

## Segment 2: Solution (0:25-0:50)

**Screen:** Consilium frontend home screen or architecture diagram.

> Consilium is a multi-specialty clinical decision system. It uses three specialist AI agents: cardiology, nephrology, and endocrinology. Each agent generates a structured recommendation. Then Consilium uses deterministic TOPSIS scoring to rank those recommendations by evidence, patient match, medication safety, and guideline priority.

## Segment 3: Frontend Demo (0:50-1:35)

**Screen:** React demo. Select Patient A and click Run Orchestration.

> Here is the clinical workspace. This synthetic patient is a 68-year-old male with HFrEF, CKD stage 4, type 2 diabetes, and Metformin use. When I run orchestration, the pipeline shows FHIR retrieval, three specialist agents, TOPSIS scoring, and final formatting. The final decision ranks the safety-critical recommendation first: stop Metformin because eGFR is below 30. It also identifies an SGLT2 inhibitor as a cross-specialty aligned action.

## Segment 4: Prompt Opinion Integration (1:35-2:20)

**Screen:** Prompt Opinion A2A connection. Show the FHIR extension toggle, then run a request.

> Consilium is not just a standalone demo. It is deployed as an A2A agent on Google Cloud Run and connected inside Prompt Opinion. The agent card advertises the official Prompt Opinion FHIR context extension, so the platform can pass FHIR URL, patient ID, and authorization context through the A2A request. When FHIR context is present, Consilium builds a patient summary from real structured resources. When context is missing or insufficient, it refuses to hallucinate and asks for more patient information.

## Segment 5: Why It Matters (2:20-2:50)

**Screen:** Returned result. Highlight ranked recommendations, Key Conflicts Resolved, and citations.

> The important part is the safety boundary. The LLM agents do not freely assign clinical scores. They generate specialist recommendations under a JSON contract. The ranking, safety overrides, and guideline priority are computed in code. That makes the system more auditable and more feasible for real clinical decision support.

## Segment 6: Closing (2:50-3:00)

**Screen:** Consilium title and one-line value statement.

> Consilium helps clinicians reconcile complex guideline conflicts faster, safer, and more transparently. It is advisory clinical decision support, not autonomous prescribing.

## Recording Checklist

- Keep the final video under 3 minutes.
- Show Prompt Opinion integration, not only the local React demo.
- Use only synthetic or de-identified patient data.
- Show the FHIR context extension toggle if possible.
- Have the patient query ready to paste before recording.
- Keep the Cloud Run service warm with one smoke request before recording.
