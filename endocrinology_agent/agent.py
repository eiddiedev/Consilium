"""endocrinology_agent — T2DM specialist.

Zero tools. Single LLM call returns structured JSON directly.
"""
import os

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

_model_name = os.getenv("ENDOCRINOLOGY_AGENT_MODEL", "gemini/gemini-2.5-flash")
_model = LiteLlm(model=_model_name)

root_agent = Agent(
    name="endocrinology_agent",
    model=_model,
    description="T2DM specialist.",
    instruction=(
        "You are an endocrinologist. The message has a patient summary.\n"
        "Reply with ONLY this JSON (no markdown, no explanation):\n"
        '{"specialty":"endocrinology","recommendation":"<2 sentences with drug names+doses>",'
        '"evidence":"<Level A/B/C>","risks":["<risk1>","<risk2>"],'
        '"citation":"ADA 2025 Sec 10"}\n'
        "Rules: Metformin contraindicated if eGFR<30. SGLT2i preferred if CKD/HF(Level A). TZDs avoid in HF."
    ),
)
