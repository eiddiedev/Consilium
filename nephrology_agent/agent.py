"""nephrology_agent — CKD specialist.

Zero tools. Single LLM call returns structured JSON directly.
"""
import os

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

_model_name = os.getenv("NEPHROLOGY_AGENT_MODEL", "gemini/gemini-2.5-flash")
_model = LiteLlm(model=_model_name)

root_agent = Agent(
    name="nephrology_agent",
    model=_model,
    description="CKD specialist.",
    instruction=(
        "You are a nephrologist. The message has a patient summary.\n"
        "Reply with ONLY this JSON (no markdown, no explanation):\n"
        '{"specialty":"nephrology","recommendation":"<2 sentences with drug names+doses>",'
        '"evidence":"<1A/1B/2A>","risks":["<risk1>","<risk2>"],'
        '"citation":"KDIGO 2024 Ch 3"}\n'
        "Rules: Metformin CONTRAINDICATED if eGFR<30(1A). SGLT2i if eGFR>=20(1A). NSAIDs avoid(2A)."
    ),
)
