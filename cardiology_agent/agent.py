"""cardiology_agent — Heart Failure specialist.

Zero tools. Single LLM call returns structured JSON directly.
"""
import os

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

_model_name = os.getenv("CARDIOLOGY_AGENT_MODEL", "gemini/gemini-2.5-flash")
_model = LiteLlm(model=_model_name, max_tokens=300)

root_agent = Agent(
    name="cardiology_agent",
    model=_model,
    description="Heart failure specialist.",
    instruction=(
        "You are a cardiologist. The message has a patient summary.\n"
        "Reply with ONLY this JSON (no markdown, no explanation):\n"
        '{"specialty":"cardiology","recommendation":"<2 sentences with drug names+doses>",'
        '"evidence":"<Class I/IIa/IIb>","risks":["<risk1>","<risk2>"],'
        '"citation":"ACC/AHA 2022 Sec 7.3.1"}\n'
        "Rules: HFrEF(EF<40%)→beta-blocker Class I+ACEi Class I+SGLT2i Class IIa. NSAIDs contraindicated."
    ),
)
