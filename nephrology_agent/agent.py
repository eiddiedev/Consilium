"""nephrology_agent — CKD specialist.

Zero tools. Single LLM call returns structured JSON directly.
Evidence level is NOT set by LLM — computed deterministically by orchestrator.
"""
import os

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

load_dotenv()

# Set NEPHROLOGY_AGENT_MODEL to deepseek/deepseek-v4-pro for the higher-quality
# DeepSeek V4 option.
_model_name = os.getenv("NEPHROLOGY_AGENT_MODEL", "deepseek/deepseek-v4-flash")
_model = LiteLlm(
    model=_model_name,
    max_tokens=800,
    temperature=0,
    response_format={"type": "json_object"},
)

root_agent = Agent(
    name="nephrology_agent",
    model=_model,
    description="CKD specialist.",
    instruction=(
        "You are a nephrologist. The message has a patient summary.\n"
        "Reply with one valid JSON object only. Do not include markdown, prose, or reasoning.\n"
        "The response must begin with { and end with }.\n"
        "Use exactly this shape:\n"
        '{"specialty":"nephrology","recommendation":"<2 sentences with drug names+doses>",'
        '"risks":["<risk1>","<risk2>"],'
        '"citation":"KDIGO 2024 Ch 3"}\n'
        "Rules: Metformin CONTRAINDICATED if eGFR<30. SGLT2i if eGFR>=20. NSAIDs avoid.\n"
        "Do NOT include an 'evidence' field — it is computed automatically."
    ),
)
