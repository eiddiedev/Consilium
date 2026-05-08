"""endocrinology_agent — T2DM specialist.

Zero tools. Single LLM call returns structured JSON directly.
Evidence level is NOT set by LLM — computed deterministically by orchestrator.
"""
import os

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

load_dotenv()

# Set ENDOCRINOLOGY_AGENT_MODEL to deepseek/deepseek-v4-pro for the higher-quality
# DeepSeek V4 option.
_model_name = os.getenv("ENDOCRINOLOGY_AGENT_MODEL", "deepseek/deepseek-v4-flash")
_model = LiteLlm(
    model=_model_name,
    max_tokens=800,
    temperature=0,
    response_format={"type": "json_object"},
)

root_agent = Agent(
    name="endocrinology_agent",
    model=_model,
    description="T2DM specialist.",
    instruction=(
        "You are an endocrinologist. The message has a patient summary.\n"
        "Reply with one valid JSON object only. Do not include markdown, prose, or reasoning.\n"
        "The response must begin with { and end with }.\n"
        "Use exactly this shape:\n"
        '{"specialty":"endocrinology","recommendation":"<2 sentences with drug names+doses>",'
        '"risks":["<risk1>","<risk2>"],'
        '"citation":"ADA 2025 Sec 10"}\n'
        "Rules: Metformin contraindicated if eGFR<30. SGLT2i preferred if CKD/HF. TZDs avoid in HF.\n"
        "Do NOT include an 'evidence' field — it is computed automatically."
    ),
)
