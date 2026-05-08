"""cardiology_agent — Heart Failure specialist.

Zero tools. Single LLM call returns structured JSON directly.
Evidence level is NOT set by LLM — computed deterministically by orchestrator.
"""
import os

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

load_dotenv()

# Set CARDIOLOGY_AGENT_MODEL to deepseek/deepseek-v4-pro for the higher-quality
# DeepSeek V4 option.
_model_name = os.getenv("CARDIOLOGY_AGENT_MODEL", "deepseek/deepseek-v4-flash")
_model = LiteLlm(
    model=_model_name,
    max_tokens=800,
    temperature=0,
    response_format={"type": "json_object"},
)

root_agent = Agent(
    name="cardiology_agent",
    model=_model,
    description="Heart failure specialist.",
    instruction=(
        "You are a cardiologist. The message has a patient summary.\n"
        "Reply with one valid JSON object only. Do not include markdown, prose, or reasoning.\n"
        "The response must begin with { and end with }.\n"
        "Use exactly this shape:\n"
        '{"specialty":"cardiology","recommendation":"<2 sentences with drug names+doses>",'
        '"risks":["<risk1>","<risk2>"],'
        '"citation":"ACC/AHA 2022 Sec 7.3.1"}\n'
        "Rules: HFrEF(EF<40%)→beta-blocker+ACEi+SGLT2i. NSAIDs contraindicated.\n"
        "Do NOT include an 'evidence' field — it is computed automatically."
    ),
)
