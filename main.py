# main.py
"""
Three exercises using OpenAI Agents SDK guardrails (input guardrails).
All Runner.run calls are expected to hit the guardrails (no model call needed).
"""

import asyncio
import re
from typing import Any

# OpenAI Agents SDK imports
from agents import (
    Agent,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    RunContextWrapper,
    Runner,
    input_guardrail,
)

# -----------------------------
# Exercise 1: Input Guardrail
# -----------------------------
@input_guardrail
async def class_timing_guardrail(
    ctx: RunContextWrapper[None], agent: Agent, input: str | list
) -> GuardrailFunctionOutput:
    # normalize input -> string
    if isinstance(input, list):
        unified = " ".join(str(getattr(i, "content", i)) for i in input)
    else:
        unified = str(input)

    text = unified.lower()
    # check exact phrase or similar (contains)
    if "change my class timing" in text or "change my class timings" in text:
        return GuardrailFunctionOutput(
            output_info={"reason": "user asked to change class timings", "text": unified},
            tripwire_triggered=True,
        )

    return GuardrailFunctionOutput(output_info={"ok": True, "text": unified}, tripwire_triggered=False)


class_timing_agent = Agent(
    name="ClassTimingAgent",
    instructions="Help students with schedule questions.",
    input_guardrails=[class_timing_guardrail],
)


# -----------------------------
# Exercise 2: Father Agent Guardrail
# Father stops child from running below 26°C
# -----------------------------
@input_guardrail
async def father_temp_guardrail(
    ctx: RunContextWrapper[None], agent: Agent, input: str | list
) -> GuardrailFunctionOutput:
    if isinstance(input, list):
        unified = " ".join(str(getattr(i, "content", i)) for i in input)
    else:
        unified = str(input)

    text = unified.lower()

    # try to extract temperature (like "24", "24°C", "24 C", "24 degrees")
    m = re.search(r"(\d{1,2})(?:\s*°\s*c|\s*°c|\s*c\b|°| degrees)?", text)
    if m:
        temp = int(m.group(1))
        info = {"detected_temp": temp, "text": unified}
        if temp < 26:
            return GuardrailFunctionOutput(output_info=info, tripwire_triggered=True)
        else:
            return GuardrailFunctionOutput(output_info=info, tripwire_triggered=False)

    # if no temperature found, do not trip (or you could choose to trip)
    return GuardrailFunctionOutput(output_info={"detected_temp": None, "text": unified}, tripwire_triggered=False)


father_agent = Agent(
    name="FatherAgent",
    instructions="Act like a caring father who decides if child may run based on temperature.",
    input_guardrails=[father_temp_guardrail],
)


# -----------------------------
# Exercise 3: Gatekeeper Agent Guardrail
# Stop students from other schools
# -----------------------------
@input_guardrail
async def gatekeeper_guardrail(
    ctx: RunContextWrapper[None], agent: Agent, input: str | list
) -> GuardrailFunctionOutput:
    if isinstance(input, list):
        unified = " ".join(str(getattr(i, "content", i)) for i in input)
    else:
        unified = str(input)

    text = unified.lower()

    # naive checks for "other school" phrases, "not my school", or explicit school name mismatch
    triggers = [
        "other school",
        "different school",
        "not from my school",
        "from other school",
        "student from ",
    ]

    # If the text explicitly says "student from My School" — allow.
    if "from my school" in text or "my school student" in text:
        return GuardrailFunctionOutput(output_info={"allowed": True, "text": unified}, tripwire_triggered=False)

    for t in triggers:
        if t in text:
            return GuardrailFunctionOutput(output_info={"reason": "other school detected", "text": unified}, tripwire_triggered=True)

    # also catch "school: <NAME>" where NAME != "My School" (optional parse)
    m = re.search(r"school[:\s-]*([a-z0-9 ]+)", text)
    if m:
        school_name = m.group(1).strip()
        if "my school" not in school_name:  # treat as other school
            return GuardrailFunctionOutput(output_info={"school_name": school_name, "text": unified}, tripwire_triggered=True)

    return GuardrailFunctionOutput(output_info={"allowed": True, "text": unified}, tripwire_triggered=False)


gatekeeper_agent = Agent(
    name="GateKeeperAgent",
    instructions="Decide whether a student can enter. Only allow 'My School' students.",
    input_guardrails=[gatekeeper_guardrail],
)


# -----------------------------
# Runner / main
# -----------------------------
async def main():
    print("\n--- Exercise 1: Input GuardRail (class timings) ---")
    try:
        # This SHOULD trip
        await Runner.run(class_timing_agent, "I want to change my class timings 😭😭")
        print("Unexpected: agent ran without tripwire.")
    except InputGuardrailTripwireTriggered:
        print("[LOG] Exercise 1: InputGuardRailTripwireTriggered (class timings)")

    print("\n--- Exercise 2: Father guardrail (temperature) ---")
    try:
        # This SHOULD trip (24 < 26)
        await Runner.run(father_agent, "Child: I want to go for a run at 24°C")
        print("Unexpected: father-agent allowed running.")
    except InputGuardrailTripwireTriggered:
        print("[LOG] Exercise 2: FatherGuardrailTripwireTriggered — child blocked (temp < 26°C)")

    print("\n--- Exercise 3: Gatekeeper guardrail (other school) ---")
    try:
        # This SHOULD trip
        await Runner.run(gatekeeper_agent, "Student from Other School wants to enter the premises")
        print("Unexpected: gatekeeper allowed entry.")
    except InputGuardrailTripwireTriggered:
        print("[LOG] Exercise 3: GateKeeperGuardrailTripwireTriggered — student blocked (other school)")


if __name__ == "__main__":
    asyncio.run(main())
