"""
llm_utils.py — Shared LLM resilience utilities for structured output calls.

CONTRACT: Any code that makes structured-output LLM calls (instructor / tool use)
MUST go through safe_structured_call() from this module. Calling instructor
directly bypasses the recovery layer and will silently waste retries on schema
errors that can never self-heal.

Background:
  Some models (e.g. llama-4-scout-17b via Groq) wrap every tool-call parameter in a
  {"value": x} object instead of returning the raw scalar. Groq's API validates the
  tool schema before instructor even sees the response and rejects it with a 400
  "tool_use_failed". Without this module:
    - instructor's internal tenacity loop retries the bad call 2-3 times, sleeping
      ~10 s between attempts → 30 s of dead time per LLM call
    - the failed_generation payload (which contains all the correct data, just
      wrapped) is silently discarded, forcing an extra raw-LLM API call

  This module fixes both problems at every call site.
"""

import ast as _ast
import json
import re
import time
from typing import Any, Optional, Type

from loguru import logger
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def unwrap_value_objects(data: Any) -> Any:
    """
    Recursively unwrap {"value": x} single-key dicts produced by some models.
    e.g. llama-4-scout wraps every tool-call field in {"value": ...}.
    Leaves all other structures untouched.
    """
    if isinstance(data, dict):
        if list(data.keys()) == ["value"]:
            return unwrap_value_objects(data["value"])
        return {k: unwrap_value_objects(v) for k, v in data.items()}
    if isinstance(data, list):
        return [unwrap_value_objects(item) for item in data]
    return data


def parse_llm_response(raw: str) -> dict:
    """
    Parse a raw LLM text response as JSON.
    Strips markdown code fences (```json ... ```) and applies unwrap_value_objects
    so any {"value": x} wrapping is removed regardless of whether the model was
    called via tool-use or plain chat completion.

    Raises json.JSONDecodeError on invalid JSON (caller handles).
    """
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE)
    parsed = json.loads(raw)
    return unwrap_value_objects(parsed)


def is_schema_error(exc: Exception) -> bool:
    """
    Return True if the exception is a provider-side tool-call schema validation
    failure (HTTP 400 / tool_use_failed).

    These errors are deterministic: the same prompt sent to the same model will
    always produce the same malformed output. Retrying wastes time and quota.
    """
    msg = str(exc)
    return (
        "400" in msg
        or "tool_use_failed" in msg
        or "tool call validation" in msg.lower()
        or "parameters for tool" in msg.lower()
    )


def is_rate_limit_error(exc: Exception) -> bool:
    """Return True if the exception looks like a provider rate-limit (429 / RESOURCE_EXHAUSTED)."""
    msg = str(exc)
    return "429" in msg or "RESOURCE_EXHAUSTED" in msg or "rate_limit" in msg.lower()


# ---------------------------------------------------------------------------
# failed_generation recovery
# ---------------------------------------------------------------------------


def extract_from_failed_generation(exc: Exception, label: str = "recovery") -> Optional[dict]:
    """
    When a provider returns a 400 tool_use_failed, it often embeds the model's
    actual output in a 'failed_generation' field. The data is correct — it is
    just wrapped in {"value": x} objects that fail schema validation.

    This function extracts that payload without making an extra API call.

    Three strategies are tried across the full exception chain:
      1. exc.body dict — present on raw SDK errors (common path when max_retries=0)
      2. Walking __cause__ / __context__ / last_attempt for a nested exc with .body
      3. ast.literal_eval on "Error code: 400 - {..." in str(exc) as string fallback

    Returns the unwrapped parameter dict on success, None on any failure.
    All outcomes are logged at INFO so they appear in journalctl without --debug.
    """
    logger.info(f"{label} | attempting failed_generation extraction from exception")

    fg_str: Optional[str] = None
    seen_ids: set = set()
    queue: list = [exc]

    while queue and fg_str is None:
        current = queue.pop(0)
        if id(current) in seen_ids:
            continue
        seen_ids.add(id(current))

        # Strategy 1: .body dict (raw SDK error, e.g. groq.BadRequestError)
        body = getattr(current, "body", None)
        if isinstance(body, dict):
            candidate = body.get("error", {}).get("failed_generation")
            if candidate:
                fg_str = candidate
                logger.info(
                    f"{label} | found failed_generation via .body "
                    f"on {type(current).__name__}"
                )

        # Strategy 2: ast.literal_eval on the Python dict repr in str(exc).
        # Groq formats its error as: "Error code: 400 - {'error': {...}}"
        # ast.literal_eval handles nested dicts and escapes correctly;
        # unlike regex it won't break on single quotes inside string values.
        if not fg_str:
            msg = str(current)
            for marker in ("- {'error'", '- {"error"'):
                idx = msg.find(marker)
                if idx != -1:
                    try:
                        parsed = _ast.literal_eval(msg[idx + 2:].strip())
                        if isinstance(parsed, dict):
                            candidate = parsed.get("error", {}).get("failed_generation")
                            if candidate:
                                fg_str = candidate
                                logger.info(
                                    f"{label} | found failed_generation via ast.literal_eval "
                                    f"on {type(current).__name__}"
                                )
                    except Exception as parse_err:
                        logger.debug(
                            f"{label} | ast.literal_eval failed "
                            f"on {type(current).__name__}: {parse_err}"
                        )
                    break

        if fg_str:
            break

        # Enqueue chained exceptions for the next iteration.
        # InstructorRetryException (tenacity) stores the last attempt as .last_attempt
        last_attempt = getattr(current, "last_attempt", None)
        if last_attempt is not None:
            try:
                last_attempt.result()
            except Exception as inner_exc:
                queue.append(inner_exc)

        if current.__cause__ is not None:
            queue.append(current.__cause__)
        elif current.__context__ is not None:
            queue.append(current.__context__)

    if not fg_str:
        logger.info(
            f"{label} | failed_generation not found in exception chain "
            "— will use raw LLM fallback"
        )
        return None

    # Bug 2: Groq sometimes returns a Python dict repr (single-quoted) instead of
    # valid JSON when the generation was truncated or malformed. Try ast.literal_eval
    # as a fallback before giving up.
    try:
        failed_gen = json.loads(fg_str)
    except Exception as json_err:
        try:
            failed_gen = _ast.literal_eval(fg_str)
            logger.info(f"{label} | JSON parse failed; ast.literal_eval recovered the payload")
        except Exception:
            logger.info(
                f"{label} | JSON parse of failed_generation failed: {json_err} "
                "— will use raw LLM fallback"
            )
            return None

    # Format from Groq: [{"name": "ModelName", "parameters": {...}}]
    # Bug 1: StructuredProfile tool calls may not wrap fields under "parameters" —
    # the key may be absent entirely. If params is empty, fall back to treating
    # the parsed item itself as the param dict so the fields are not silently dropped.
    if isinstance(failed_gen, list) and failed_gen:
        params = failed_gen[0].get("parameters", {})
        if not params:
            logger.info(
                f"{label} | list[0]['parameters'] empty "
                f"(shape keys: {list(failed_gen[0].keys())}) "
                "— trying list[0] as param dict"
            )
            params = {k: v for k, v in failed_gen[0].items() if k != "name"}
    elif isinstance(failed_gen, dict):
        params = failed_gen.get("parameters", failed_gen)
    else:
        logger.info(
            f"{label} | failed_generation parsed but structure unexpected "
            "— will use raw LLM fallback"
        )
        return None

    result = unwrap_value_objects(params)
    logger.info(
        f"{label} | SUCCESS — recovered {len(result)} fields from failed_generation "
        f"(shape: {type(failed_gen).__name__})"
    )
    if not result:
        logger.info(f"{label} | recovered 0 fields — treating as failed recovery")
        return None
    return result


# ---------------------------------------------------------------------------
# Unified structured-output call
# ---------------------------------------------------------------------------


def safe_structured_call(
    client: Any,
    model: str,
    prompt: str,
    response_model: Type[BaseModel],
    *,
    max_tokens: int = 700,
    temperature: float = 0,
    label: str = "llm",
    max_retries: int = 3,
) -> Optional[dict]:
    """
    Call an instructor client for structured output with full resilience.

    CONTRACT: Any code that makes structured-output LLM calls MUST use this
    function. Do not call instructor directly — that bypasses the recovery layer.

    Behavior
    --------
    - Sets max_retries=0 on each attempt to prevent instructor's internal tenacity
      delays. Without this, a 400 schema error causes instructor to retry internally
      (sleeping ~10 s between each) before raising, adding 30 s+ of dead time per call.
    - Schema errors (400 / tool_use_failed) are deterministic: same prompt + same
      model = same broken output. The retry loop is short-circuited immediately.
      The failed_generation payload is inspected first — if it contains the model's
      actual output, the data is recovered and validated through pydantic without
      an extra API call.
    - Rate limits get exponential backoff (10 s, 20 s, 40 s).
    - Other transient errors get exponential backoff and are retried.

    Parameters
    ----------
    client        : Instructor-wrapped LLM client
    model         : Model name string (provider-specific)
    prompt        : User prompt string
    response_model: Pydantic model class for structured output
    max_tokens    : Maximum tokens in the response (default 700)
    temperature   : Sampling temperature (default 0)
    label         : Log prefix, e.g. "scorer" or "profile" (default "llm")
    max_retries   : Maximum number of OUR retry attempts (default 3)

    Returns
    -------
    model_dump() dict on success (pydantic-validated), or None if all attempts
    fail (caller should fall back to a raw LLM call with parse_llm_response).
    """
    logger.info(f"{label} | safe_structured_call starting ({model}, max_retries={max_retries})")

    for attempt in range(max_retries):
        logger.info(f"{label} | attempt {attempt + 1}/{max_retries} — calling instructor tool API")
        try:
            result = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
                response_model=response_model,
                # Disable instructor's internal tenacity loop. Without max_retries=0,
                # a schema-mismatch 400 causes instructor to retry 2-3 times with its
                # own sleeps before raising — burning 30 s+ per call. We own retries.
                max_retries=0,
            )
            dims = result.model_dump()
            logger.info(f"{label} | attempt {attempt + 1} succeeded")
            return dims

        except Exception as exc:
            if is_schema_error(exc):
                # Schema mismatch is deterministic: same prompt → same model → same
                # broken output. Short-circuit the retry loop immediately and attempt
                # zero-cost recovery from the failed_generation payload.
                logger.info(
                    f"{label} | attempt {attempt + 1} — schema error (tool_use_failed). "
                    "Short-circuiting retry loop; schema mismatch won't self-heal."
                )
                recovered = extract_from_failed_generation(exc, label=label)
                if recovered is not None:
                    # Run through pydantic to apply field validators
                    # (score clamping, list deduplication, disqualified zeroing, etc.)
                    try:
                        validated = response_model(**recovered)
                        dims = validated.model_dump()
                        logger.info(
                            f"{label} | recovery + pydantic validation succeeded "
                            "— no extra API call needed"
                        )
                        return dims
                    except Exception as val_err:
                        # Pydantic rejected the recovered dict (unexpected field names,
                        # etc.). Return the raw unwrapped dict — it's still useful.
                        logger.info(
                            f"{label} | pydantic validation of recovered dims failed "
                            f"({val_err}) — returning raw recovered dict"
                        )
                        return recovered
                # Recovery found nothing useful; signal caller to use raw LLM.
                logger.info(f"{label} | recovery failed — caller should use raw LLM fallback")
                return None

            if is_rate_limit_error(exc):
                if attempt < max_retries - 1:
                    wait = 2 ** attempt * 10
                    logger.warning(
                        f"{label} | attempt {attempt + 1} rate-limited — retrying in {wait}s"
                    )
                    time.sleep(wait)
                else:
                    logger.warning(f"{label} | rate limit on final attempt — giving up")
                    return None
            else:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt * 10
                    logger.warning(
                        f"{label} | attempt {attempt + 1} failed — retrying in {wait}s: {exc}"
                    )
                    time.sleep(wait)
                else:
                    logger.warning(
                        f"{label} | attempt {attempt + 1} failed (final) — giving up: {exc}"
                    )
                    return None

    return None
