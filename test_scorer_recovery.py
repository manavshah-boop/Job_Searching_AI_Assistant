#!/usr/bin/env python
"""
test_scorer_recovery.py — Unit tests for the malformed-structured-output recovery layer.

Tests _unwrap_value_objects and extract_from_failed_generation using the exact
error payload captured from a live Groq / llama-4-scout-17b production failure.

Run with:
    python test_scorer_recovery.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from llm_utils import unwrap_value_objects, extract_from_failed_generation

# ---------------------------------------------------------------------------
# Exact payload captured from production logs (Apr 16 05:50:08)
# ---------------------------------------------------------------------------

_FAILED_GENERATION = (
    '[\n  {\n    "name": "ScoreResult",\n    "parameters": {\n'
    '      "compensation": {\n        "value": 8\n      },\n'
    '      "disqualified": {\n        "value": false\n      },\n'
    '      "disqualify_reason": {\n        "value": ""\n      },\n'
    '      "flags": {\n        "value": ["compensation below minimum"]\n      },\n'
    '      "growth": {\n        "value": 9\n      },\n'
    '      "location": {\n        "value": 8\n      },\n'
    '      "one_liner": {\n        "value": "Strong fit for AI engineer role with growth opportunities"\n      },\n'
    '      "reasons": {\n        "value": ["relevant experience", "matching tech stack", "AI-native company"]\n      },\n'
    '      "role_fit": {\n        "value": 9\n      },\n'
    '      "seniority": {\n        "value": 8\n      },\n'
    '      "stack_match": {\n        "value": 9\n      }\n'
    '    }\n  }\n]'
)

_GROQ_400_BODY = {
    "error": {
        "message": (
            "tool call validation failed: parameters for tool ScoreResult did not match schema: "
            "errors: [`/compensation`: expected integer, but got object, "
            "`/disqualify_reason`: expected string, but got object, "
            "`/flags`: expected array, but got object, "
            "`/disqualified`: expected boolean, but got object, "
            "`/stack_match`: expected integer, but got object, "
            "`/growth`: expected integer, but got object, "
            "`/location`: expected integer, but got object, "
            "`/one_liner`: expected string, but got object, "
            "`/role_fit`: expected integer, but got object, "
            "`/seniority`: expected integer, but got object, "
            "`/reasons`: expected array, but got object]"
        ),
        "type": "invalid_request_error",
        "code": "tool_use_failed",
        "failed_generation": _FAILED_GENERATION,
    }
}

# ---------------------------------------------------------------------------
# Fake exception classes that mirror the real SDK shapes
# ---------------------------------------------------------------------------


class FakeGroqBadRequestError(Exception):
    """
    Mirrors groq.BadRequestError: has a .body dict and a string repr of
    "Error code: 400 - {<body dict>}".
    """
    def __init__(self, body: dict) -> None:
        self.body = body
        self.status_code = 400
        super().__init__(f"Error code: 400 - {body!r}")


class FakeInstructorRetryException(Exception):
    """
    Mirrors instructor.exceptions.InstructorRetryException wrapping tenacity.
    Does NOT have .body. str() returns the multi-attempt XML that instructor
    formats. The original exception is accessible via last_attempt.result().
    """
    def __init__(self, last_exc: Exception) -> None:
        self._last_exc = last_exc
        xml = (
            "<failed_attempts>\n"
            f"<exception>\n    {last_exc}\n</exception>\n"
            f"<exception>\n    {last_exc}\n</exception>\n"
            "</failed_attempts>\n"
            f"<last_exception>\n    {last_exc}\n</last_exception>"
        )
        super().__init__(xml)

    @property
    def last_attempt(self):
        exc = self._last_exc

        class _FakeAttempt:
            def result(self_inner):
                raise exc

        return _FakeAttempt()


class FakeChainedError(Exception):
    """Error whose __cause__ is a FakeGroqBadRequestError."""
    def __init__(self, cause: Exception) -> None:
        super().__init__("outer wrapper")
        self.__cause__ = cause


# ---------------------------------------------------------------------------
# Tests: _unwrap_value_objects
# ---------------------------------------------------------------------------


def test_unwrap_flat_scalar_fields():
    """Every field wrapped in {"value": x} should be unwrapped to x."""
    wrapped = {
        "role_fit": {"value": 9},
        "stack_match": {"value": 8},
        "disqualified": {"value": False},
        "one_liner": {"value": "Great fit"},
    }
    result = unwrap_value_objects(wrapped)
    assert result == {
        "role_fit": 9,
        "stack_match": 8,
        "disqualified": False,
        "one_liner": "Great fit",
    }, f"Flat unwrap failed: {result}"
    print("PASS  _unwrap_value_objects: flat scalar fields")


def test_unwrap_list_fields():
    """List fields wrapped in {"value": [...]} should unwrap to plain lists."""
    wrapped = {
        "reasons": {"value": ["relevant experience", "matching tech stack"]},
        "flags": {"value": ["compensation below minimum"]},
    }
    result = unwrap_value_objects(wrapped)
    assert result["reasons"] == ["relevant experience", "matching tech stack"]
    assert result["flags"] == ["compensation below minimum"]
    print("PASS  _unwrap_value_objects: list fields")


def test_unwrap_already_flat_passthrough():
    """Values that are already flat should pass through unchanged."""
    flat = {"role_fit": 9, "disqualified": False, "reasons": ["a", "b"]}
    result = unwrap_value_objects(flat)
    assert result == flat
    print("PASS  _unwrap_value_objects: already-flat passthrough")


def test_unwrap_full_production_params():
    """Unwrap the exact parameters object from the production failed_generation."""
    raw_params = json.loads(_FAILED_GENERATION)[0]["parameters"]
    result = unwrap_value_objects(raw_params)
    assert result["compensation"] == 8,        f"compensation: {result['compensation']}"
    assert result["disqualified"] is False,    f"disqualified: {result['disqualified']}"
    assert result["disqualify_reason"] == "",  f"disqualify_reason: {result['disqualify_reason']}"
    assert result["growth"] == 9,              f"growth: {result['growth']}"
    assert result["location"] == 8,            f"location: {result['location']}"
    assert result["role_fit"] == 9,            f"role_fit: {result['role_fit']}"
    assert result["seniority"] == 8,           f"seniority: {result['seniority']}"
    assert result["stack_match"] == 9,         f"stack_match: {result['stack_match']}"
    assert result["flags"] == ["compensation below minimum"]
    assert result["reasons"] == ["relevant experience", "matching tech stack", "AI-native company"]
    assert "growth opportunities" in result["one_liner"]
    print("PASS  _unwrap_value_objects: full production params")


# ---------------------------------------------------------------------------
# Tests: extract_from_failed_generation — strategy 1 (.body)
# ---------------------------------------------------------------------------


def test_extract_from_raw_groq_error_via_body():
    """Raw groq.BadRequestError with .body — the common path when max_retries=0."""
    exc = FakeGroqBadRequestError(_GROQ_400_BODY)
    dims = extract_from_failed_generation(exc)

    assert dims is not None, "Expected dims, got None"
    assert dims["role_fit"] == 9,         f"role_fit={dims['role_fit']}"
    assert dims["stack_match"] == 9,      f"stack_match={dims['stack_match']}"
    assert dims["seniority"] == 8,        f"seniority={dims['seniority']}"
    assert dims["compensation"] == 8,     f"compensation={dims['compensation']}"
    assert dims["disqualified"] is False, f"disqualified={dims['disqualified']}"
    assert dims["reasons"] == ["relevant experience", "matching tech stack", "AI-native company"]
    assert dims["flags"] == ["compensation below minimum"]
    print("PASS  extract_from_failed_generation: raw .body strategy")


# ---------------------------------------------------------------------------
# Tests: extract_from_failed_generation — strategy 2 (ast.literal_eval)
# ---------------------------------------------------------------------------


def test_extract_from_string_repr_via_ast():
    """
    Exception with no .body — recovery falls back to ast.literal_eval on str(exc).
    Mirrors what happens when instructor wraps the error without exposing .body.
    """
    class BodylessError(Exception):
        """No .body attribute; str() matches Groq's error format."""
        def __init__(self, body: dict) -> None:
            super().__init__(f"Error code: 400 - {body!r}")

    exc = BodylessError(_GROQ_400_BODY)
    dims = extract_from_failed_generation(exc)

    assert dims is not None, "Expected dims from string repr fallback, got None"
    assert dims["role_fit"] == 9
    assert dims["stack_match"] == 9
    assert dims["disqualified"] is False
    print("PASS  extract_from_failed_generation: ast.literal_eval string fallback")


# ---------------------------------------------------------------------------
# Tests: extract_from_failed_generation — strategy 3 (exception chain)
# ---------------------------------------------------------------------------


def test_extract_from_instructor_retry_exception_via_chain():
    """
    InstructorRetryException wraps the raw Groq error via last_attempt.
    Recovery should walk the chain and find the nested .body.
    """
    inner = FakeGroqBadRequestError(_GROQ_400_BODY)
    exc = FakeInstructorRetryException(inner)

    # Confirm no .body on the outer exception
    assert not hasattr(exc, "body") or getattr(exc, "body", None) is None

    dims = extract_from_failed_generation(exc)

    assert dims is not None, "Expected dims from chained exception, got None"
    assert dims["role_fit"] == 9
    assert dims["disqualified"] is False
    print("PASS  extract_from_failed_generation: InstructorRetryException chain walk")


def test_extract_from_chained_cause():
    """Recovery walks __cause__ to find the original Groq error."""
    inner = FakeGroqBadRequestError(_GROQ_400_BODY)
    exc = FakeChainedError(inner)

    dims = extract_from_failed_generation(exc)

    assert dims is not None, "Expected dims from __cause__ chain, got None"
    assert dims["role_fit"] == 9
    print("PASS  extract_from_failed_generation: __cause__ chain walk")


# ---------------------------------------------------------------------------
# Tests: graceful None return on unrecoverable input
# ---------------------------------------------------------------------------


def test_extract_returns_none_on_unrelated_error():
    """A plain ValueError with no error body should return None cleanly."""
    exc = ValueError("some unrelated error")
    dims = extract_from_failed_generation(exc)
    assert dims is None, f"Expected None for unrelated error, got {dims}"
    print("PASS  extract_from_failed_generation: unrelated error -> None")


def test_extract_returns_none_on_missing_failed_generation():
    """A 400 body without failed_generation should return None cleanly."""

    class NoFGError(Exception):
        def __init__(self) -> None:
            self.body = {"error": {"message": "some other 400", "code": "bad_request"}}
            super().__init__(f"Error code: 400 - {self.body!r}")

    dims = extract_from_failed_generation(NoFGError())
    assert dims is None, f"Expected None when failed_generation absent, got {dims}"
    print("PASS  extract_from_failed_generation: missing failed_generation -> None")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def main():
    print("\n" + "=" * 65)
    print("test_scorer_recovery.py — malformed structured output recovery")
    print("=" * 65 + "\n")

    tests = [
        # _unwrap_value_objects
        test_unwrap_flat_scalar_fields,
        test_unwrap_list_fields,
        test_unwrap_already_flat_passthrough,
        test_unwrap_full_production_params,
        # extract_from_failed_generation
        test_extract_from_raw_groq_error_via_body,
        test_extract_from_string_repr_via_ast,
        test_extract_from_instructor_retry_exception_via_chain,
        test_extract_from_chained_cause,
        test_extract_returns_none_on_unrelated_error,
        test_extract_returns_none_on_missing_failed_generation,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAIL  {test.__name__}: {e}")
            failed += 1

    print(f"\n{'=' * 65}")
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 65)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
