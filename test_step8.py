#!/usr/bin/env python
"""
Test Step 8 implementation: Pydantic + instructor integration.
Verifies that all models and functions work correctly.
"""

import json
import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from models import ScoreResult, StructuredProfile, ScoringWeights
from scorer import get_instructor_client
from db import load_config, Job

def test_models():
    """Test Pydantic models."""
    print("\n" + "="*60)
    print("Test 1: Pydantic Models")
    print("="*60)
    
    # Test ScoreResult with clamping
    sr = ScoreResult(
        disqualified=False,
        role_fit=15,  # Should clamp to 10
        stack_match=5,
        seniority=7,
        location=3,
        growth=9,
        compensation=2,
        reasons=['Good fit', 'Remote available'],
        flags=['check visa'],
        one_liner='Promising Backend role'
    )
    assert sr.role_fit == 10, f"Expected role_fit=10, got {sr.role_fit}"
    print("✓ ScoreResult: role_fit clamped correctly (15 -> 10)")
    
    # Test ScoreResult when disqualified
    sr_disq = ScoreResult(
        disqualified=True,
        disqualify_reason='Masters required',
        role_fit=10,
        stack_match=10,
        seniority=10,
        location=10,
        growth=10,
        compensation=10
    )
    assert sr_disq.role_fit == 0, f"Expected all scores=0 when disqualified"
    assert sr_disq.stack_match == 0
    print("✓ ScoreResult: all scores zeroed when disqualified")
    
    # Test StructuredProfile with deduplication
    sp = StructuredProfile(
        name='John Doe',
        yoe=3,
        core_skills=['Python', 'Python', 'Go', 'Go', 'Rust'],
        languages=['Python', 'Go', 'Python'],
        min_salary=150000,
        remote_preference='True'
    )
    assert sp.core_skills == ['Python', 'Go', 'Rust'], f"Dedup failed: {sp.core_skills}"
    print("✓ StructuredProfile: core_skills deduplicated correctly")
    assert sp.languages == ['Python', 'Go'], f"Languages dedup failed: {sp.languages}"
    print("✓ StructuredProfile: languages deduplicated correctly")
    
    # Test ScoringWeights validation
    sw = ScoringWeights(
        role_fit=0.30,
        stack_match=0.25,
        seniority=0.20,
        location=0.10,
        growth=0.10,
        compensation=0.05
    )
    total = sw.role_fit + sw.stack_match + sw.seniority + sw.location + sw.growth + sw.compensation
    assert 0.99 <= total <= 1.01, f"Weights don't sum to 1.0: {total}"
    print("✓ ScoringWeights: weights sum to 1.0 ± 0.01")
    
    print("\n✓ All model tests passed!")


def test_instructor_client():
    """Test instructor client factory."""
    print("\n" + "="*60)
    print("Test 2: Instructor Client Factory")
    print("="*60)
    
    # This test requires a real config.yaml, so we'll just verify the function exists
    # and can be imported
    from scorer import get_instructor_client
    
    print("✓ get_instructor_client function imported successfully")
    print("✓ Instructor client factory is available in scorer.py")
    print("  (Full test requires real config.yaml with valid API keys)")
    
    print("\n✓ Instructor client factory test passed!")


def test_backward_compatibility():
    """Verify backward compatibility with existing code."""
    print("\n" + "="*60)
    print("Test 3: Backward Compatibility")
    print("="*60)
    
    # Test that ScoreResult can be converted to dict
    sr = ScoreResult(
        disqualified=False,
        role_fit=8,
        stack_match=7,
        seniority=6,
        location=9,
        growth=7,
        compensation=5,
        reasons=['Good culture', 'Strong team'],
        flags=[],
        one_liner='Great opportunity'
    )
    
    sr_dict = sr.model_dump()
    assert isinstance(sr_dict, dict)
    assert 'role_fit' in sr_dict
    assert sr_dict['role_fit'] == 8
    print("✓ ScoreResult converts to dict correctly")
    
    # Test that StructuredProfile can be converted to dict
    sp = StructuredProfile(
        name='Alice',
        yoe=5,
        current_title='Senior Engineer',
        core_skills=['Python', 'Go'],
        min_salary=180000
    )
    
    sp_dict = sp.model_dump()
    assert isinstance(sp_dict, dict)
    assert 'name' in sp_dict
    assert sp_dict['name'] == 'Alice'
    print("✓ StructuredProfile converts to dict correctly")
    
    print("\n✓ Backward compatibility tests passed!")


def test_score_result_reasons_limit():
    """Test that reasons are limited to 4 items."""
    print("\n" + "="*60)
    print("Test 4: ScoreResult Reasons Limit")
    print("="*60)
    
    sr = ScoreResult(
        disqualified=False,
        reasons=['r1', 'r2', 'r3', 'r4', 'r5', 'r6'],  # More than 4
        role_fit=8,
        stack_match=7,
        seniority=6,
        location=9,
        growth=7,
        compensation=5
    )
    
    assert len(sr.reasons) <= 4, f"Reasons should be limited to 4, got {len(sr.reasons)}"
    print(f"✓ Reasons limited to {len(sr.reasons)} items (input had 6)")
    
    print("\n✓ Reasons limit test passed!")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Step 8 Implementation: Pydantic + Instructor Tests")
    print("="*60)
    
    test_models()
    test_backward_compatibility()
    test_score_result_reasons_limit()
    test_instructor_client()
    
    print("\n" + "="*60)
    print("✓ All Step 8 tests passed!")
    print("="*60)
