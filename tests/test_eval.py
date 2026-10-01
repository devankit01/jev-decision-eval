"""Unit tests for eval.py — no network calls required."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval import (
    score_to_band,
    evaluate_answers,
    PROBLEMS,
    DATASET,
    ALL_MODELS,
    OLLAMA_MODELS,
    JEV_MODEL,
)


# --- score_to_band ---

def test_score_to_band_low():
    assert score_to_band(0.0) == "low"
    assert score_to_band(0.32) == "low"

def test_score_to_band_mid():
    assert score_to_band(0.33) == "mid"
    assert score_to_band(0.66) == "mid"

def test_score_to_band_high():
    assert score_to_band(0.67) == "high"
    assert score_to_band(1.0) == "high"


# --- evaluate_answers ---

def test_evaluate_choice_correct():
    questions = {
        "team": {"type": "choice", "instructions": "...", "criteria": {"billing": "", "bug": ""}}
    }
    answers = {"team": {"type": "choice", "choice": "billing", "confidence": 0.9}}
    ground_truth = {"team": "billing"}
    correct, total, avg_conf = evaluate_answers(answers, ground_truth, questions)
    assert correct == 1
    assert total == 1
    assert avg_conf == pytest.approx(0.9)

def test_evaluate_choice_wrong():
    questions = {
        "team": {"type": "choice", "instructions": "...", "criteria": {"billing": "", "bug": ""}}
    }
    answers = {"team": {"type": "choice", "choice": "bug", "confidence": 0.7}}
    ground_truth = {"team": "billing"}
    correct, total, _ = evaluate_answers(answers, ground_truth, questions)
    assert correct == 0
    assert total == 1

def test_evaluate_noul_true_correct():
    questions = {"refund": {"type": "noul", "instructions": "..."}}
    answers = {"refund": {"type": "noul", "noul": 0.95}}
    ground_truth = {"refund": True}
    correct, total, _ = evaluate_answers(answers, ground_truth, questions)
    assert correct == 1
    assert total == 1

def test_evaluate_noul_false_correct():
    questions = {"refund": {"type": "noul", "instructions": "..."}}
    answers = {"refund": {"type": "noul", "noul": 0.3}}
    ground_truth = {"refund": False}
    correct, total, _ = evaluate_answers(answers, ground_truth, questions)
    assert correct == 1
    assert total == 1

def test_evaluate_noul_threshold_boundary():
    questions = {"refund": {"type": "noul", "instructions": "..."}}
    # exactly 0.5 counts as True
    answers = {"refund": {"type": "noul", "noul": 0.5}}
    ground_truth = {"refund": True}
    correct, total, _ = evaluate_answers(answers, ground_truth, questions)
    assert correct == 1

def test_evaluate_score_band_correct():
    questions = {"urgency": {"type": "score", "instructions": "...", "criteria": ["Low", "High"]}}
    answers = {"urgency": {"type": "score", "score": 0.85}}
    ground_truth = {"urgency_band": "high"}
    correct, total, _ = evaluate_answers(answers, ground_truth, questions)
    assert correct == 1
    assert total == 1

def test_evaluate_score_band_wrong():
    questions = {"urgency": {"type": "score", "instructions": "...", "criteria": ["Low", "High"]}}
    answers = {"urgency": {"type": "score", "score": 0.1}}
    ground_truth = {"urgency_band": "high"}
    correct, total, _ = evaluate_answers(answers, ground_truth, questions)
    assert correct == 0
    assert total == 1

def test_evaluate_multi_question():
    questions = {
        "team": {"type": "choice", "instructions": "...", "criteria": {"billing": ""}},
        "refund": {"type": "noul", "instructions": "..."},
        "urgency": {"type": "score", "instructions": "...", "criteria": ["Low", "High"]},
    }
    answers = {
        "team": {"type": "choice", "choice": "billing", "confidence": 0.95},
        "refund": {"type": "noul", "noul": 0.9},
        "urgency": {"type": "score", "score": 0.8},
    }
    ground_truth = {"team": "billing", "refund": True, "urgency_band": "high"}
    correct, total, avg_conf = evaluate_answers(answers, ground_truth, questions)
    assert correct == 3
    assert total == 3
    assert avg_conf == pytest.approx(0.95)

def test_evaluate_missing_ground_truth_key_skipped():
    questions = {"team": {"type": "choice", "instructions": "...", "criteria": {}}}
    answers = {"team": {"type": "choice", "choice": "billing"}}
    ground_truth = {}  # no ground truth for "team"
    correct, total, _ = evaluate_answers(answers, ground_truth, questions)
    assert total == 0

def test_evaluate_missing_answer_counts_as_wrong():
    questions = {"team": {"type": "choice", "instructions": "...", "criteria": {}}}
    answers = {}  # model gave no answer
    ground_truth = {"team": "billing"}
    correct, total, _ = evaluate_answers(answers, ground_truth, questions)
    assert correct == 0
    assert total == 1


# --- Dataset sanity checks ---

def test_dataset_has_all_problems():
    assert set(DATASET.keys()) == {"ecommerce", "healthcare", "finance"}

def test_dataset_sample_count():
    for key, samples in DATASET.items():
        assert len(samples) == 15, f"{key} should have 15 samples, got {len(samples)}"

def test_dataset_samples_have_state_and_ground_truth():
    for prob_key, samples in DATASET.items():
        for i, s in enumerate(samples):
            assert "state" in s, f"{prob_key}[{i}] missing 'state'"
            assert "ground_truth" in s, f"{prob_key}[{i}] missing 'ground_truth'"
            assert isinstance(s["state"], str) and len(s["state"]) > 0

def test_ecommerce_ground_truth_keys():
    for i, s in enumerate(DATASET["ecommerce"]):
        gt = s["ground_truth"]
        assert "team" in gt, f"ecommerce[{i}] missing 'team'"
        assert gt["team"] in ("billing", "bug", "account")
        assert "refund" in gt
        assert isinstance(gt["refund"], bool)
        assert "urgency_band" in gt
        assert gt["urgency_band"] in ("low", "mid", "high")

def test_healthcare_ground_truth_keys():
    for i, s in enumerate(DATASET["healthcare"]):
        gt = s["ground_truth"]
        assert "urgency" in gt
        assert gt["urgency"] in ("emergency", "urgent", "routine")
        assert "specialty" in gt
        assert gt["specialty"] in ("cardiology", "neurology", "general", "other")

def test_finance_ground_truth_keys():
    for i, s in enumerate(DATASET["finance"]):
        gt = s["ground_truth"]
        assert "verdict" in gt
        assert gt["verdict"] in ("fraud", "legitimate", "review")
        assert "block" in gt
        assert isinstance(gt["block"], bool)


# --- Model list ---

def test_all_models_list():
    assert JEV_MODEL == "jev"
    assert "nimble" in OLLAMA_MODELS
    assert "tev1" in OLLAMA_MODELS
    assert "tev1:0.8b" in OLLAMA_MODELS
    assert JEV_MODEL in ALL_MODELS
