"""
Jev-style Decision Model Evaluation
Compares: jev (TypeSafe cloud), nimble (9B), tev1 (4B), tev1:0.8b
Across 3 industries: E-commerce, Healthcare, Finance
"""

import argparse
import httpx
import os
import time
import subprocess
import sys

# load .env if present
_env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(_env_path):
    for _line in open(_env_path):
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

# --- Configuration ---

OLLAMA_URL = "http://localhost:11434/v1/systemone"
JEV_URL = "https://jevtypesafeai.com/api/v1/decide"
JEV_API_KEY = os.getenv("JEV_API_KEY", "")

OLLAMA_MODELS = ["nimble", "tev1", "tev1:0.8b"]
JEV_MODEL = "jev"
ALL_MODELS = [JEV_MODEL] + OLLAMA_MODELS

# --- Problem Definitions ---

PROBLEMS = {
    "ecommerce": {
        "name": "E-commerce Ticket Triage",
        "questions": {
            "team": {
                "type": "choice",
                "instructions": "Which team should handle this ticket?",
                "criteria": {
                    "billing": "Payments, charges, and refunds",
                    "bug": "Software bugs and technical issues",
                    "account": "Login, account access, and profile",
                },
            },
            "refund": {
                "type": "noul",
                "instructions": "Does the customer explicitly ask for a refund?",
            },
            "urgency": {
                "type": "score",
                "instructions": "How urgent is this ticket?",
                "criteria": ["Routine", "Soon", "Urgent"],
            },
        },
    },
    "healthcare": {
        "name": "Healthcare Symptom Urgency",
        "questions": {
            "urgency": {
                "type": "choice",
                "instructions": "What is the urgency level for this patient?",
                "criteria": {
                    "emergency": "Immediate life-threatening situation",
                    "urgent": "Needs care within 24 hours",
                    "routine": "Can wait for scheduled appointment",
                },
            },
            "specialty": {
                "type": "choice",
                "instructions": "Which specialty should see this patient?",
                "criteria": {
                    "cardiology": "Heart and cardiovascular conditions",
                    "neurology": "Brain and nervous system conditions",
                    "general": "General practitioner",
                    "other": "Other specialty required",
                },
            },
            "follow_up": {
                "type": "noul",
                "instructions": "Does the patient need a follow-up within 24 hours?",
            },
        },
    },
    "finance": {
        "name": "Finance Fraud Detection",
        "questions": {
            "verdict": {
                "type": "choice",
                "instructions": "Is this transaction fraudulent?",
                "criteria": {
                    "fraud": "Likely fraudulent transaction",
                    "legitimate": "Normal legitimate transaction",
                    "review": "Unusual but needs manual review",
                },
            },
            "risk": {
                "type": "score",
                "instructions": "What is the fraud risk level of this transaction?",
                "criteria": ["Low", "Medium", "High"],
            },
            "block": {
                "type": "noul",
                "instructions": "Should this transaction be blocked immediately?",
            },
        },
    },
}

# --- Synthetic Dataset with Ground Truth ---

DATASET = {
    "ecommerce": [
        {
            "state": "I was charged twice for order #12345. Please refund the extra charge immediately.",
            "ground_truth": {"team": "billing", "refund": True, "urgency_band": "high"},
        },
        {
            "state": "The checkout button doesn't work on Safari. I can't complete my purchase.",
            "ground_truth": {"team": "bug", "refund": False, "urgency_band": "mid"},
        },
        {
            "state": "I can't log into my account. It says my password is wrong but I reset it twice.",
            "ground_truth": {"team": "account", "refund": False, "urgency_band": "mid"},
        },
        {
            "state": "My credit card was charged $299 but I only ordered the $29 item.",
            "ground_truth": {"team": "billing", "refund": True, "urgency_band": "high"},
        },
        {
            "state": "The app crashes every time I open the product page for category shoes.",
            "ground_truth": {"team": "bug", "refund": False, "urgency_band": "low"},
        },
        {
            "state": "I never received a confirmation email after placing my order.",
            "ground_truth": {"team": "account", "refund": False, "urgency_band": "low"},
        },
        {
            "state": "You charged me for a subscription I cancelled 3 months ago. I want my money back.",
            "ground_truth": {"team": "billing", "refund": True, "urgency_band": "high"},
        },
        {
            "state": "Images are not loading on the product listing page.",
            "ground_truth": {"team": "bug", "refund": False, "urgency_band": "low"},
        },
        {
            "state": "I need to update my shipping address but the form won't save.",
            "ground_truth": {"team": "account", "refund": False, "urgency_band": "low"},
        },
        {
            "state": "Wrong item was shipped. I ordered blue but received red. I need a refund.",
            "ground_truth": {"team": "billing", "refund": True, "urgency_band": "high"},
        },
        {
            "state": "Discount code SAVE20 is not applying at checkout. Shows an error.",
            "ground_truth": {"team": "bug", "refund": False, "urgency_band": "mid"},
        },
        {
            "state": "My order history shows duplicate orders I didn't place.",
            "ground_truth": {"team": "account", "refund": False, "urgency_band": "mid"},
        },
        {
            "state": "I was charged a delivery fee but I selected free shipping at checkout.",
            "ground_truth": {"team": "billing", "refund": True, "urgency_band": "mid"},
        },
        {
            "state": "Search results don't filter by price. The filter slider is broken.",
            "ground_truth": {"team": "bug", "refund": False, "urgency_band": "low"},
        },
        {
            "state": "My two-factor authentication code is not being accepted.",
            "ground_truth": {"team": "account", "refund": False, "urgency_band": "mid"},
        },
    ],
    "healthcare": [
        {
            "state": "Severe chest pain radiating to left arm, sweating heavily, started 20 minutes ago.",
            "ground_truth": {"urgency": "emergency", "specialty": "cardiology"},
        },
        {
            "state": "Mild headache for 2 days, no fever, probably stress from work.",
            "ground_truth": {"urgency": "routine", "specialty": "general"},
        },
        {
            "state": "Sudden severe headache described as worst of my life, some neck stiffness.",
            "ground_truth": {"urgency": "emergency", "specialty": "neurology"},
        },
        {
            "state": "Blood pressure reading at home was 170/110, feeling dizzy.",
            "ground_truth": {"urgency": "urgent", "specialty": "cardiology"},
        },
        {
            "state": "Persistent cough for 3 weeks, no fever, slight shortness of breath.",
            "ground_truth": {"urgency": "urgent", "specialty": "general"},
        },
        {
            "state": "Minor cut on finger from cooking, cleaned and bandaged it already.",
            "ground_truth": {"urgency": "routine", "specialty": "general"},
        },
        {
            "state": "Numbness and tingling in left arm and face for the past hour.",
            "ground_truth": {"urgency": "emergency", "specialty": "neurology"},
        },
        {
            "state": "Routine annual checkup needed, feeling fine overall.",
            "ground_truth": {"urgency": "routine", "specialty": "general"},
        },
        {
            "state": "Palpitations and irregular heartbeat noticed for past 3 hours.",
            "ground_truth": {"urgency": "urgent", "specialty": "cardiology"},
        },
        {
            "state": "Migraine with aura, similar to previous episodes, took my usual medication.",
            "ground_truth": {"urgency": "routine", "specialty": "neurology"},
        },
        {
            "state": "Difficulty breathing after allergic reaction to peanuts, used EpiPen.",
            "ground_truth": {"urgency": "emergency", "specialty": "other"},
        },
        {
            "state": "Knee pain after running, mild swelling, can still walk.",
            "ground_truth": {"urgency": "routine", "specialty": "other"},
        },
        {
            "state": "Chest tightness and wheezing, asthma inhaler is not helping.",
            "ground_truth": {"urgency": "urgent", "specialty": "general"},
        },
        {
            "state": "Vision changes in left eye, seeing flashing lights and floaters since this morning.",
            "ground_truth": {"urgency": "urgent", "specialty": "neurology"},
        },
        {
            "state": "Fever of 103F for 2 days, body aches, possible flu.",
            "ground_truth": {"urgency": "urgent", "specialty": "general"},
        },
    ],
    "finance": [
        {
            "state": "ATM withdrawal of $800 in Lagos, Nigeria. Account holder lives in Chicago, IL. No travel notice on file.",
            "ground_truth": {"verdict": "fraud", "block": True},
        },
        {
            "state": "Monthly Netflix subscription charge $15.99. Regular recurring transaction for 18 months.",
            "ground_truth": {"verdict": "legitimate", "block": False},
        },
        {
            "state": "Amazon purchase $234.50, same device and IP as last 50 transactions.",
            "ground_truth": {"verdict": "legitimate", "block": False},
        },
        {
            "state": "Wire transfer $9,800 to unknown overseas account. First international wire for this account.",
            "ground_truth": {"verdict": "fraud", "block": True},
        },
        {
            "state": "Gas station purchase $45.20 near user's registered home address.",
            "ground_truth": {"verdict": "legitimate", "block": False},
        },
        {
            "state": "5 declined transactions in 10 minutes followed by a $1 test charge.",
            "ground_truth": {"verdict": "fraud", "block": True},
        },
        {
            "state": "Grocery store purchase $87.30. User shops here every week.",
            "ground_truth": {"verdict": "legitimate", "block": False},
        },
        {
            "state": "Online purchase $1,200 gaming laptop. New merchant, new device, shipping to different address than billing.",
            "ground_truth": {"verdict": "review", "block": False},
        },
        {
            "state": "Credit card used in Miami and New York within 2 hours. Both charges over $500.",
            "ground_truth": {"verdict": "fraud", "block": True},
        },
        {
            "state": "Payroll direct deposit $3,450 from known employer, same amount every 2 weeks.",
            "ground_truth": {"verdict": "legitimate", "block": False},
        },
        {
            "state": "Dental insurance copay $25 at registered dentist office.",
            "ground_truth": {"verdict": "legitimate", "block": False},
        },
        {
            "state": "Cryptocurrency purchase $5,000 on new exchange. Account only 2 months old.",
            "ground_truth": {"verdict": "review", "block": False},
        },
        {
            "state": "Pharmacy purchase $12.50 near user's home address.",
            "ground_truth": {"verdict": "legitimate", "block": False},
        },
        {
            "state": "Multiple $200 gift card purchases at 3 different stores within 1 hour.",
            "ground_truth": {"verdict": "fraud", "block": True},
        },
        {
            "state": "Utility bill payment $120 to same provider as last 24 months.",
            "ground_truth": {"verdict": "legitimate", "block": False},
        },
    ],
}

# --- Core Functions ---


def run_decision(
    model: str, state: str, questions: dict
) -> tuple:
    """
    Call /v1/systemone. Returns (answers, latency_ms, usage).
    Routes to TypeSafe cloud for 'jev', Ollama local for all others.
    """
    if model == JEV_MODEL:
        payload = {"state": state, "questions": questions}
        start = time.time()
        resp = httpx.post(
            JEV_URL,
            json=payload,
            headers={"Authorization": f"Bearer {JEV_API_KEY}"},
            timeout=30.0,
        )
    else:
        payload = {"model": model, "state": state, "questions": questions}
        start = time.time()
        resp = httpx.post(OLLAMA_URL, json=payload, timeout=60.0)

    resp.raise_for_status()
    latency_ms = (time.time() - start) * 1000
    data = resp.json()
    return data["answers"], latency_ms, data.get("usage", {})


def score_to_band(score: float) -> str:
    """Map 0-1 score to low/mid/high band."""
    if score < 0.33:
        return "low"
    elif score < 0.67:
        return "mid"
    return "high"


def evaluate_answers(answers: dict, ground_truth: dict, questions: dict) -> tuple:
    """
    Compare model answers to ground truth.
    Returns (correct_count, total_count, avg_confidence).
    - choice: exact match
    - noul: threshold 0.5 (>=0.5 -> True)
    - score: banded match (low/mid/high)
    """
    correct = 0
    total = 0
    confidences = []

    for q_name, q_def in questions.items():
        q_type = q_def["type"]
        answer = answers.get(q_name, {})

        if q_type == "choice":
            gt = ground_truth.get(q_name)
            if gt is None:
                continue
            total += 1
            if answer.get("choice") == gt:
                correct += 1
            if "confidence" in answer:
                confidences.append(answer["confidence"])

        elif q_type == "noul":
            gt = ground_truth.get(q_name)
            if gt is None:
                continue
            total += 1
            predicted = answer.get("noul", 0.0) >= 0.5
            if predicted == gt:
                correct += 1

        elif q_type == "score":
            # ground truth key is <q_name>_band
            gt_band = ground_truth.get(f"{q_name}_band")
            if gt_band is None:
                continue
            total += 1
            predicted_band = score_to_band(answer.get("score", 0.0))
            if predicted_band == gt_band:
                correct += 1

    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return correct, total, avg_conf


def run_problem(model: str, problem_key: str, dataset: dict = None) -> dict:
    """
    Run model on all samples for one industry problem.
    Returns metrics dict with accuracy, avg_confidence, avg_latency_ms, total_tokens, errors.
    """
    if dataset is None:
        dataset = DATASET
    problem = PROBLEMS[problem_key]
    samples = dataset[problem_key]

    total_correct = 0
    total_questions = 0
    latencies = []
    confidences = []
    total_tokens = 0
    errors = 0

    print(f"  [{model}] {problem['name']}...", end="", flush=True)

    for sample in samples:
        try:
            answers, latency_ms, usage = run_decision(
                model, sample["state"], problem["questions"]
            )
            correct, total, avg_conf = evaluate_answers(
                answers, sample["ground_truth"], problem["questions"]
            )
            total_correct += correct
            total_questions += total
            latencies.append(latency_ms)
            if avg_conf > 0:
                confidences.append(avg_conf)
            total_tokens += usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        except Exception as e:
            errors += 1
            print(f"\n    error: {e}", end="", flush=True)

    accuracy = (total_correct / total_questions * 100) if total_questions > 0 else 0.0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

    print(f" done ({len(samples) - errors}/{len(samples)} ok)", flush=True)

    return {
        "accuracy": accuracy,
        "avg_confidence": avg_conf,
        "avg_latency_ms": avg_latency,
        "total_tokens": total_tokens,
        "errors": errors,
        "samples_ok": len(samples) - errors,
    }


def print_report(results: dict, models: list = None) -> None:
    """Print per-problem and overall comparison table."""
    if models is None:
        models = ALL_MODELS
    problems = list(PROBLEMS.keys())

    print("\n" + "=" * 72)
    print("  Jev-Style Decision Model Evaluation")
    print("=" * 72)

    for prob_key in problems:
        prob_name = PROBLEMS[prob_key]["name"]
        n_samples = len(DATASET[prob_key])
        print(f"\n{prob_name} — {n_samples} samples")
        print(
            f"  {'Model':<14} {'Accuracy':>10} {'Avg Conf':>10} {'Avg Latency':>13} {'Tokens':>8} {'Errors':>7}"
        )
        print("  " + "-" * 64)
        for model in models:
            r = results[model][prob_key]
            print(
                f"  {model:<14} {r['accuracy']:>9.1f}% {r['avg_confidence']:>10.3f}"
                f" {r['avg_latency_ms']:>11.0f}ms {r['total_tokens']:>8} {r['errors']:>7}"
            )

    print(f"\nOVERALL SUMMARY")
    print(
        f"  {'Model':<14} {'Mean Acc':>10} {'Mean Conf':>10} {'Mean Latency':>13}"
    )
    print("  " + "-" * 50)
    for model in models:
        mean_acc = sum(results[model][p]["accuracy"] for p in problems) / len(problems)
        mean_conf = sum(results[model][p]["avg_confidence"] for p in problems) / len(problems)
        mean_lat = sum(results[model][p]["avg_latency_ms"] for p in problems) / len(problems)
        print(
            f"  {model:<14} {mean_acc:>9.1f}% {mean_conf:>10.3f} {mean_lat:>11.0f}ms"
        )

    print("=" * 72)


def pull_ollama_models() -> None:
    """Pull all required Ollama decision models."""
    print("Pulling Ollama decision models...")
    for model in OLLAMA_MODELS:
        print(f"  ollama pull {model}")
        result = subprocess.run(
            ["ollama", "pull", model],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"  WARNING: failed to pull {model}: {result.stderr.strip()}")
        else:
            print(f"  {model} ready")


def main() -> None:
    parser = argparse.ArgumentParser(description="Jev-style decision model eval")
    parser.add_argument("--hf", action="store_true", help="Load real data from HuggingFace instead of synthetic samples")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Which models to run (e.g. --models jev tev1:0.8b). Defaults to all.")
    args = parser.parse_args()

    if args.hf:
        from data_hf import load_all
        dataset = load_all()
    else:
        dataset = DATASET

    models_to_run = args.models if args.models else ALL_MODELS
    # skip pull for jev (cloud) and only pull requested ollama models
    ollama_to_pull = [m for m in models_to_run if m != JEV_MODEL]
    if ollama_to_pull:
        print("Checking Ollama models...")
        installed = {line.split()[0] for line in subprocess.check_output(["ollama", "list"]).decode().splitlines()[1:] if line.strip()}
        for m in ollama_to_pull:
            if m not in installed:
                print(f"  pulling {m}...")
                subprocess.run(["ollama", "pull", m], check=True)
            else:
                print(f"  {m} already installed")

    n = len(next(iter(dataset.values())))
    print(f"\nRunning evaluation: {models_to_run} × 3 industries × {n} samples...\n")
    results: dict = {}
    for model in models_to_run:
        results[model] = {}
        for prob_key in PROBLEMS:
            results[model][prob_key] = run_problem(model, prob_key, dataset)

    print_report(results, models_to_run)


if __name__ == "__main__":
    main()
