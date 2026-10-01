"""
Load real data from HuggingFace datasets.
Returns DATASET-compatible dict with same schema as eval.py synthetic data.

Datasets used:
  E-commerce : bitext/Bitext-customer-support-llm-chatbot-training-dataset
  Healthcare : gretelai/symptom_to_diagnosis
  Finance    : SetFit/enron_spam  (email phishing / fraud classification)
"""

from datasets import load_dataset
import random

random.seed(42)
N = 20  # samples per industry


# ── E-COMMERCE ────────────────────────────────────────────────────────────────
# bitext customer support: instruction text + intent label

BILLING_INTENTS = {
    "payment_issue", "refund_request", "track_refund", "get_refund",
    "check_refund_policy", "get_invoice", "check_invoice",
    "check_payment_methods", "check_cancellation_fee",
}
ACCOUNT_INTENTS = {
    "registration_problems", "edit_account", "recover_password",
    "switch_account", "newsletter_subscription", "set_up_shipping_address",
}

def intent_to_team(intent: str) -> str:
    if intent in BILLING_INTENTS:
        return "billing"
    if intent in ACCOUNT_INTENTS:
        return "account"
    return "bug"

def intent_to_refund(intent: str) -> bool:
    return intent in {"refund_request", "get_refund", "track_refund"}

def intent_to_urgency_band(intent: str) -> str:
    high = {"refund_request", "payment_issue", "get_refund"}
    low  = {"newsletter_subscription", "get_invoice", "check_refund_policy"}
    if intent in high:
        return "high"
    if intent in low:
        return "low"
    return "mid"


def load_ecommerce(n: int = N) -> list[dict]:
    ds = load_dataset(
        "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
        split="train",
    )
    by_team: dict[str, list] = {"billing": [], "bug": [], "account": []}
    for row in ds:
        team = intent_to_team(row["intent"])
        by_team[team].append(row)

    samples = []
    per_team = n // 3
    for rows in by_team.values():
        for row in random.sample(rows, min(per_team, len(rows))):
            samples.append({
                "state": row["instruction"],
                "ground_truth": {
                    "team": intent_to_team(row["intent"]),
                    "refund": intent_to_refund(row["intent"]),
                    "urgency_band": intent_to_urgency_band(row["intent"]),
                },
            })

    random.shuffle(samples)
    return samples[:n]


# ── HEALTHCARE ────────────────────────────────────────────────────────────────
# gretelai/symptom_to_diagnosis: symptom description → disease name
# We map disease → urgency + specialty

EMERGENCY_DISEASES = {
    "heart attack", "myocardial infarction", "stroke", "cardiac arrest",
    "pulmonary embolism", "aortic aneurysm", "hemorrhagic stroke",
    "meningitis", "sepsis", "anaphylaxis",
}
URGENT_DISEASES = {
    "pneumonia", "appendicitis", "kidney failure", "dengue", "typhoid",
    "malaria", "diabetes", "hypertension", "urinary tract infection",
    "heart disease", "bronchitis", "asthma",
}

CARDIOLOGY_DISEASES = {
    "heart attack", "myocardial infarction", "cardiac arrest", "heart disease",
    "hypertension", "arrhythmia", "coronary artery disease", "angina",
}
NEUROLOGY_DISEASES = {
    "stroke", "migraine", "epilepsy", "alzheimer's disease", "parkinson's disease",
    "cervical spondylosis", "brain tumor", "multiple sclerosis",
}

def disease_to_urgency(disease: str) -> str:
    d = disease.lower()
    if any(e in d for e in EMERGENCY_DISEASES):
        return "emergency"
    if any(u in d for u in URGENT_DISEASES):
        return "urgent"
    return "routine"

def disease_to_specialty(disease: str) -> str:
    d = disease.lower()
    if any(c in d for c in CARDIOLOGY_DISEASES):
        return "cardiology"
    if any(n in d for n in NEUROLOGY_DISEASES):
        return "neurology"
    if d in {"diabetes", "pneumonia", "typhoid", "dengue", "malaria",
             "urinary tract infection", "asthma", "bronchitis"}:
        return "general"
    return "other"


def load_healthcare(n: int = N) -> list[dict]:
    ds = load_dataset("gretelai/symptom_to_diagnosis", split="train")

    # stratify across urgency levels
    by_urgency: dict[str, list] = {"emergency": [], "urgent": [], "routine": []}
    for row in ds:
        disease = row["output_text"].strip()
        urgency = disease_to_urgency(disease)
        state = row["input_text"].strip()
        if len(state) < 20:
            continue
        by_urgency[urgency].append({
            "state": state[:400],
            "ground_truth": {
                "urgency": urgency,
                "specialty": disease_to_specialty(disease),
            },
        })

    samples = []
    per_level = n // 3
    for rows in by_urgency.values():
        samples.extend(random.sample(rows, min(per_level, len(rows))))

    # fill remainder from any bucket
    all_rows = [r for rows in by_urgency.values() for r in rows]
    while len(samples) < n and all_rows:
        candidate = random.choice(all_rows)
        if candidate not in samples:
            samples.append(candidate)

    random.shuffle(samples)
    return samples[:n]


# ── FINANCE ───────────────────────────────────────────────────────────────────
# SetFit/enron_spam: email text + spam label
# spam = phishing / fraud attempt; ham = legitimate email

FRAUD_KEYWORDS = {
    "million", "lottery", "prize", "winner", "bank account", "transfer",
    "credit card", "paypal", "wire transfer", "urgent", "confidential",
    "nigerian", "inheritance", "claim", "free money", "investment",
}

def email_to_verdict(label: str, text: str) -> str:
    if label == "spam":
        return "fraud"
    lower = text.lower()
    hits = sum(1 for k in FRAUD_KEYWORDS if k in lower)
    if hits >= 2:
        return "review"
    return "legitimate"

def email_to_block(label: str) -> bool:
    return label == "spam"


def load_finance(n: int = N) -> list[dict]:
    ds = load_dataset("SetFit/enron_spam", split="train")

    spam_rows = [r for r in ds if r["label_text"] == "spam" and len(r["text"]) > 30]
    ham_rows  = [r for r in ds if r["label_text"] == "ham"  and len(r["text"]) > 30]

    chosen_spam = random.sample(spam_rows, min(n // 2, len(spam_rows)))
    chosen_ham  = random.sample(ham_rows,  min(n - len(chosen_spam), len(ham_rows)))
    chosen = chosen_spam + chosen_ham
    random.shuffle(chosen)

    samples = []
    for row in chosen[:n]:
        text    = row["text"][:300].strip()
        label   = row["label_text"]
        verdict = email_to_verdict(label, text)
        samples.append({
            "state": f"Email received: {text}",
            "ground_truth": {
                "verdict": verdict,
                "block": email_to_block(label),
            },
        })
    return samples


# ── Public API ────────────────────────────────────────────────────────────────

def load_all(n: int = N) -> dict[str, list[dict]]:
    print("Loading HuggingFace datasets...")
    print("  [1/3] bitext/Bitext-customer-support-llm-chatbot-training-dataset")
    ecommerce = load_ecommerce(n)
    print(f"        {len(ecommerce)} samples")

    print("  [2/3] gretelai/symptom_to_diagnosis")
    healthcare = load_healthcare(n)
    print(f"        {len(healthcare)} samples")

    print("  [3/3] SetFit/enron_spam")
    finance = load_finance(n)
    print(f"        {len(finance)} samples")

    return {"ecommerce": ecommerce, "healthcare": healthcare, "finance": finance}


if __name__ == "__main__":
    data = load_all()
    for prob, samples in data.items():
        print(f"\n── {prob} ({len(samples)} samples) ──")
        for s in samples[:2]:
            print(f"  state:        {s['state'][:90]}...")
            print(f"  ground_truth: {s['ground_truth']}")
