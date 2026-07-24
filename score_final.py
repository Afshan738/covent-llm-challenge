import json, re

CANON_KEYS = [
    "parcel_id", "situs_address", "situs_city", "situs_state", "situs_zip",
    "owner_name", "owner_mailing_address", "owner_mailing_city",
    "owner_mailing_state", "owner_mailing_zip",
]
EVAL_KEYS = [k for k in CANON_KEYS if k != "parcel_id"]  

def extract_json(text):
    if not text:
        return None
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

def score(predictions, targets, eval_keys=EVAL_KEYS):
    n = len(targets)
    valid_json = 0
    full_match = 0
    field_correct = {k: 0 for k in eval_keys}
    field_total = {k: 0 for k in eval_keys}
    for pred_raw, target in zip(predictions, targets):
        pred = extract_json(pred_raw)
        if pred is not None:
            valid_json += 1
        else:
            pred = {}
        record_ok = True
        for k in eval_keys:
            field_total[k] += 1
            target_val = str(target.get(k) or "").strip().lower()
            pred_val = str(pred.get(k) or "").strip().lower()
            if target_val == pred_val:
                field_correct[k] += 1
            else:
                record_ok = False
        if record_ok:
            full_match += 1
    return {
        "n": n,
        "valid_json_rate": round(valid_json / n, 4) if n else 0,
        "full_record_exact_match_9_fields": round(full_match / n, 4) if n else 0,
        "per_field_accuracy": {k: round(field_correct[k] / field_total[k], 4) for k in eval_keys},
    }

with open("test_set.jsonl") as f:
    test_pairs = [json.loads(line) for line in f]
with open("local_preds_cache.json") as f:
    local_preds = json.load(f)
with open("opus_preds_cache.json") as f:
    opus_preds = json.load(f)

targets = [p["target"] for p in test_pairs]
counties = sorted(set(p["county"] for p in test_pairs))

results = {
    "n_test_pairs": len(test_pairs),
    "held_out_counties": counties,
    "fine_tuned_qwen25_1_5b": score(local_preds, targets),
    "claude_opus_4_8": score(opus_preds, targets),
}

with open("FINAL_eval_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
print("\nSaved to FINAL_eval_results.json")
