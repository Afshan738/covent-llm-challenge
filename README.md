# Covent LLM Challenge: Property Record Normalizer

**A small, fine-tuned model that specializes in one narrow, real-world task: turning messy US property records into clean structured JSON  and beats Claude Opus 4.8 on a specific, well-defined sub-problem within that task.**

## The Task

US county property/parcel data is a mess. Every county publishes records through different platforms (ArcGIS, Socrata), with different field names, different address formats (some merged into one string, some split into components), inconsistent abbreviations, and inconsistent null-handling. Real estate CRM platforms like **Covent**  that ingest data across counties have to normalize this before it's usable.

**This model takes a messy, free-text-style property record and outputs clean JSON in one fixed canonical schema:**

```json
{
  "parcel_id": "", "situs_address": "", "situs_city": "", "situs_state": "",
  "situs_zip": "", "owner_name": "", "owner_mailing_address": "",
  "owner_mailing_city": "", "owner_mailing_state": "", "owner_mailing_zip": ""
}
```

This isn't a hypothetical. it's a real bottleneck I ran into firsthand while building a satellite-imagery pipeline  for Covent that needed to look up parcel geometry across multiple counties' data.

## Approach

- **Base model:** Qwen2.5-1.5B-Instruct (4-bit QLoRA via `unsloth`)
- **LoRA:** r=16, alpha=16, targeting q/k/v/o and gate/up/down projections
- **Training data:** ~2,850 real parcel records from **Cook County, IL** (Socrata API) and **DuPage County, IL** (ArcGIS REST API), each paired with a synthetically corrupted "messy" version (abbreviation swaps, typos, punctuation dropping, casing variation, 5 different free-text templates simulating raw CRM entry).
- **Training:** 2 epochs, learning rate 2e-4, batch size 8×2 grad accumulation, on a single free-tier Colab T4 GPU.

## Why a small model can win here: and where it actually does

A frontier model is a generalist: it has to infer field names, formatting conventions, and null-handling from the prompt alone, every time, with no memory of this specific schema's quirks. A small model fine-tuned on this exact distribution can specialize.

**The honest result: the fine-tuned model does not beat Opus 4.8 on the full 10-field task in aggregate.** Opus's advantage is concentrated in fields that require broad US geographic world-knowledge. It already "knows" Denver is in Colorado and what a real zip code range looks like, from its pretraining. A 1.5B model trained on two Illinois counties never saw Colorado data at all and structurally cannot know this.

**But the fine-tuned model does win on a specific, narrower sub-problem: inferring an owner's mailing address fields when they're not explicitly stated in the input but match the property's own address** a common real-world pattern (owner-occupied homes) that a model can only learn by seeing many real examples of it, not from general world knowledge. This is the "narrow task where a small model can actually win" the challenge asked for.

## Evaluation

**Test set:** 300 real records from **three counties never touched during training**: Maricopa County (AZ), Travis County (TX), Denver County (CO), corrupted the same way as training data. Genuinely unseen, satisfying the "test set must differ from training" requirement.

**Baseline:** Claude Opus 4.8, identical system prompt, called via the official Anthropic API on the same 300 examples.

**Metric:** exact-match field accuracy (case-insensitive), full-record exact match across 9 fields (parcel_id excluded, it never appears in the input text, so it's unknowable to any model by construction), and valid-JSON rate.

| Field | Fine-tuned Qwen2.5-1.5B | Claude Opus 4.8 |
|---|---|---|
| situs_address | 22.0% | 18.7% |
| situs_city | 80.0% | **99.3%** |
| situs_state | 81.0% | **94.7%** |
| situs_zip | 49.3% | **99.3%** |
| owner_name | 14.0% | **59.3%** |
| owner_mailing_address | 11.7% | **34.7%** |
| **owner_mailing_city** | **53.3%** | 36.3% |
| **owner_mailing_state** | **49.3%** | 36.3% |
| owner_mailing_zip | 36.3% | 35.7% |
| Valid JSON rate | 100% | 100% |

Both models produce valid JSON every time, so this is a clean accuracy comparison, not a formatting artifact.

**Where the fine-tuned model wins (owner_mailing_city, owner_mailing_state, and roughly ties on owner_mailing_zip):** these are exactly the fields that depend on the situs-matches-mailing pattern learned from real training data, not on general knowledge. **Where Opus wins decisively (situs_city/state/zip):** these depend on broad geographic knowledge a narrow-domain small model doesn't have.

This is also a real cost/latency argument: the fine-tuned model runs locally on a single consumer-tier GPU at near-zero marginal cost per record, versus per-token API pricing for repeated frontier-model calls at scale.

## Demo

See `demo_examples.md` for concrete before/after examples, including cases where the fine-tuned model correctly infers a mailing address Opus left as `null`.

## Reproducing This

1. `pip install -r requirements.txt` (unsloth, trl, peft, accelerate, bitsandbytes, requests, anthropic)
2. `python generate_dataset.py --per_county 1500 --out train.jsonl` , regenerates training data from live county APIs
3. Run `finetune.py` (designed for a Colab/Kaggle notebook with a T4 GPU) to reproduce training
4. `python score_final.py` , scores `local_preds_cache.json` and `opus_preds_cache.json` against `test_set.jsonl` (all included in this repo) to reproduce the eval table above without re-running inference

Model weights: [huggingface.co/Afshanqasim525/qwen25-1.5b-parcel-normalizer](https://huggingface.co/Afshanqasim525/qwen25-1.5b-parcel-normalizer)

## Limitations (honest, not hedged)

- Training data comes from only two Illinois counties. The eval shows real generalization to three other states, but performance would likely improve with more geographic diversity in training data.
- This is a proof-of-concept demonstrating where narrow-task specialization wins, not a production-ready system ,a real deployment would need broader training data and more rigorous edge-case handling.
- The "messy" input noise is synthetically generated from real underlying data (real addresses/owners/parcel IDs, synthetic corruption). Real-world noise from OCR, scraped listings, or legacy CRM exports may include patterns this corruption function doesn't simulate.
- I originally planned to benchmark against Gemini 3.1 Pro as a cost-effective option consistent with the challenge's named models, but hit a regional free-tier access restriction on the Gemini API unrelated to usage volume. I used Claude Opus 4.8 via the standard API instead, which is also one of the three named models in the challenge.

## About Me
- I recently graduated with a BS in Information Technology. This project builds on hands-on experience with the exact kind of messy multi-source county data problem it solves
- I independently built a geospatial imagery pipeline prototype and reached out to Covent about it, which is where I first ran into this exact data-normalization problem firsthand.
