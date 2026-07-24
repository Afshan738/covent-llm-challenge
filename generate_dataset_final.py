"""
Covent LLM Challenge - Training Data Generator
Pulls real parcel records from Cook County (IL) and DuPage County (IL),
maps them into a canonical schema, and generates (messy_input -> clean_json)
pairs for fine-tuning a normalizer model.

Usage:
    python generate_dataset.py --per_county 1500 --out train.jsonl
"""

import argparse
import json
import random
import time
import requests

COOK_URL = "https://datacatalog.cookcountyil.gov/resource/3723-97qp.json"
DUPAGE_URL = "https://gis.dupageco.org/arcgis/rest/services/DuPage_County_IL/ParcelsWithRealEstateCC/MapServer/0/query"

# some county servers reject the default python-requests user agent, so we
# pretend to be a browser
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}


def _get_with_retry(session, url, params, timeout=30, retries=4):
    """GET with exponential backoff, since county servers are sometimes slow or flaky."""
    last_err = None
    for attempt in range(retries):
        try:
            resp = session.get(url, params=params, timeout=timeout, headers=HEADERS)
            resp.raise_for_status()
            return resp
        except (requests.exceptions.ConnectTimeout,
                requests.exceptions.ReadTimeout,
                requests.exceptions.ConnectionError) as e:
            last_err = e
            wait = 2 ** attempt
            print(f"  request failed ({e.__class__.__name__}), retrying in {wait}s...")
            time.sleep(wait)
    raise last_err


def fetch_cook_records(limit=1000, offset=0, session=None):
    """Cook County parcel address/owner records, via their Socrata open data API."""
    session = session or requests
    params = {"$limit": limit, "$offset": offset, "$order": "year DESC"}
    resp = _get_with_retry(session, COOK_URL, params, timeout=30)
    return resp.json()


def fetch_dupage_records(limit=1000, offset=0, session=None):
    """DuPage County parcel + billing records, via their ArcGIS REST service."""
    session = session or requests
    fields = ("PIN,BILLNAME,BILLADDRL1,BILLADDRL2,BILLCITY,BILLSTATE,BILLZIP,"
              "BILLZIPSUF,PROPSTNUM,PROPSTDIR,PROPSTNAME,PROPAPT,PROPCITY,"
              "PROPSTATE,PROPZIP,PROPZIPSUF")
    params = {
        "where": "1=1",
        "outFields": fields,
        "resultRecordCount": limit,
        "resultOffset": offset,
        "returnGeometry": "false",
        "f": "json",
    }
    resp = _get_with_retry(session, DUPAGE_URL, params, timeout=30)
    data = resp.json()
    return [f["attributes"] for f in data.get("features", [])]


def map_cook_record(rec):
    if not rec.get("prop_address_full") or not rec.get("pin"):
        return None
    return {
        "parcel_id": rec.get("pin"),
        "situs_address": rec.get("prop_address_full"),
        "situs_city": (rec.get("prop_address_city_name") or "").title(),
        "situs_state": rec.get("prop_address_state"),
        "situs_zip": rec.get("prop_address_zipcode_1"),
        "owner_name": rec.get("owner_address_name"),
        "owner_mailing_address": rec.get("owner_address_full"),
        "owner_mailing_city": (rec.get("owner_address_city_name") or "").title(),
        "owner_mailing_state": rec.get("owner_address_state"),
        "owner_mailing_zip": rec.get("owner_address_zipcode_1"),
    }


def _clean(s):
    """DuPage's string fields are fixed-width and space-padded, so strip them."""
    return s.strip() if s else s


def map_dupage_record(rec):
    pin = _clean(rec.get("PIN"))
    # PROPSTNUM/PROPSTDIR are usually empty in this dataset even though the
    # schema defines them - the full address is normally just sitting in
    # PROPSTNAME instead, so fall back to that when the split fields aren't there
    parts = [rec.get("PROPSTNUM"), rec.get("PROPSTDIR"),
             _clean(rec.get("PROPSTNAME")), rec.get("PROPAPT")]
    situs_address = _clean(rec.get("PROPSTNAME")) if not rec.get("PROPSTNUM") else \
        " ".join(p for p in parts if p)
    if not situs_address or not pin:
        return None
    return {
        "parcel_id": pin,
        "situs_address": situs_address,
        "situs_city": _clean(rec.get("PROPCITY")),
        "situs_state": rec.get("PROPSTATE"),
        "situs_zip": rec.get("PROPZIP"),
        "owner_name": _clean(rec.get("BILLNAME")),
        "owner_mailing_address": _clean(rec.get("BILLADDRL1")),
        "owner_mailing_city": _clean(rec.get("BILLCITY")),
        "owner_mailing_state": rec.get("BILLSTATE"),
        "owner_mailing_zip": rec.get("BILLZIP"),
    }


# turning a clean record into a messy, realistic-looking input string

STREET_ABBR = {
    "STREET": "ST", "AVENUE": "AVE", "BOULEVARD": "BLVD", "DRIVE": "DR",
    "ROAD": "RD", "LANE": "LN", "COURT": "CT", "PLACE": "PL",
    "TRAIL": "TRL", "CIRCLE": "CIR", "TERRACE": "TER", "PARKWAY": "PKWY",
}
STREET_EXPAND = {v: k for k, v in STREET_ABBR.items()}

CITY_ABBR = {
    "CHICAGO HEIGHTS": "CHICAGO HTS",
    "SAINT": "ST",
}


def _rand_case(s):
    style = random.choice(["upper", "lower", "title", "asis"])
    if style == "upper":
        return s.upper()
    if style == "lower":
        return s.lower()
    if style == "title":
        return s.title()
    return s


def _swap_street_suffix(addr):
    words = addr.split()
    for i, w in enumerate(words):
        wu = w.upper().rstrip(".")
        if wu in STREET_ABBR and random.random() < 0.5:
            words[i] = STREET_ABBR[wu]
        elif wu in STREET_EXPAND and random.random() < 0.3:
            words[i] = STREET_EXPAND[wu]
    return " ".join(words)


def _typo(s, rate=0.06):
    """Randomly drop, swap, or duplicate a character to simulate a typo."""
    if not s or random.random() > 0.4:
        return s
    chars = list(s)
    idx = random.randrange(len(chars))
    op = random.choice(["drop", "swap", "dup"])
    if op == "drop" and len(chars) > 3:
        del chars[idx]
    elif op == "swap" and idx < len(chars) - 1:
        chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
    elif op == "dup":
        chars.insert(idx, chars[idx])
    return "".join(chars)


def corrupt_record(canon):
    """
    Builds a messy free-text input simulating a raw CRM entry, derived from
    a clean canonical record. Returns the messy string; the clean canon dict
    stays untouched as the training target.
    """
    addr = _swap_street_suffix(canon["situs_address"])
    addr = _typo(addr)
    city = canon["situs_city"] or ""
    if city.upper() in CITY_ABBR and random.random() < 0.5:
        city = CITY_ABBR[city.upper()]
    city = _rand_case(city)
    state = canon["situs_state"] or ""
    zipc = canon["situs_zip"] or ""
    owner = canon["owner_name"] or ""
    owner = _typo(owner, rate=0.03)

    parts = [addr, city, state, zipc]
    if random.random() < 0.3:
        parts.remove(state)  # real listings often drop the state when it seems obvious
    sep = random.choice([", ", " ", " - "])
    address_blob = sep.join(p for p in parts if p)

    template = random.choice([
        "{addr} | Owner: {owner}",
        "{owner} - {addr}",
        "Property: {addr}. Owner on file: {owner}",
        "{addr}\nOwner: {owner}",
        "{addr} owner={owner}",
    ])
    return template.format(addr=address_blob, owner=owner)


def build_pairs(records, mapper, n):
    pairs = []
    random.shuffle(records)
    for rec in records:
        canon = mapper(rec)
        if not canon:
            continue
        if not all([canon["situs_address"], canon["owner_name"]]):
            continue
        messy = corrupt_record(canon)
        pairs.append({"input": messy, "target": canon})
        if len(pairs) >= n:
            break
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_county", type=int, default=1500)
    ap.add_argument("--out", default="train.jsonl")
    args, _unknown = ap.parse_known_args()

    session = requests.Session()

    print("Fetching Cook County records...")
    cook_raw = []
    offset = 0
    while len(cook_raw) < args.per_county * 2:  # overfetch since some rows get skipped
        batch = fetch_cook_records(limit=1000, offset=offset, session=session)
        if not batch:
            break
        cook_raw.extend(batch)
        offset += 1000
        time.sleep(0.2)
    print(f"  got {len(cook_raw)} raw Cook records")

    print("Fetching DuPage County records...")
    dupage_raw = []
    offset = 0
    while len(dupage_raw) < args.per_county * 2:
        batch = fetch_dupage_records(limit=1000, offset=offset, session=session)
        if not batch:
            break
        dupage_raw.extend(batch)
        offset += 1000
        time.sleep(0.2)
    print(f"  got {len(dupage_raw)} raw DuPage records")

    cook_pairs = build_pairs(cook_raw, map_cook_record, args.per_county)
    dupage_pairs = build_pairs(dupage_raw, map_dupage_record, args.per_county)
    print(f"Built {len(cook_pairs)} Cook pairs, {len(dupage_pairs)} DuPage pairs")

    all_pairs = cook_pairs + dupage_pairs
    random.shuffle(all_pairs)

    with open(args.out, "w") as f:
        for p in all_pairs:
            f.write(json.dumps(p) + "\n")

    print(f"Wrote {len(all_pairs)} training pairs to {args.out}")
    print("\nSample:")
    print(json.dumps(all_pairs[0], indent=2))


if __name__ == "__main__":
    main()
