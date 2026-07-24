# Demo Examples

Real before/after cases from the held-out test set (Maricopa, Travis, Denver counties — never seen in training).

## Example (Maricopa County)

**Raw messy input:**

```
702 S 114TH LN AVONDALE 5323, Avondale, AZ, 85323
Owner: NGUYEN AN K/NHAN
```

**Fine-tuned model output:**

```json
{
  "owner_mailing_address": "702 S 114TH ST",
  "owner_mailing_city": "Avondale",
  "owner_mailing_state": "AZ",
  "owner_mailing_zip": "85323",
  "owner_name": "NGUYEN AN KHANH",
  "parcel_id": "0216109001",
  "situs_address": "702 S 114TH LN",
  "situs_city": "Avondale",
  "situs_state": "AZ",
  "situs_zip": "85323"
}
```

**Claude Opus 4.8 output:**

```json
{
  "parcel_id": null,
  "situs_address": "702 S 114TH LN",
  "situs_city": "Avondale",
  "situs_state": "AZ",
  "situs_zip": "85323",
  "owner_name": "NGUYEN AN K/NHAN",
  "owner_mailing_address": null,
  "owner_mailing_city": null,
  "owner_mailing_state": null,
  "owner_mailing_zip": null
}
```

**Ground truth:**

```json
{
  "parcel_id": "10101019",
  "situs_address": "702 S 114TH LN   AVONDALE  85323",
  "situs_city": "AVONDALE",
  "situs_state": "AZ",
  "situs_zip": "85323",
  "owner_name": "NGUYEN AN K/NHAN",
  "owner_mailing_address": "702 S 114TH LN",
  "owner_mailing_city": "AVONDALE",
  "owner_mailing_state": "AZ",
  "owner_mailing_zip": "85323"
}
```

---

## Example (Maricopa County)

**Raw messy input:**

```
706 S 114TH LNA VONDALE 85323 - Avondale - AZ - 85323
Owner: SILIVAN LAURA M
```

**Fine-tuned model output:**

```json
{
  "owner_mailing_address": "706 E 194TH ST",
  "owner_mailing_city": "Avondale",
  "owner_mailing_state": "AZ",
  "owner_mailing_zip": "85323",
  "owner_name": "SILVIAN LAURA M",
  "parcel_id": "0215104007",
  "situs_address": "706 S 114TH PL",
  "situs_city": "Avondale",
  "situs_state": "AZ",
  "situs_zip": "85323"
}
```

**Claude Opus 4.8 output:**

```json
{
  "parcel_id": null,
  "situs_address": "706 S 114TH LN",
  "situs_city": "Avondale",
  "situs_state": "AZ",
  "situs_zip": "85323",
  "owner_name": "SILIVAN LAURA M",
  "owner_mailing_address": null,
  "owner_mailing_city": null,
  "owner_mailing_state": null,
  "owner_mailing_zip": null
}
```

**Ground truth:**

```json
{
  "parcel_id": "10101020",
  "situs_address": "706 S 114TH LN   AVONDALE  85323",
  "situs_city": "AVONDALE",
  "situs_state": "AZ",
  "situs_zip": "85323",
  "owner_name": "SILLIVAN LAURA M",
  "owner_mailing_address": "706 S 114TH LN",
  "owner_mailing_city": "AVONDALE",
  "owner_mailing_state": "AZ",
  "owner_mailing_zip": "85323"
}
```

---

## Example (Maricopa County)

**Raw messy input:**

```
710 S 114TH LN AVONDALE 85323, Avondale, AZ, 85323 owner=JAVIER O CORDOVA BROWN REVOCABLE LIVING TRUST
```

**Fine-tuned model output:**

```json
{
  "owner_mailing_address": "710 E 194TH ST",
  "owner_mailing_city": "Avondale",
  "owner_mailing_state": "AZ",
  "owner_mailing_zip": "85323",
  "owner_name": "JAVIER O CORDOVA BROWN REVOCABLE LIVING TRUST",
  "parcel_id": "0216104007",
  "situs_address": "710 S 114TH LN",
  "situs_city": "Avondale",
  "situs_state": "AZ",
  "situs_zip": "85323"
}
```

**Claude Opus 4.8 output:**

```json
{
  "parcel_id": null,
  "situs_address": "710 S 114TH LN",
  "situs_city": "Avondale",
  "situs_state": "AZ",
  "situs_zip": "85323",
  "owner_name": "JAVIER O CORDOVA BROWN REVOCABLE LIVING TRUST",
  "owner_mailing_address": null,
  "owner_mailing_city": null,
  "owner_mailing_state": null,
  "owner_mailing_zip": null
}
```

**Ground truth:**

```json
{
  "parcel_id": "10101021",
  "situs_address": "710 S 114TH LN   AVONDALE  85323",
  "situs_city": "AVONDALE",
  "situs_state": "AZ",
  "situs_zip": "85323",
  "owner_name": "JAVIER O CORDOVA BROWN REVOCABLE LIVING TRUST",
  "owner_mailing_address": "9858 N LUNAR DR",
  "owner_mailing_city": "FLAGSTAFF",
  "owner_mailing_state": "AZ",
  "owner_mailing_zip": "86004"
}
```

---

## Example (Maricopa County)

**Raw messy input:**

```
714 S 114TH LANE AVONALE 85323, Avondale, 85323 owner=HULSEY ROBERT K/PAMELA  M
```

**Fine-tuned model output:**

```json
{
  "owner_mailing_address": "714 S 114TH ST",
  "owner_mailing_city": "Avondale",
  "owner_mailing_state": "IL",
  "owner_mailing_zip": "60511",
  "owner_name": "HULSEY, ROBERT K / PAMELA M",
  "parcel_id": "0219104011",
  "situs_address": "714 S 114TH LN",
  "situs_city": "Avondale",
  "situs_state": "IL",
  "situs_zip": "60511"
}
```

**Claude Opus 4.8 output:**

```json
{
  "parcel_id": null,
  "situs_address": "714 S 114TH LANE",
  "situs_city": "Avondale",
  "situs_state": "AZ",
  "situs_zip": "85323",
  "owner_name": "HULSEY ROBERT K/PAMELA M",
  "owner_mailing_address": null,
  "owner_mailing_city": null,
  "owner_mailing_state": null,
  "owner_mailing_zip": null
}
```

**Ground truth:**

```json
{
  "parcel_id": "10101022",
  "situs_address": "714 S 114TH LN   AVONDALE  85323",
  "situs_city": "AVONDALE",
  "situs_state": "AZ",
  "situs_zip": "85323",
  "owner_name": "HULSEY ROBERT K/PAMELA M",
  "owner_mailing_address": "714 S 114TH LN",
  "owner_mailing_city": "AVONDALE",
  "owner_mailing_state": "AZ",
  "owner_mailing_zip": "85323"
}
```

---

## Example (Maricopa County)

**Raw messy input:**

```
DAVIS ERIC A/SHANA I - 11439 W HADLEY STREET AVONDALE 85323, avondale, AZ, 85323
```

**Fine-tuned model output:**

```json
{
  "owner_mailing_address": "11439 W HADLEY ST",
  "owner_mailing_city": "AVONDALE",
  "owner_mailing_state": "AZ",
  "owner_mailing_zip": "85323",
  "owner_name": "DAVIS,ERIC & SHANA I",
  "parcel_id": "0216307011",
  "situs_address": "11439 W HADLEY ST",
  "situs_city": "AVONDALE",
  "situs_state": "AZ",
  "situs_zip": "85323"
}
```

**Claude Opus 4.8 output:**

```json
{
  "parcel_id": null,
  "situs_address": "11439 W HADLEY STREET",
  "situs_city": "AVONDALE",
  "situs_state": "AZ",
  "situs_zip": "85323",
  "owner_name": "DAVIS ERIC A/SHANA I",
  "owner_mailing_address": null,
  "owner_mailing_city": null,
  "owner_mailing_state": null,
  "owner_mailing_zip": null
}
```

**Ground truth:**

```json
{
  "parcel_id": "10101023",
  "situs_address": "11439 W HADLEY ST   AVONDALE  85323",
  "situs_city": "AVONDALE",
  "situs_state": "AZ",
  "situs_zip": "85323",
  "owner_name": "DAVIS ERIC A/SHANA I",
  "owner_mailing_address": "11439 W HADLEY ST",
  "owner_mailing_city": "AVONDALE",
  "owner_mailing_state": "AZ",
  "owner_mailing_zip": "85323"
}
```

> **Note:** the model reliably infers that mailing city/state/zip match the situs address, but sometimes hallucinates the wrong street number or street name for the full `owner_mailing_address` field itself (e.g. "LN" vs "ST" swaps, or an incorrect street name) — a known limitation reflected in the lower `owner_mailing_address` accuracy (11.7%) in the eval table above.
