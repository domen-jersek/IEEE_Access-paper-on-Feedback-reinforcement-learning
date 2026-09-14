# Retrieval Landscape & Data Analysis
## Baseline Retrieval Quality and Dataset Properties for Feedback-Driven RAG

*Generated: 2026-09-13, prior to feedback DB completion. All analyses use the FAISS index (all-MiniLM-L6-v2, 384-dim) and the canonicalized dataset (1,595 tickets). No API calls.*

---

## 1. Dataset Overview

### 1.1 Raw ticket statistics (from `tickets_large_first_reply_label.csv`)

| Property | Value |
|----------|-------|
| Total tickets (after dedup) | 1,595 |
| Unique teams | 37 |
| Unique raw labels (`label_auto`) | 118 |
| Coarse intent classes | 17 |
| Service request / Incident | 1,453 / 143 |
| Tickets with no description | 1 |
| Tickets with description < 50 chars | 124 |

### 1.2 Reply characteristics

| Property | Value |
|----------|-------|
| Mean reply length | 1,352 chars (median: 800) |
| Std reply length | 1,508 chars |
| Form-redirect replies ("below you will find the additional form...") | **1,187 (74.4%)** |
| Auto-created replies ("This additional ticket is automatically created...") | 254 (15.9%) |
| English-translation-request replies | 7 (0.4%) |

### 1.3 Team distribution

| Tier | Teams | Tickets covered |
|------|-------|-----------------|
| Large (>50 tickets) | 9 | 1,274 (79.9%) |
| Medium (20-50) | 8 | 200 (12.5%) |
| Small (<20) | 20 | 122 (7.6%) |

Top 5 teams: Group (309), File & Print (198), Robot Process Automation (148), Account Management (145), Development Platform (138).

### 1.4 Class distribution (`label_auto`)

| Tier | Labels | Tickets covered |
|------|--------|-----------------|
| Large (>100) | 5 | 781 (49.0%) |
| Medium (50-100) | 4 | 302 (18.9%) |
| Small (10-50) | 14 | 290 (18.2%) |
| Tiny (<10) | 95 | 223 (14.0%) |

---

## 2. Baseline Retrieval Quality (FAISS-only, no feedback)

### 2.1 Overall distribution — 319 dev queries, seed 42

| Statistic | Value |
|-----------|-------|
| Mean top-1 FAISS | 0.808 |
| Median top-1 FAISS | 0.845 |
| Std | 0.168 |
| Min | 0.345 |
| Max | 1.000 |

### 2.2 Quintile breakdown

| Quintile | Range | n | Mean top-1 FAISS |
|----------|-------|---|-------------------|
| Q1 (weakest) | 0.345–0.632 | 64 | 0.539 |
| Q2 | 0.632–0.792 | 64 | 0.721 |
| Q3 | 0.792–0.898 | 63 | 0.846 |
| Q4 | 0.898–0.979 | 64 | 0.946 |
| Q5 (strongest) | 0.979–1.000 | 64 | 0.989 |

### 2.3 Difficulty buckets

| Bucket | Range | n | % | Interpretation |
|--------|-------|---|---|----------------|
| **Lost causes** | 0.00–0.55 | 31 | 10% | Even max lift (+0.20) brings best candidate to ~0.75 — still poor. Feedback unlikely to rescue. |
| **Feedbacks sweet spot** | 0.55–0.70 | 53 | 17% | Retrieval is mediocre. Lift can meaningfully re-rank. |
| **Feedbacks sweet spot** | 0.70–0.85 | 79 | 25% | Retrieval is decent. Lift can meaningfully re-rank. |
| **Risky** | 0.85–0.95 | 60 | 19% | Good baseline. Feedback may displace correct candidates. |
| **Perfect** | 0.95–1.00 | 95 | **30%** | Near-identical templates in KB. Feedback can only hurt. |

**Key finding: 42% of tickets are in the feedback sweet spot (0.55–0.85). 30% are protected (0.95+). 10% are beyond rescue (<0.55).**

### 2.4 Retrieval margin (top-1 minus top-2 FAISS) and spread (max-min) — 30-sample analysis

The retrieval landscape is **bimodal**:

- **Flat retrieval** (onboarding/offboarding form templates): margins ~0.001–0.004, spreads ~0.002–0.006. The top-5 are nearly identical template variants. Feedback has essentially no room to re-rank meaningfully — all candidates contain the same procedure.
- **Peaked retrieval** (VPN, hardware, software support): margins ~0.02–0.15, spreads ~0.05–0.22. The top-1 is clearly better than positions 2-5. Feedback could disrupt this if it promotes a wrong-but-popular candidate.

### 2.5 Per-team retrieval quality on dev set (n ≥ 5 per team)

| Team | n | Mean top-1 FAISS | Std |
|------|---|-------------------|-----|
| Network On-Prem (LAN,WLAN,WAN) | 5 | **0.590** | 0.036 |
| Network Cloud (Azure, Remote Access) | 5 | **0.623** | 0.074 |
| Salesforce | 16 | **0.675** | 0.153 |
| Service Desk | 6 | **0.713** | 0.164 |
| Group | 57 | 0.762 | 0.173 |
| Development Platform | 37 | 0.776 | 0.163 |
| Information Security Office | 8 | 0.807 | 0.120 |
| SAP & Synertrade | 5 | 0.839 | 0.136 |
| File & Print | 42 | 0.843 | 0.189 |
| Robot Process Automation | 29 | 0.846 | 0.115 |
| Network Access | 15 | 0.857 | 0.159 |
| Backend Application Srv. & Project Support | 16 | 0.892 | 0.146 |
| Account Management | 33 | 0.915 | 0.123 |
| IT Office Access Italy | 17 | **0.925** | 0.055 |

**Gap between worst and best: 0.335 (0.590 vs 0.925).** The 4 lowest teams (below 0.72) are the strongest candidates for feedback benefit. The 2 highest teams (above 0.91) are likely harmed by any feedback intervention.

### 2.6 Per-class retrieval quality on dev set

| Intent class | n | Mean top-1 FAISS | Std |
|-------------|---|-------------------|-----|
| hardware_support | 10 | **0.535** | 0.187 |
| project_tools | 5 | **0.602** | 0.176 |
| enterprise_systems | 5 | **0.622** | 0.173 |
| absence | 7 | **0.679** | 0.071 |
| email_support | 7 | **0.712** | 0.152 |
| other | 56 | 0.713 | 0.173 |
| software_support | 14 | 0.746 | 0.151 |
| vpn_access | 23 | 0.754 | 0.183 |
| password_reset | 6 | 0.802 | 0.052 |
| software_license | 48 | 0.816 | 0.137 |
| admin_rights | 51 | 0.821 | 0.121 |
| offboarding | 40 | **0.947** | 0.057 |
| onboarding | 40 | **0.973** | 0.024 |

The 5 lowest classes (0.53–0.71) would most benefit from feedback. Onboarding and offboarding (0.95–0.97) have near-perfect retrieval — the templates dominate.

### 2.7 Sample retrieval examples

**Hard query** (R-100, top-1=0.442): "Purchase IT ITA - Amazon - 100x TAG NFC for office access." Team: Admin - Local IT purchase. The top-5 candidates are all purchase-related but for different items (iPad, cables, digital signature). Semantic similarity is high (same department, same template structure), but the specific procedure for NFC tags differs.

**Easy query** (R-1002, top-1=0.984): "Gft Italia_new_entry" — onboarding form. The top-5 are all other employee onboarding entries with identical auto-created replies. Trivially correct retrieval.

**Moderate query** (R-1007, top-1=0.790): VPN access request. The top candidate has FAISS 0.79 with a margin of 0.10 to #2 (0.69). Strong top-1 signal. Feedback that disrupts this would hurt.

---

## 3. Data properties relevant to the paper

### 3.1 The form-redirect problem

74% of reference replies begin with "Below you will find the additional form information..." These are structured templates that map a request type to a specific form. The retrieval task for these is: *find another ticket of the same type.*

This means:
- **The baseline retrieval quality ceiling is high** — many queries have perfect matches.
- **Feedback's potential to help is limited to the 26% non-form tickets** — unless feedback can also distinguish sub-types within a form category.
- **Any gate must recognize overdetermined queries** — form-redirect tickets with top-1 FAISS > 0.90 should never receive feedback.

### 3.2 Near-duplicate structure

The KB contains 50 near-duplicate groups (147 tickets affected). Many share identical first replies. This creates retrieval "clusters" where the top-5 candidates are functionally equivalent. Feedback that promotes a different candidate from the same cluster changes nothing. Feedback that promotes a candidate from a different cluster changes the procedure.

### 3.3 Class imbalance and feedback sparsity

- 95 of 118 `label_auto` values have fewer than 10 tickets.
- After mapping to 17 intent classes, the smallest classes have 1-5 tickets in dev.
- The M4 (team∩class intersection) routing will receive zero feedback signal for 33%+ of queries because the (team, class) combo was never observed in train. This is a structural limitation, not a scope-quality problem.

---

## 4. Planned Experiment Matrix

### Phase A: Core Methods on Dev Set (seed 42, 319 queries)

| # | Method | Protocol | Agg mode | Gate | Purpose |
|---|--------|----------|----------|------|---------|
| A1 | baseline | — | — | none | Reference point, generation-cost baseline |
| A2 | M1 global | conditioned | continuous | none | Primary method |
| A3 | M1 global | blind | continuous | none | Blind vs conditioned comparison |
| A4 | M1 global | conditioned | binary | none | Continuous vs binary ablation |
| A5 | M2 team | conditioned | continuous | none | Team-scoped feedback |
| A6 | M3 class | conditioned | continuous | none | Class-scoped feedback |
| A7 | M4 intersection | conditioned | continuous | none | Team∩class — strictest scope |

### Phase B: Gates and Ablations on Dev

| # | Method | Protocol | Agg mode | Gate | Hypothesis |
|---|--------|----------|----------|------|------------|
| B1 | M1 global | conditioned | continuous | static 0.70 | Low ceiling: gate feedback when baseline is already strong |
| B2 | M1 global | conditioned | continuous | static 0.85 | Medium ceiling: feedback only for ≤Q3 queries |
| B3 | M1 global | conditioned | continuous | static 0.95 | High ceiling: feedback only for ≤Q4 queries |
| B4 | M1 global | conditioned | continuous | learned | Multivariate gate combining retrieval confidence, margin, evidence density, team/class priors |

### Phase C: Directed Hypotheses on Dev

| # | Method | What's tested | Requires |
|---|--------|---------------|----------|
| C1 | M1 global + form-exclusion | Skip feedback when reference reply is a form-redirect (~5 lines of code in protocol.py) | Code change |
| C2 | M1 global + evidence min-n=3 | Only apply lift when candidate has ≥3 observations from train queries | Lift formula tweak |
| C3 | M1 global + class-aware weights | Weight pos/neg by class similarity (cosine of query intent vs voter intent) | Routing tweak |

### Phase D: Cross-Model Comparison (optional, if Phase A is promising)

| # | Generator | Method | Protocol | Split |
|---|-----------|--------|----------|-------|
| D1 | gpt-4o-mini | M1 global | conditioned | dev |
| D2 | gpt-5.6-luna | M1 global | conditioned | dev |

### Phase E: Final Eval (run once at end, locked configs)

| # | Method | Protocol | Agg | Gate | Split | Seeds | Regime |
|---|--------|----------|-----|------|-------|-------|--------|
| E1 | baseline | — | — | — | eval | 42,123,456,789,1024 | random |
| E2 | best from Phase A-B | best | best | best | eval | 42,123,456,789,1024 | random |
| E3 | best from Phase A-B | best | best | best | eval | 42 | disjoint |

---

## 5. Data-level Claims Supporting the Paper

| # | Claim | Supporting evidence | Section |
|---|-------|---------------------|---------|
| C1 | Retrieval quality is bimodal — 30% near-perfect, 10% very poor, 60% in a usable middle | Sec 2.3 difficulty buckets | Sec 2 |
| C2 | 74% of reference replies are form-redirects, creating a natural ceiling on what feedback can improve | Sec 1.2, 3.1 | Sec 1, 3 |
| C3 | Per-team retrieval quality varies by 2× (0.59 → 0.93 top-1 FAISS) | Sec 2.5 | Sec 2 |
| C4 | Per-class retrieval quality similarly ranges from 0.54 (hardware) to 0.97 (onboarding) | Sec 2.6 | Sec 2 |
| C5 | The retrieval margin distribution is bimodal — flat for templates, peaked for unique queries | Sec 2.4 | Sec 2 |
| C6 | M4 intersection routing will have zero signal for 33%+ of queries due to data sparsity | Sec 3.3 | Sec 3 |
| C7 | Near-duplicate tickets create retrieval clusters where top-5 candidates are functionally identical | Sec 3.2 | Sec 3 |
| C8 | The "sweet spot" for feedback (top-1 FAISS 0.55–0.85) covers ~42% of queries | Sec 2.3 | Sec 2 |

---

## 6. Reproducibility

All numbers above can be reproduced by running:

```bash
python experiments/02_build_index.py
```

followed by querying the FAISS index for the dev split (seed 42). The CSV analysis uses `tickets_large_first_reply_label.csv` directly. No API keys required for any of the above.