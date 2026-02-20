# phone-agent — Complete Specification

A German-language AI phone service agent built on LiveKit + Azure that handles inbound calls
end-to-end: greets callers, verifies them against a customer database, takes a structured service
order, and logs everything to Azure Blob Storage.

This document is self-contained: a fresh implementation can be built from it alone.

---

## 1. Single-file architecture

Everything lives in **one Python file** (`agent.py`).
Entry point: `python agent.py start`

---

## 2. Dependencies (`requirements.txt`)

```
python-dotenv>=1.0.0
azure-storage-blob>=12.19.0
livekit-agents>=0.12.0
livekit-plugins-azure>=0.3.0
livekit-plugins-openai>=0.3.0
livekit-plugins-turn-detector>=0.3.0   # optional
azure-communication-email>=1.0.0        # optional
```

> **No fuzzy-matching library** (rapidfuzz, fuzzywuzzy, etc.) is used.
> All similarity algorithms are implemented in pure Python (see §8).

---

## 3. Environment variables

### Required

| Variable | Purpose |
|---|---|
| `AZURE_STORAGE_CONNECTION_STRING` | Azure Blob Storage (config, customers, call logs) |

All other credentials can be supplied either as env vars **or** via the blob config file (§5).

### Optional / overridable via blob config

| Variable | Default | Blob config key |
|---|---|---|
| `AZURE_OPENAI_ENDPOINT` | — | `azure_openai.endpoint` |
| `AZURE_OPENAI_API_KEY` | — | `azure_openai.api_key` |
| `AZURE_OPENAI_DEPLOYMENT` | — | `azure_openai.deployment` |
| `OPENAI_API_VERSION` | `"2024-10-01-preview"` | `azure_openai.api_version` |
| `LIVEKIT_URL` | — | `livekit.url` |
| `LIVEKIT_API_KEY` | — | `livekit.api_key` |
| `LIVEKIT_API_SECRET` | — | `livekit.api_secret` |
| `BLOB_CONTAINER` | `"assistant"` | — |
| `CONFIG_BLOB` | `"config/latest.json"` | — |
| `CUSTOMERS_BLOB` | `"data/customers.csv"` | — |
| `CALLS_PREFIX` | `"calls/"` | — |
| `AGENT_NAME` | `"phone-assistant"` | — |
| `LOG_LEVEL` | `"INFO"` | — |
| `DOTENV_PATH` / `DOTENV_FILE` | auto-search | — |
| `COMMUNICATION_CONNECTION_STRING_EMAIL` | — | — |
| `AZURE_COMMUNICATION_CONNECTION_STRING` | — | — |

### Credential bootstrap order

**Azure OpenAI**: check env vars first → fall back to `azure_openai.*` in blob config → raise
`RuntimeError` if still missing after both.

**LiveKit**: check env vars first → fall back to `livekit.*` in blob config → non-fatal if still
missing (LiveKit manages its own connection).

**ACS Email**: env var only; skipped silently if absent.

### Local dev `.env` search order

```python
# 1. explicit: $DOTENV_PATH or $DOTENV_FILE
# 2. next to script: .env.local, .env
# 3. CWD: .env.local, .env
# 4. find_dotenv(usecwd=True) — walks upward
# Never overrides already-set env vars (override=False)
```

---

## 4. Azure Blob layout

```
<container>/             # default: "assistant"
  config/
    latest.json          # agent configuration (§5)
  data/
    customers.csv        # customer database (§6)
  calls/
    <call_id>.json       # one log file per call (§13)
```

---

## 5. Blob config schema (`config/latest.json`)

Every field is optional — the agent falls back to the defaults shown.

```json
{
  "meta": {
    "saved_at": null,
    "saved_by": null
  },

  "livekit": {
    "url": "wss://...",
    "api_key": "...",
    "api_secret": "..."
  },

  "azure_openai": {
    "endpoint": "https://...",
    "api_key": "...",
    "deployment": "gpt-4o",
    "api_version": "2024-10-01-preview"
  },

  "agent": {
    "system_prompt": "Du bist ein deutscher Telefon-Serviceassistent.\nAblauf: Begrüßen -> Kundennummer + Name des Anrufers + (Firmenname oder Standort) abfragen -> per Tool verifizieren -> Serviceauftrag aufnehmen -> am Ende kurz zusammenfassen.\nRegeln: Keine Emojis. Kurze Sätze. Immer eine Frage auf einmal.",
    "welcome_message": "Willkommen beim Service. Bitte nennen Sie Ihre Kundennummer ODER Firma und Ort.",
    "runtime_rules_template": "Laufende Konfiguration (muss eingehalten werden):\n- Name des Anrufers abfragen: {{ask_caller_name}}\n- Max. Verifikationsversuche: {{max_attempts}}\nRegeln:\n- Sprich immer deutsch.\n- Kurze Sätze. Keine Emojis.\n- Verifiziere den Kunden IMMER über das Tool verify_customer.\n- Wenn locked=true zurückkommt: freundlich abbrechen oder an menschlichen Support verweisen.\n- Nach erfolgreicher Verifikation: Serviceauftrag strukturiert aufnehmen und submit_service_order verwenden.",
    "fuzzy_rules": "Fuzzy-Suche & Datenschutz (verbindlich):\n- Nutze zuerst search_customers zur internen Kandidatensuche.\n- Nenne niemals Kandidaten oder Adressen aus dem System.\n- Stelle genau eine Frage pro Turn: Standort oder Straßen-Prefix.\n- Verifiziere nur, wenn decision=allow und coverage_ok=true. Sonst eine Zusatzfrage oder Eskalation.\n- Gib niemals customer_id oder interne Systemdaten an den Anrufer weiter.\n- Biete Buchstabieren an, wenn unklar."
  },

  "speech": {
    "stt_languages": ["de-AT", "de-DE"],
    "tts_language": "de-DE",
    "tts_voice": "de-DE-KatjaNeural"
  },

  "turn": {
    "preset": "very_patient",
    "enabled": true,
    "min_endpointing_delay": 1.5,
    "max_endpointing_delay": 20.0
  },

  "customer_verification": {
    "ask_caller_name": true,
    "max_attempts": 3,
    "fuzzy": {
      "enabled": true,
      "max_candidates_internal": 5,
      "thresholds": {
        "allow": 0.86,
        "ask": 0.72,
        "block": 0.50
      },
      "weights": {
        "name": 0.60,
        "ort": 0.30,
        "addr": 0.10
      },
      "normalization": {
        "umlaute": "ae_oe_ue",
        "eszett": "ss",
        "punct": "drop",
        "spaces": "collapse",
        "legal_suffixes": ["gmbh", "mbh", "ag", "kg", "ug", "ohg", "kgaa", "co", "holding", "gruppe", "group"]
      },
      "phonetic": "koelner",
      "coverage": {
        "name": 0.80,
        "ort": 0.80,
        "addr": 0.60,
        "min_positives": 1
      },
      "conflict_penalty": {
        "name_threshold": 0.85,
        "ort_threshold": 0.60,
        "same_name_penalty": 0.05
      },
      "stt": {
        "phrase_hints": "from_csv",
        "phrase_hints_max": 500
      },
      "disambiguation": {
        "max_turns": 2,
        "order": ["ort", "strassen_prefix"]
      },
      "privacy": {
        "no_candidate_enumeration": true,
        "masked_confirmations_only": true
      }
    }
  },

  "email": {
    "enabled": false,
    "sender": "",
    "recipients": [],
    "subject_template": "Service Call {{callId}}"
  },

  "storage": {
    "customers_csv_blob": "data/customers.csv",
    "calls_prefix": "calls/"
  },

  "llm": {
    "temperature": 0.2
  }
}
```

---

## 6. Customer CSV schema

Stored at `data/customers.csv` (override with `storage.customers_csv_blob`).

**Exact columns** (fixed schema, no aliases):

| Column | Required | Description |
|---|---|---|
| `customer_id` | yes | Unique identifier, used for exact-match lookup |
| `firmenname` | yes | Company name, used for fuzzy search and verification |
| `standort` | yes | City / site name, used as fuzzy disambiguation signal |
| `adresse` | no | Street address, used as fuzzy disambiguation signal |
| `country` | no | Country, stored in call log only |

Example:

```csv
customer_id,firmenname,standort,adresse,country
1,Palfinger,Bergheim,Lamprechtshausen,Österreich
```

---

## 7. Data models (Python dataclasses)

Copy verbatim into the agent file:

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class VerifiedCustomer:
    customer_id: str
    firmenname: str
    standort: str
    adresse: str
    country: str
    caller_name: str


@dataclass
class ServiceOrder:
    order_id: str          # os.urandom(8).hex()
    problem: str
    priority: str          # default "normal"
    contact_phone: str
    preferred_time: str
    timestamp_utc: str     # datetime.now(timezone.utc).isoformat()


@dataclass
class SessionArtifacts:
    config: Dict[str, Any] = field(default_factory=dict)
    customers: List[Dict[str, str]] = field(default_factory=list)

    verified_customer: Optional[VerifiedCustomer] = None
    service_order: Optional[ServiceOrder] = None

    transcript: List[Dict[str, Any]] = field(default_factory=list)
    # entries: {"role": "agent"|"user", "text": str, "interrupted": bool}
    summary: Optional[str] = None

    caller_number: Optional[str] = None
    call_id: Optional[str] = None

    verification_attempts: int = 0

    # Fuzzy diagnostics (masked, no PII — written to call log)
    fz_decision: Optional[str] = None       # "allow" | "ask" | "block"
    fz_score: Optional[float] = None
    fz_coverage_ok: Optional[bool] = None
    fz_features: Dict[str, float] = field(default_factory=dict)
    fz_asked: List[str] = field(default_factory=list)
```

---

## 8. Fuzzy matching system (pure Python)

All code below is copied verbatim from the agent. No external library is needed.

### 8.1 Helpers and normalization

```python
import re
from typing import Set

_LEGAL_SUFFIX_STOPWORDS: Set[str] = {
    "gmbh", "mbh", "ag", "kg", "ug", "ohg", "kgaa", "co", "co.", "holding", "gruppe", "group"
}

def _collapse_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def _strip_punct(s: str) -> str:
    return re.sub(r"[^\w\s\-]", " ", s or "")

def _german_umlaut_norm(s: str, cfg: Dict[str, Any]) -> str:
    if not s:
        return s
    if (cfg or {}).get("umlaute", "ae_oe_ue") == "ae_oe_ue":
        s = (
            s.replace("Ä", "Ae").replace("Ö", "Oe").replace("Ü", "Ue")
             .replace("ä", "ae").replace("ö", "oe").replace("ü", "ue")
        )
    if (cfg or {}).get("eszett", "ss") == "ss":
        s = s.replace("ß", "ss")
    return s

def normalize_text(s: str, norm_cfg: Dict[str, Any]) -> str:
    s = s or ""
    s = _german_umlaut_norm(s, norm_cfg)
    if (norm_cfg or {}).get("punct", "drop") == "drop":
        s = _strip_punct(s)
    s = _collapse_spaces(s)
    s = s.casefold()
    # drop legal suffixes loaded from config (or module-level default)
    legal = set(norm_cfg.get("legal_suffixes", [])) or _LEGAL_SUFFIX_STOPWORDS
    tokens = [t for t in s.split(" ") if t and t not in legal]
    return " ".join(tokens)
```

### 8.2 Similarity functions

```python
def _ngrams(s: str, n: int = 3) -> Set[str]:
    s = _collapse_spaces(s)
    if len(s) < n:
        return {s} if s else set()
    return {s[i:i+n] for i in range(len(s)-n+1)}

def sim_trigram(a: str, b: str) -> float:
    A, B = _ngrams(a, 3), _ngrams(b, 3)
    if not A or not B:
        return 0.0
    inter = len(A & B)
    union = len(A | B)
    return inter / union if union else 0.0

def sim_levenshtein(a: str, b: str) -> float:
    a, b = a or "", b or ""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    la, lb = len(a), len(b)
    dp = list(range(lb + 1))
    for i in range(1, la + 1):
        prev = dp[0]
        dp[0] = i
        ca = a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ca == b[j - 1] else 1
            cur = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev, dp[j] = dp[j], cur
    return 1.0 - (dp[lb] / max(la, lb))

def sim_token_set(a: str, b: str) -> float:
    ta = set(t for t in _collapse_spaces(a).split(" ") if t)
    tb = set(t for t in _collapse_spaces(b).split(" ") if t)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0
```

### 8.3 Kölner Phonetik

Used as a **booster only**: if the phonetic codes of query and row value match, the field
similarity is raised to `max(computed_sim, 0.95)`. Applied to `name` and `ort`.

```python
def phonetic_koelner(s: str) -> str:
    s = (s or "").lower()
    s = _german_umlaut_norm(s, {"umlaute": "ae_oe_ue", "eszett": "ss"})
    s = re.sub(r"[^a-z]", "", s)
    if not s:
        return ""

    def code_char(ch: str, prev: str, nxt: str) -> str:
        if ch in "aeiouyj":  return "0"
        if ch == "h":        return ""
        if ch == "b":        return "1"
        if ch == "p":        return "1" if nxt != "h" else "3"
        if ch in "dt":       return "2"
        if ch in "fvw":      return "3"
        if ch in "gkq":      return "4"
        if ch == "x":        return "48"
        if ch == "c":        return "4"
        if ch in "szß":      return "8"
        if ch == "l":        return "5"
        if ch in "mn":       return "6"
        if ch == "r":        return "7"
        return ""

    out: List[str] = []
    for i, ch in enumerate(s):
        prev = s[i - 1] if i > 0 else ""
        nxt  = s[i + 1] if i + 1 < len(s) else ""
        out.append(code_char(ch, prev, nxt))
    dedup: List[str] = []
    for c in out:
        if not dedup or c != dedup[-1]:
            dedup.append(c)
    return "".join(dedup)
```

### 8.4 Feature scoring (CSV columns only)

Only the three columns that exist in the CSV are scored: `firmenname` → `name`,
`standort` → `ort`, `adresse` → `addr`.

```python
def _compute_feature_scores(
    name_q: str, ort_q: str, street_prefix_q: str,
    row: Dict[str, str],
    norm_cfg: Dict[str, Any],
) -> Dict[str, float]:
    name_r = normalize_text(row.get("firmenname", "") or "", norm_cfg)
    ort_r  = normalize_text(row.get("standort", "") or "", norm_cfg)
    addr_r = normalize_text(row.get("adresse", "") or "", norm_cfg)

    # name: trigram + levenshtein + token-set, phonetic boost
    name_char = max(sim_trigram(name_q, name_r), sim_levenshtein(name_q, name_r))
    name_sim  = max(name_char, sim_token_set(name_q, name_r))
    if name_q and name_r and phonetic_koelner(name_q) == phonetic_koelner(name_r):
        name_sim = max(name_sim, 0.95)

    # ort: trigram + levenshtein, phonetic boost
    ort_sim = max(sim_trigram(ort_q, ort_r), sim_levenshtein(ort_q, ort_r))
    if ort_q and ort_r and phonetic_koelner(ort_q) == phonetic_koelner(ort_r):
        ort_sim = max(ort_sim, 0.95)

    # addr: prefix match or trigram
    addr_sim = 0.0
    if street_prefix_q and addr_r:
        if addr_r.startswith(street_prefix_q):
            addr_sim = 0.8
        else:
            addr_sim = 0.6 * sim_trigram(street_prefix_q, addr_r)

    return {
        "name": float(name_sim),
        "ort":  float(ort_sim),
        "addr": float(addr_sim),
    }
```

### 8.5 Weighted score, coverage, conflict penalty

```python
def _weighted_score(features: Dict[str, float], weights: Dict[str, float]) -> float:
    score = 0.0
    total_w = 0.0
    for k, v in features.items():
        w = float(weights.get(k, 0.0))
        score   += v * w
        total_w += w
    return (score / total_w) if total_w > 0 else 0.0

def _coverage_ok(features: Dict[str, float], cov_cfg: Dict[str, Any]) -> bool:
    positives = 0
    if features.get("name", 0.0) >= float(cov_cfg.get("name", 0.80)): positives += 1
    if features.get("ort",  0.0) >= float(cov_cfg.get("ort",  0.80)): positives += 1
    if features.get("addr", 0.0) >= float(cov_cfg.get("addr", 0.60)): positives += 1
    return positives >= int(cov_cfg.get("min_positives", 1))

def _conflict_penalty(features: Dict[str, float], pen_cfg: Dict[str, Any]) -> float:
    penalty = 0.0
    name_th = float(pen_cfg.get("name_threshold", 0.85))
    ort_th  = float(pen_cfg.get("ort_threshold",  0.60))
    pen     = float(pen_cfg.get("same_name_penalty", 0.05))
    # High name match but ort very low → slight penalty (ambiguous same-name companies)
    if features.get("name", 0.0) >= name_th and features.get("ort", 0.0) < ort_th:
        penalty += pen
    return penalty
```

### 8.6 Disambiguation helper

```python
def _choose_ask_next(
    order: List[str],
    provided: Dict[str, str],
    asked: List[str],
    features: Dict[str, float],
) -> str:
    already = set(asked or [])
    for feat in order:
        if feat in already:
            continue
        if feat == "ort" and not provided.get("ort"):
            return "ort"
        if feat == "strassen_prefix" and not provided.get("street_prefix"):
            return "strassen_prefix"
    # fallback: weakest remaining feature
    weakest = min(features.items(), key=lambda kv: kv[1])[0] if features else ""
    return {"addr": "strassen_prefix"}.get(weakest, "")
```

### 8.7 Decision logic

```
if score >= thresholds.allow AND coverage_ok  →  decision = "allow"
elif score >= thresholds.ask                  →  decision = "ask"
else                                          →  decision = "block"
```

---

## 9. Tools (`@function_tool`)

### 9.1 `verify_customer`

Two lookup paths: **customer_id** (exact) or **firmenname** (exact case-insensitive).
No standort/address matching — the fuzzy pre-screen (`search_customers`) handles disambiguation.

```python
@function_tool
async def verify_customer(
    context: RunContext,
    customer_id: str = "",
    caller_name: str = "",
    firmenname: str = "",
) -> Dict[str, Any]:
    """
    Verifiziert Kunden gegen customers.csv.
    Pfad 1: customer_id (exakt).
    Pfad 2: firmenname (exakt, case-insensitive).
    """
    artifacts: SessionArtifacts = context.userdata
    artifacts.verification_attempts += 1

    max_attempts = int(_get_cfg(artifacts, "customer_verification", "max_attempts", default=3) or 3)
    if artifacts.verification_attempts > max_attempts:
        return {"ok": False, "locked": True,
                "reason": f"Maximale Verifikationsversuche erreicht ({max_attempts})."}

    customer_id_in   = (customer_id or "").strip()
    firmenname_in    = (firmenname  or "").strip()

    # Path 1: exact match on customer_id
    if customer_id_in:
        row = next(
            (r for r in artifacts.customers
             if any((v or "").strip() == customer_id_in for v in r.values())),
            None,
        )
        if not row:
            return {"ok": False, "reason": "Kundennummer nicht gefunden."}

    # Path 2: exact case-insensitive match on firmenname column
    elif firmenname_in:
        row = next(
            (r for r in artifacts.customers
             if (r.get("firmenname") or "").strip().lower() == firmenname_in.lower()),
            None,
        )
        if not row:
            return {"ok": False, "reason": "Firmenname nicht gefunden."}

    else:
        return {"ok": False, "reason": "Bitte nennen Sie Ihre Kundennummer oder Ihren Firmennamen."}

    ask_name = bool(_get_cfg(artifacts, "customer_verification", "ask_caller_name", default=True))
    caller_name = (caller_name or "").strip()
    if ask_name and not caller_name:
        return {"ok": False, "reason": "Name des Anrufers fehlt."}

    vc = VerifiedCustomer(
        customer_id=row.get("customer_id") or "",
        firmenname=row.get("firmenname")   or "",
        standort=row.get("standort")       or "",
        adresse=row.get("adresse")         or "",
        country=row.get("country")         or "",
        caller_name=caller_name            or "",
    )
    artifacts.verified_customer = vc
    return {"ok": True, "customer_id": vc.customer_id, "firmenname": vc.firmenname,
            "standort": vc.standort, "caller_name": vc.caller_name}
```

### 9.2 `search_customers`

Fuzzy pre-screen. **Never speak `best_candidate_customer_id` to the caller.**

```python
@function_tool
async def search_customers(
    context: RunContext,
    firmenname: str = "",
    standort: str = "",
    street_prefix: str = "",
) -> Dict[str, Any]:
    """
    Interne fuzzy Kandidatensuche OHNE Kandidatennennung nach außen.
    Sucht nur über CSV-Spalten: firmenname, standort, adresse.
    """
    artifacts: SessionArtifacts = context.userdata
    cv_cfg    = (artifacts.config or {}).get("customer_verification") or {}
    fuzzy_cfg = (cv_cfg.get("fuzzy") or {})

    thresholds = fuzzy_cfg.get("thresholds") or {"allow": 0.86, "ask": 0.72, "block": 0.50}
    weights    = fuzzy_cfg.get("weights")    or {"name": 0.60, "ort": 0.30, "addr": 0.10}
    norm_cfg   = fuzzy_cfg.get("normalization") or {"umlaute": "ae_oe_ue", "eszett": "ss", "punct": "drop"}
    cov_cfg    = fuzzy_cfg.get("coverage")   or {"name": 0.80, "ort": 0.80, "addr": 0.60, "min_positives": 1}
    pen_cfg    = fuzzy_cfg.get("conflict_penalty") or {}
    disamb     = fuzzy_cfg.get("disambiguation") or {"max_turns": 2, "order": ["ort", "strassen_prefix"]}
    max_k      = int(fuzzy_cfg.get("max_candidates_internal", 5) or 5)

    name_q          = normalize_text(firmenname,    norm_cfg)
    ort_q           = normalize_text(standort,      norm_cfg)
    street_prefix_q = normalize_text(street_prefix, norm_cfg)

    provided = {"ort": ort_q, "street_prefix": street_prefix_q}

    scored: List[Tuple[str, float, Dict[str, float]]] = []
    for row in artifacts.customers or []:
        cid = (row.get("customer_id") or "").strip()
        if not cid:
            continue
        feats      = _compute_feature_scores(name_q, ort_q, street_prefix_q, row, norm_cfg)
        score_base = _weighted_score(feats, weights)
        penalty    = _conflict_penalty(feats, pen_cfg)
        scored.append((cid, max(0.0, score_base - penalty), feats))

    scored.sort(key=lambda t: t[1], reverse=True)
    shortlist = scored[:max_k] if max_k > 0 else scored
    best_cid, best_score, best_feats = (shortlist[0] if shortlist else ("", 0.0, {}))

    cov_ok = _coverage_ok(best_feats, cov_cfg) if shortlist else False

    allow_th = float(thresholds.get("allow", 0.86))
    ask_th   = float(thresholds.get("ask",   0.72))

    if best_score >= allow_th and cov_ok:
        decision = "allow"
        ask_next = ""
    elif best_score >= ask_th:
        decision = "ask"
        ask_next = _choose_ask_next(
            list(disamb.get("order") or ["ort", "strassen_prefix"]),
            provided, artifacts.fz_asked, best_feats,
        )
    else:
        decision = "block"
        ask_next = _choose_ask_next(
            list(disamb.get("order") or []),
            provided, artifacts.fz_asked, best_feats,
        )

    artifacts.fz_decision    = decision
    artifacts.fz_score       = float(best_score)
    artifacts.fz_coverage_ok = bool(cov_ok)
    artifacts.fz_features    = {k: float(best_feats.get(k, 0.0)) for k in ("name", "ort", "addr")}

    newly = []
    if ort_q:           newly.append("ort")
    if street_prefix_q: newly.append("strassen_prefix")
    seen = set(artifacts.fz_asked)
    for x in newly:
        if x not in seen:
            artifacts.fz_asked.append(x)
            seen.add(x)

    return {
        "ok": True,
        "decision": decision,
        "score": float(best_score),
        "coverage_ok": bool(cov_ok),
        "best_candidate_customer_id": best_cid,   # INTERNAL ONLY; never speak this
        "ask_next": ask_next,
        "features": artifacts.fz_features,
        "privacy": {"enumerated": False},
    }
```

### 9.3 `submit_service_order`

```python
@function_tool
async def submit_service_order(
    context: RunContext,
    problem: str,
    priority: str = "normal",
    contact_phone: str = "",
    preferred_time: str = "",
) -> Dict[str, Any]:
    """Speichert einen Serviceauftrag im Session-Context."""
    artifacts: SessionArtifacts = context.userdata
    if not artifacts.verified_customer:
        return {"ok": False, "reason": "Kunde ist nicht verifiziert."}

    order = ServiceOrder(
        order_id=os.urandom(8).hex(),
        problem=(problem or "").strip(),
        priority=(priority or "normal").strip(),
        contact_phone=(contact_phone or "").strip(),
        preferred_time=(preferred_time or "").strip(),
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )
    artifacts.service_order = order
    return {"ok": True, "order_id": order.order_id}
```

---

## 10. System prompt construction

```python
# Template substitution (same {{key}} syntax as email subjects)
def _render_template(template: str, vars: Dict[str, str]) -> str:
    def repl(m):
        return vars.get((m.group(1) or "").strip(), m.group(0))
    return re.sub(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}", repl, template or "")

# At session start:
agent_cfg = cfg.get("agent", {})
cv_cfg    = cfg.get("customer_verification", {})

base_prompt = (
    agent_cfg.get("system_prompt")
    or agent_cfg.get("instructions")
    or _default_config()["agent"]["system_prompt"]
)

ask_name     = bool(cv_cfg.get("ask_caller_name", True))
max_attempts = int(cv_cfg.get("max_attempts", 3) or 3)

runtime_rules_tmpl = (
    agent_cfg.get("runtime_rules_template")
    or _default_config()["agent"]["runtime_rules_template"]
)
runtime_rules = _render_template(runtime_rules_tmpl, {
    "ask_caller_name": "ja" if ask_name else "nein",
    "max_attempts": str(max_attempts),
})

fuzzy_enabled = bool((cv_cfg.get("fuzzy") or {}).get("enabled", True))
fuzzy_rules   = ""
if fuzzy_enabled:
    fuzzy_rules = (
        agent_cfg.get("fuzzy_rules")
        or _default_config()["agent"]["fuzzy_rules"]
    )

instructions = "\n\n".join(filter(None, [base_prompt, runtime_rules, fuzzy_rules])).strip()
```

---

## 11. Phrase hints for STT

If `fuzzy.stt.phrase_hints == "from_csv"`, extract company names and city names from the customer
list and pass to `azure_speech.STT(hints=phrase_hints)`. Cap at `fuzzy.stt.phrase_hints_max`
(default 500). Fall back silently if the STT plugin version doesn't support `hints`.

```python
phrase_hints: List[str] = []
use_hints = (fuzzy_cfg.get("stt") or {}).get("phrase_hints") == "from_csv"
max_hints  = int((fuzzy_cfg.get("stt") or {}).get("phrase_hints_max", 500))
if use_hints and customers:
    seen: Set[str] = set()
    for r in customers:
        for k in ("firmenname", "standort"):
            v = (r.get(k) or "").strip()
            if v and v not in seen:
                seen.add(v)
                phrase_hints.append(v)
                if len(phrase_hints) >= max_hints:
                    break
        if len(phrase_hints) >= max_hints:
            break

try:
    stt = azure_speech.STT(language=stt_langs, hints=phrase_hints) if phrase_hints \
          else azure_speech.STT(language=stt_langs)
except TypeError:
    stt = azure_speech.STT(language=stt_langs)
```

---

## 12. Call flow

```
STARTUP (per call)
  1.  Connect to LiveKit room (audio-only via AutoSubscribe.AUDIO_ONLY)
  2.  Wait for participant → extract caller number
      (SIP attrs: "sip.phoneNumber", "caller.number", "sip.callerNumber")
  3.  Load blob config → run Azure OpenAI bootstrap if env vars missing
  4.  Load customers CSV (whitespace-stripped, UTF-8-BOM-safe)
  5.  Derive call_id: job.id / job.dispatch_id / room.name
  6.  Optionally init MultilingualModel turn detector
  7.  Init Azure OpenAI LLM (from env vars or blob config)
  8.  Init Azure Speech STT (multilingual + optional phrase hints)
  9.  Init Azure Speech TTS (configured voice/language)
 10.  Build system prompt from blob config templates (§10)
 11.  Create Agent with tools:
        [verify_customer, submit_service_order, search_customers]  if fuzzy enabled
        [verify_customer, submit_service_order]                    otherwise
 12.  Register shutdown callback (persist_and_notify)
 13.  Start AgentSession
 14.  Say welcome_message (allow_interruptions=False)

CONVERSATION
 15.  Caller provides customer number or company name
 16.  Agent calls search_customers (fuzzy pre-screen) → decision: allow / ask / block
 17.  If "ask": ask one disambiguation question (standort or street prefix)
 18.  Repeat until "allow" or attempt limit reached
 19.  Agent calls verify_customer (customer_id OR firmenname)
 20.  Agent collects service order: problem, priority, callback, preferred time
 21.  Agent calls submit_service_order
 22.  Agent summarises and closes

SHUTDOWN (async callback)
 23.  Build call log JSON (§13)
 24.  Write to blob: calls/{call_id}.json
 25.  If email.enabled: send ACS email with log + config as attachments
```

---

## 13. Call log schema (`calls/{call_id}.json`)

```json
{
  "call_id": "room_abc123",
  "room": "room_abc123",
  "timestamp": "2024-01-01T12:00:00+00:00",
  "caller": "+43...",
  "config_meta": {"saved_at": "...", "saved_by": "..."},
  "verified_customer": {
    "customer_id": "1",
    "firmenname": "Palfinger",
    "standort": "Bergheim",
    "adresse": "Lamprechtshausen",
    "country": "Österreich",
    "caller_name": "Max Mustermann"
  },
  "service_order": {
    "order_id": "a1b2c3d4e5f6g7h8",
    "problem": "Kran reagiert nicht",
    "priority": "hoch",
    "contact_phone": "+43...",
    "preferred_time": "Montag 9-12 Uhr",
    "timestamp_utc": "2024-01-01T12:05:00+00:00"
  },
  "transcript": [
    {"role": "agent", "text": "Willkommen beim Service..."},
    {"role": "user",  "text": "Guten Tag, ich bin Max Mustermann von Palfinger..."}
  ],
  "summary": null,
  "fuzzy_diagnostics": {
    "enabled": true,
    "decision": "allow",
    "score": 0.91,
    "coverage_ok": true,
    "features": {"name": 0.95, "ort": 0.82, "addr": 0.0},
    "asked_features": ["ort"],
    "privacy": {"no_candidate_enumeration": true}
  }
}
```

---

## 14. Email notification (optional)

Triggered at call end if `email.enabled: true`.

- **Sender**: `email.sender`
- **Recipients**: `email.recipients` (list of strings)
- **Subject**: `email.subject_template` rendered with `{{callId}}`, `{{room}}`, `{{customerId}}`
- **Body**: plain-text summary (call ID, caller number, customer, order excerpt)
- **Attachments**:
  - `call_{call_id}.json` — full call log
  - `config_{call_id}.json` — config snapshot (for audit)

ACS connection string: `COMMUNICATION_CONNECTION_STRING_EMAIL` or
`AZURE_COMMUNICATION_CONNECTION_STRING`. Skipped silently if missing or
`azure.communication.email` is not installed.

---

## 15. Key design principles

| Principle | Implementation |
|---|---|
| **Fallback-first** | Every config field has a default; missing blob config never crashes startup |
| **Config-driven** | All behavior (prompt text, speech, fuzzy params) driven by blob JSON |
| **Credential flexibility** | Azure OpenAI + LiveKit: env vars OR blob config |
| **Privacy-first** | Fuzzy results never enumerate candidates; only masked confirmations |
| **Zero extra ML deps** | Pure Python fuzzy; no rapidfuzz, fuzzywuzzy, etc. |
| **Audit trail** | Every call → structured JSON log with masked fuzzy diagnostics (no PII) |
| **Version resilience** | All LiveKit API imports wrapped in try/except |
| **STT robustness** | Kölner Phonetik + phrase hints from CSV handle transcription noise |

---

## 16. What was improved vs. the simpler baseline (`agent.py`)

| Aspect | Baseline | Improved |
|---|---|---|
| Customer matching | Raw CSV in LLM context window | Pure-Python fuzzy via `search_customers` tool |
| Privacy | LLM sees all customer data | Candidate list hidden; masked confirmations only |
| Scale | Breaks with large CSVs | O(n) Python scan, no context pressure |
| STT accuracy | Hard failures on transcription errors | Trigram + Levenshtein + Kölner Phonetik |
| Prompt text | Hardcoded German strings | Configurable templates in blob config |
| Credentials | Azure OpenAI: env vars only | Env vars OR blob config |
| Fuzzy thresholds | Magic numbers in source | All in `fuzzy.*` config |
| Coverage / penalty | Hardcoded constants | `fuzzy.coverage.*` and `fuzzy.conflict_penalty.*` |
| Phrase hints limit | Hardcoded 500 | `fuzzy.stt.phrase_hints_max` |
| Legal suffixes | Hardcoded module set | `fuzzy.normalization.legal_suffixes` |
| Disambiguation | Not present | Configurable order, up to `max_turns` questions |
