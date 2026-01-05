"""### Language Detection"""

# pip -q install openai pandas tqdm scikit-learn requests

import pandas as pd
import os

os.environ["OPENROUTER_API_KEY"] = "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"

csv_path = f'datasets/language_detection_data.csv'
df = pd.read_csv(csv_path)

required = {"text","language","dialect","accent"}
missing = required - set(df.columns)
assert not missing, f"Missing columns: {missing}"

df.head()

# Ensure string columns (only for text input; NOT for label extraction)
df["text"] = df["text"].astype(str)

# =========================
# Your fixed label taxonomy
# =========================

EXTRA_LANGUAGES = {
    "Avestan",
    # Eastern Iranian - Northeastern
    "Ossetic", "Yagnobi", "Yassic",

    # Eastern Iranian - Southeastern
    "Pamir", "Pashto",

    # Western Iranian - Northwestern
    "Balochi", "Caspian", "Central Iran", "Kurdish", "Ormuri-Parachi",
    "Semnani", "Talysh", "Zaza-Gorani", "Unclassified",

    # Western Iranian - Southwestern
    "Fars", "Luri", "Persian", "Tat",
}

EXTRA_DIALECTS_BY_LANGUAGE = {
    "Avestan": {"General"},
    "Ossetic": {"General"},
    "Yagnobi": {"General"},
    "Yassic": {"General"},

    "Pamir": {"Ishkashimi (Afghanistan)", "Munji (Afghanistan)", "Sanglechi (Afghanistan)", "Wakhi (Afghanistan)", "Yidgha (Pakistan)"},
    "Pashto": {"Pashto, Central (Pakistan)", "Pashto, Northern (Pakistan)", "Pashto, Southern (Afghanistan)", "Waneci (Pakistan)"},

    "Balochi": {
        "Balochi, Eastern (Pakistan)",
        "Balochi, Southern (Pakistan)",
        "Balochi, Western (Pakistan)",
        "Bashkardi (Iran)",
        "Koroshi (Iran)",
    },

    "Caspian": {"Gilaki (Iran)", "Mazandarani (Iran)", "Shahmirzadi (Iran)"},

    "Central Iran": {
        "Ashtiani (Iran)", "Dari, Zoroastrian (Iran)", "Gazi (Iran)", "Khunsari (Iran)",
        "Natanzi (Iran)", "Nayini (Iran)", "Parsi-Dari (Iran)", "Sivandi (Iran)",
        "Soi (Iran)", "Vafsi (Iran)",
    },

    "Kurdish": {
        "Kurdish, Central (Iraq)",
        "Kurdish, Northern (Turkey)",
        "Kurdish, Southern (Iran)",
        "Laki (Iran)",
    },

    "Ormuri-Parachi": {"Ormuri (Pakistan)", "Parachi (Afghanistan)"},

    "Semnani": {"Lasgerdi (Iran)", "Sangisari (Iran)", "Semnani (Iran)", "Sorkhei (Iran)"},

    "Talysh": {
        "Alviri-Vidari (Iran)", "Eshtehardi (Iran)", "Gozarkhani (Iran)", "Harzani (Iran)",
        "Kabatei (Iran)", "Kajali (Iran)", "Karingani (Iran)", "Kho'ini (Iran)",
        "Koresh-e Rostam (Iran)", "Maraghei (Iran)", "Razajerdi (Iran)", "Rudbari (Iran)",
        "Shahrudi (Iran)", "Takestani (Iran)", "Talysh (Azerbaijan)", "Taromi, Upper (Iran)",
    },

    "Zaza-Gorani": {
        "Bajelani (Iraq)", "Gurani (Iran)", "Kakayi (Iraq)", "Shabak (Iraq)",
        "Zazaki, Northern (Turkey)", "Zazaki, Southern (Turkey)",
    },

    "Unclassified": {"Dezfuli (Iran)"},

    "Fars": {"Fars, Southwestern (Iran)", "Lari (Iran)"},

    "Luri": {"Bakhtiari (Iran)", "Kumzari (Oman)", "Luri, Northern (Iran)", "Luri, Southern (Iran)"},

    "Persian": {
        "Aimaq (Afghanistan)", "Bukharic (Uzbekistan)", "Dari (Afghanistan)", "Dehwari (Pakistan)",
        "Hazaragi (Afghanistan)", "Judeo-Persian (Iran)", "Pahlavani (Afghanistan)",
        "Persian, Iranian (Iran)", "Tajik (Tajikistan)",
    },

    "Tat": {"Judeo-Tat (Russian Federation)", "Tat, Muslim (Azerbaijan)"},
}

EXTRA_ACCENTS_BY_LANGUAGE_DIALECT = {
    ("Persian", "Persian, Iranian (Iran)"): {"Isfahani", "Shirazi", "Mashhadi", "Yazdi", "Khorasani", "General"},
    ("Persian", "Dari (Afghanistan)"): {"Kaboli", "General"},
    ("Persian", "Hazaragi (Afghanistan)"): {"Hazaragi", "General"},
    ("Kurdish", "Kurdish, Southern (Iran)"): {"Kurdish, Southern", "General"},
    ("Caspian", "Mazandarani (Iran)"): {"Tonekaboni", "General"},
    ("Semnani", "Semnani (Iran)"): {"Semnani", "General"},
    ("Central Iran", "Dari, Zoroastrian (Iran)"): {"Yazdi", "General"},
    ("Unclassified", "Dezfuli (Iran)"): {"Dezfuli", "General"},
    ("Pashto", "Pashto, Southern (Afghanistan)"): {"Isfahani", "Shirazi", "Mashhadi", "Yazdi", "Khorasani", "General"},
    ("Pashto", "Pashto, Central (Pakistan)"): {"Isfahani", "Shirazi", "Mashhadi", "Yazdi", "Khorasani", "General"},
    ("Luri", "Bakhtiari (Iran)"): {"Chaharmahali", "General"},
}

# =========================
# Optional "Unknown" safety
# =========================
ADD_UNKNOWN = True
UNKNOWN = "Unknown"

# Final LANGUAGES list (NO dataset merge)
LANGUAGES = sorted(EXTRA_LANGUAGES | ({UNKNOWN} if ADD_UNKNOWN else set()))

# Final DIALECTS_BY_LANGUAGE (NO dataset merge)
DIALECTS_BY_LANGUAGE = {}
for lang in LANGUAGES:
    if ADD_UNKNOWN and lang == UNKNOWN:
        DIALECTS_BY_LANGUAGE[UNKNOWN] = [UNKNOWN]
        continue

    dialects = set(EXTRA_DIALECTS_BY_LANGUAGE.get(lang, set()))
    if ADD_UNKNOWN:
        dialects.add(UNKNOWN)

    DIALECTS_BY_LANGUAGE[lang] = sorted(dialects)

# Final ACCENTS_BY_LANGUAGE_DIALECT (NO dataset merge)
ACCENTS_BY_LANGUAGE_DIALECT = {}
for (lang, dia), accs in EXTRA_ACCENTS_BY_LANGUAGE_DIALECT.items():
    s = set(accs)
    if ADD_UNKNOWN:
        s.add(UNKNOWN)
    ACCENTS_BY_LANGUAGE_DIALECT[(lang, dia)] = s

# If you want: default accents for any (lang,dialect) not listed
def get_allowed_accents(lang: str, dia: str):
    return sorted(ACCENTS_BY_LANGUAGE_DIALECT.get((lang, dia), {UNKNOWN} if ADD_UNKNOWN else set()))

print("✅ Label spaces ready (no CSV extraction)")
print("Languages:", len(LANGUAGES))
print("Example languages:", LANGUAGES[:5])

import os
import re, json, time
from typing import Any, Dict, List, Optional, Tuple
from openai import OpenAI, BadRequestError
from tqdm import tqdm

# =========================
# Prompt debugging controls
# =========================
DEBUG_PRINT_PROMPTS = True        # set False to silence
MAX_CHARS_PER_MSG = 2500          # truncate printed messages (printing only)
MAX_ITEMS_IN_LIST_PRINT = 60      # show at most N labels from allowed list (printing only)

def _truncate(s: str, n: int) -> str:
    s = s or ""
    return s if len(s) <= n else s[:n] + f"\n... [TRUNCATED {len(s)-n} chars]"

def _pretty_allowed_list(labels: List[str]) -> str:
    """Pretty-print allowed list but keep it short for printing only."""
    if len(labels) <= MAX_ITEMS_IN_LIST_PRINT:
        return json.dumps(labels, ensure_ascii=False)
    head = labels[:MAX_ITEMS_IN_LIST_PRINT]
    return json.dumps(head, ensure_ascii=False) + f" ... (+{len(labels)-len(head)} more)"

def print_messages(messages: List[Dict[str, str]], title: str = ""):
    """Print role+content for the exact messages sent to the API."""
    if not DEBUG_PRINT_PROMPTS:
        return
    print("\n" + "="*90)
    if title:
        print(title)
        print("-"*90)
    for m in messages:
        role = m.get("role", "").upper()
        content = m.get("content", "")
        print(f"[{role}]")
        print(_truncate(content, MAX_CHARS_PER_MSG))
        print()
    print("="*90 + "\n")


# =========================
# Core prompt templates (UPDATED: less "jailbreak-looking")
# =========================
SYSTEM_PROMPT = (
    "You are a classifier. "
    "Given a text and an allowed label list, choose exactly one label from the list. "
    "Return a JSON object."
)

def _json_only_instructions(label_key: str) -> str:
    return (
        f'Output format (JSON): {{"{label_key}": "<LABEL>"}}\n'
        "Constraint:\n"
        f'- The value for "{label_key}" must be exactly one item from the allowed list.\n'
    )

def model_supports_system_role(model_id: str) -> bool:
    return ("gemma" not in model_id.lower())

def make_messages(model_id: str, user_content: str) -> List[Dict[str, str]]:
    # Gemma rejects system/developer instruction -> merge into user
    if model_supports_system_role(model_id):
        return [{"role":"system","content":SYSTEM_PROMPT},
                {"role":"user","content":user_content}]
    return [{"role":"user","content":SYSTEM_PROMPT + "\n\n" + user_content}]

def make_language_messages(model_id: str, text: str, allowed_languages: List[str], debug: bool=False) -> List[Dict[str, str]]:
    allowed_full = json.dumps(allowed_languages, ensure_ascii=False)
    allowed_print = _pretty_allowed_list(allowed_languages)

    user = (
        "Task: Identify the language of the given text.\n"
        + _json_only_instructions("language")
        + f"Allowed languages (choose ONE): {allowed_full}\n"
        "Text:\n"
        f"{text}"
    )

    if debug and DEBUG_PRINT_PROMPTS:
        user_preview = (
            "Task: Identify the language of the given text.\n"
            + _json_only_instructions("language")
            + f"Allowed languages (choose ONE): {allowed_print}\n"
            "Text:\n"
            + _truncate(text, 800)
        )
        print_messages(make_messages(model_id, user_preview), title="PHASE 1 — LANGUAGE")

    return make_messages(model_id, user)

def make_dialect_messages(model_id: str, text: str, language: str, allowed_dialects: List[str], debug: bool=False) -> List[Dict[str, str]]:
    allowed_full = json.dumps(allowed_dialects, ensure_ascii=False)
    allowed_print = _pretty_allowed_list(allowed_dialects)

    user = (
        "Task: Identify the dialect of the text (language is given).\n"
        + _json_only_instructions("dialect")
        + f"Language: {language}\n"
        + f"Allowed dialects (choose ONE): {allowed_full}\n"
        "Text:\n"
        f"{text}"
    )

    if debug and DEBUG_PRINT_PROMPTS:
        user_preview = (
            "Task: Identify the dialect of the text (language is given).\n"
            + _json_only_instructions("dialect")
            + f"Language: {language}\n"
            + f"Allowed dialects (choose ONE): {allowed_print}\n"
            "Text:\n"
            + _truncate(text, 800)
        )
        print_messages(make_messages(model_id, user_preview), title="PHASE 2 — DIALECT")

    return make_messages(model_id, user)

def make_accent_messages(model_id: str, text: str, language: str, dialect: str, allowed_accents: List[str], debug: bool=False) -> List[Dict[str, str]]:
    allowed_full = json.dumps(allowed_accents, ensure_ascii=False)
    allowed_print = _pretty_allowed_list(allowed_accents)

    user = (
        "Task: Identify the accent of the text (language and dialect are given).\n"
        + _json_only_instructions("accent")
        + f"Language: {language}\n"
        + f"Dialect: {dialect}\n"
        + f"Allowed accents (choose ONE): {allowed_full}\n"
        "Text:\n"
        f"{text}"
    )

    if debug and DEBUG_PRINT_PROMPTS:
        user_preview = (
            "Task: Identify the accent of the text (language and dialect are given).\n"
            + _json_only_instructions("accent")
            + f"Language: {language}\n"
            + f"Dialect: {dialect}\n"
            + f"Allowed accents (choose ONE): {allowed_print}\n"
            "Text:\n"
            + _truncate(text, 800)
        )
        print_messages(make_messages(model_id, user_preview), title="PHASE 3 — ACCENT")

    return make_messages(model_id, user)


# =========================
# Parsing helpers
# =========================
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

def extract_first_json(text: str) -> Optional[dict]:
    if not text:
        return None
    text = text.strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    m = _JSON_RE.search(text)
    if m:
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None

def normalize_label(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()

def coerce_to_allowed(raw: str, allowed: List[str]) -> str:
    raw_n = normalize_label(str(raw or ""))
    allowed_map = {normalize_label(a): a for a in allowed}
    if raw_n in allowed_map:
        return allowed_map[raw_n]
    for a_norm, a in allowed_map.items():
        if raw_n and (raw_n in a_norm or a_norm in raw_n):
            return a
    return str(raw or "")


# =========================
# OpenRouter client + call
# =========================
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

def provider_prefs_for_model(model: str) -> dict:
    # Avoid Azure for OpenAI models to reduce Azure content_filter false positives.
    if model.startswith("openai/"):
        return {
            "provider": {
                "order": ["openai"],
                "ignore": ["azure"],
                "allow_fallbacks": True,
            }
        }
    return {}

def call_chat_json(model: str, messages: List[Dict[str,str]], max_tokens: int = 120, temperature: float = 0.0) -> str:
    extra = provider_prefs_for_model(model)

    # Try strict JSON mode; ONLY retry without response_format for format-related failures
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format={"type":"json_object"},
            extra_body=extra if extra else None,
        )
        return (resp.choices[0].message.content or "").strip()

    except BadRequestError as e:
        msg = str(e).lower()
        format_related = ("response_format" in msg) or ("json_object" in msg) or ("invalid_argument" in msg)

        if not format_related:
            # let caller handle (e.g., content_filter)
            raise

        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body=extra if extra else None,
        )
        return (resp.choices[0].message.content or "").strip()

def safe_call(model_id: str, messages: List[Dict[str,str]], fallback_key: str) -> str:
    try:
        return call_chat_json(model_id, messages)
    except BadRequestError as e:
        s = str(e)
        # Azure/OpenAI content filter / Responsible AI policy triggers -> keep run alive
        if ("content_filter" in s) or ("ResponsibleAIPolicyViolation" in s) or ("responsible ai policy" in s.lower()):
            return json.dumps({fallback_key: UNKNOWN}, ensure_ascii=False)
        raise

from sklearn.metrics import accuracy_score, f1_score
import csv
from pathlib import Path
import re
import pandas as pd
import time
from tqdm import tqdm

MODELS_8 = [
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "qwen/qwen3-32b",
    "qwen/qwen3-14b",
    "google/gemma-3-27b-it",
    "google/gemma-3-12b-it",
    "meta-llama/llama-3.3-70b-instruct",
    "meta-llama/llama-3.1-8b-instruct",
]

OUTDIR = Path("/home/sadegh/Translation/results")
OUTDIR.mkdir(parents=True, exist_ok=True)

# Settings
DIALECT_LANGUAGE_SOURCE = "gold"         # or "pred"
ACCENT_LANGUAGE_DIALECT_SOURCE = "gold"  # or "pred"
N_SAMPLES = None                         # None for full dataset
SLEEP_S = 0.1

# Keep original order
df_eval = df.copy()
if N_SAMPLES is not None:
    df_eval = df_eval.head(N_SAMPLES).reset_index(drop=True)

fieldnames = [
    "model_id","row_id","text",
    "gold_language","pred_language",
    "gold_dialect","pred_dialect",
    "gold_accent","pred_accent",
    "raw_language_response","raw_dialect_response","raw_accent_response",
]

def safe_name(model_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", model_id)

for model_id in MODELS_8:
    print("\n=== Running:", model_id, "===")

    model_tag = safe_name(model_id)
    rows_path = OUTDIR / f"results_{model_tag}.csv"
    metrics_path = OUTDIR / f"metrics_{model_tag}.csv"

    # per-model storage
    lg, lp = [], []
    dg, dp = [], []
    ag, ap = [], []

    # create per-model results file
    # Check if metrics already exist for this model; if so, skip
    if metrics_path.exists():
        print(f"⏭️  Skipping {model_id} — metrics file already exists: {metrics_path}")
        continue
    with rows_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for row_id, row in enumerate(tqdm(df_eval.itertuples(index=False), total=len(df_eval))):
            text = row.text
            gold_lang, gold_dia, gold_acc = row.language, row.dialect, row.accent

            # 1) language
            raw_lang = safe_call(model_id, make_language_messages(model_id, text, LANGUAGES), "language")
            pl_json = extract_first_json(raw_lang) or {}
            pred_lang = coerce_to_allowed(pl_json.get("language",""), LANGUAGES)

            # 2) dialect
            dia_lang = gold_lang if DIALECT_LANGUAGE_SOURCE == "gold" else pred_lang
            allowed_dialects = DIALECTS_BY_LANGUAGE.get(dia_lang, [UNKNOWN] if ADD_UNKNOWN else [])
            raw_dia = safe_call(model_id, make_dialect_messages(model_id, text, dia_lang, allowed_dialects), "dialect")
            pd_json = extract_first_json(raw_dia) or {}
            pred_dia = coerce_to_allowed(pd_json.get("dialect",""), allowed_dialects)

            # 3) accent
            if ACCENT_LANGUAGE_DIALECT_SOURCE == "gold":
                acc_lang, acc_dia = gold_lang, gold_dia
            else:
                acc_lang, acc_dia = pred_lang, pred_dia

            allowed_accents = sorted(
                ACCENTS_BY_LANGUAGE_DIALECT.get((acc_lang, acc_dia), {UNKNOWN} if ADD_UNKNOWN else set())
            )
            raw_acc = safe_call(model_id, make_accent_messages(model_id, text, acc_lang, acc_dia, allowed_accents), "accent")
            pa_json = extract_first_json(raw_acc) or {}
            pred_acc = coerce_to_allowed(pa_json.get("accent",""), allowed_accents)

            # write row
            w.writerow({
                "model_id": model_id,
                "row_id": row_id,
                "text": text,
                "gold_language": gold_lang,
                "pred_language": pred_lang,
                "gold_dialect": gold_dia,
                "pred_dialect": pred_dia,
                "gold_accent": gold_acc,
                "pred_accent": pred_acc,
                "raw_language_response": raw_lang,
                "raw_dialect_response": raw_dia,
                "raw_accent_response": raw_acc,
            })

            # collect for metrics
            lg.append(gold_lang); lp.append(pred_lang)
            dg.append(gold_dia);  dp.append(pred_dia)
            ag.append(gold_acc);  ap.append(pred_acc)

            time.sleep(SLEEP_S)

    # per-model metrics file
    metrics_rows = [
        {"model_id": model_id, "task":"language",
         "accuracy": accuracy_score(lg, lp),
         "macro_f1": f1_score(lg, lp, average="macro"),
         "n": len(lg)},
        {"model_id": model_id, "task":"dialect",
         "accuracy": accuracy_score(dg, dp),
         "macro_f1": f1_score(dg, dp, average="macro"),
         "n": len(dg)},
        {"model_id": model_id, "task":"accent",
         "accuracy": accuracy_score(ag, ap),
         "macro_f1": f1_score(ag, ap, average="macro"),
         "n": len(ag)},
    ]
    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)

    print("Saved:")
    print(" -", rows_path)
    print(" -", metrics_path)
