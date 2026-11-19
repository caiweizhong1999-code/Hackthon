"""
Natural language -> Parsed Query (JSON)
Flow: LLM parsing -> Pydantic structure validation -> dataset semantic validation -> retry with the error on failure -> rule-based fallback if it still fails
Dependencies: pip install pydantic anthropic
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ValidationError

# ----------------------------------------------------------------------
# 1. Dataset schema (edit this to match your real data)
# ----------------------------------------------------------------------
DATASET_SCHEMA: dict[str, dict[str, Any]] = {
    "age":       {"type": "numeric"},
    "sex":       {"type": "categorical", "values": ["male", "female"]},
    "treatment": {"type": "categorical", "values": ["A", "B", "placebo"]},
    "stage":     {"type": "categorical", "values": ["I", "II", "III", "IV"]},
    "os_time":   {"type": "numeric"},
    "os_event":  {"type": "numeric"},
}

OPS = [">", "<", ">=", "<=", "==", "!=", "in"]
OUTCOMES = ["overall_survival", "progression_free_survival"]
PLOT_TYPES = ["km_curve", "boxplot", "histogram", "bar"]

# ----------------------------------------------------------------------
# 2. Pydantic models
# ----------------------------------------------------------------------
class Filter(BaseModel):
    column: str
    op: Literal[">", "<", ">=", "<=", "==", "!=", "in"]
    value: Union[int, float, str, list[Union[int, float, str]]]


class ParsedQuery(BaseModel):
    filters: list[Filter] = []
    outcome: Literal["overall_survival", "progression_free_survival"] = "overall_survival"
    group_by: Optional[str] = None
    plot_type: Literal["km_curve", "boxplot", "histogram", "bar"] = "km_curve"


class QueryValidationError(ValueError):
    pass


def validate_semantics(q: ParsedQuery, schema: dict = DATASET_SCHEMA) -> ParsedQuery:
    """Columns must exist; numeric columns use numeric comparisons; categorical values must be in the allowed set."""
    errors: list[str] = []
    for f in q.filters:
        col = schema.get(f.column)
        if col is None:
            errors.append(f"unknown column '{f.column}', valid: {list(schema)}")
            continue
        vals = f.value if isinstance(f.value, list) else [f.value]
        if f.op == "in" and not isinstance(f.value, list):
            errors.append(f"op 'in' requires a list value for '{f.column}'")
        if col["type"] == "numeric":
            if f.op == "in" or not all(isinstance(v, (int, float)) for v in vals):
                errors.append(f"column '{f.column}' is numeric; got op={f.op}, value={f.value!r}")
        else:
            if f.op not in ("==", "!=", "in"):
                errors.append(f"column '{f.column}' is categorical; op '{f.op}' not allowed")
            bad = [v for v in vals if v not in col["values"]]
            if bad:
                errors.append(f"invalid value(s) {bad} for '{f.column}', valid: {col['values']}")
    if q.group_by is not None:
        col = schema.get(q.group_by)
        if col is None:
            errors.append(f"unknown group_by column '{q.group_by}'")
        elif col["type"] != "categorical":
            errors.append(f"group_by column '{q.group_by}' must be categorical")
    if errors:
        raise QueryValidationError("; ".join(errors))
    return q


def parse_and_validate(raw: str | dict, schema: dict = DATASET_SCHEMA) -> ParsedQuery:
    if isinstance(raw, str):
        raw = raw.strip()
        raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.M).strip()  # strip possible markdown fences
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as e:
            raise QueryValidationError(f"invalid JSON: {e}") from e
    try:
        q = ParsedQuery.model_validate(raw)
    except ValidationError as e:
        raise QueryValidationError(str(e)) from e
    return validate_semantics(q, schema)


# ----------------------------------------------------------------------
# 3. Prompt
# ----------------------------------------------------------------------
FEW_SHOTS = [
    ("Show survival curves for patients older than 65 by treatment",
     {"filters": [{"column": "age", "op": ">", "value": 65}],
      "outcome": "overall_survival", "group_by": "treatment", "plot_type": "km_curve"}),
    ("Compare progression-free survival of female stage III or IV patients across treatments",
     {"filters": [{"column": "sex", "op": "==", "value": "female"},
                  {"column": "stage", "op": "in", "value": ["III", "IV"]}],
      "outcome": "progression_free_survival", "group_by": "treatment", "plot_type": "km_curve"}),
    ("KM curve for everyone on placebo, split by sex",
     {"filters": [{"column": "treatment", "op": "==", "value": "placebo"}],
      "outcome": "overall_survival", "group_by": "sex", "plot_type": "km_curve"}),
    ("Age distribution of patients under 50",
     {"filters": [{"column": "age", "op": "<", "value": 50}],
      "outcome": "overall_survival", "group_by": None, "plot_type": "histogram"}),
    ("Overall survival for treatment A patients aged 40 to 60, grouped by stage",
     {"filters": [{"column": "treatment", "op": "==", "value": "A"},
                  {"column": "age", "op": ">=", "value": 40},
                  {"column": "age", "op": "<=", "value": 60}],
      "outcome": "overall_survival", "group_by": "stage", "plot_type": "km_curve"}),
]


def build_system_prompt(schema: dict = DATASET_SCHEMA) -> str:
    cols = "\n".join(
        f"- {name} ({c['type']})" + (f", values: {c['values']}" if c["type"] == "categorical" else "")
        for name, c in schema.items()
    )
    shots = "\n\n".join(
        f"Input: {text}\nOutput: {json.dumps(out, ensure_ascii=False)}" for text, out in FEW_SHOTS
    )
    return f"""You translate a clinician's natural-language request into a JSON query for a survival-analysis tool.

Dataset columns:
{cols}

Output JSON schema:
{{
  "filters": [{{"column": <column name>, "op": one of {OPS}, "value": <number | string | list>}}],
  "outcome": one of {OUTCOMES},
  "group_by": <categorical column name or null>,
  "plot_type": one of {PLOT_TYPES}
}}

Rules:
- Use ONLY the column names and categorical values listed above; normalize synonyms (e.g. "women" -> sex == "female").
- Numeric columns use numeric values; "in" requires a list.
- "between X and Y" becomes two filters (>= X and <= Y).
- If the outcome or plot type is unspecified, default to overall_survival and km_curve.
- Return ONLY the JSON object. No prose, no markdown.

Examples:

{shots}"""


# ----------------------------------------------------------------------
# 4. LLM parsing (retry with error feedback)
# ----------------------------------------------------------------------
def llm_parse(text: str, schema: dict = DATASET_SCHEMA, max_retries: int = 2,
              model: str | None = None) -> ParsedQuery:
    import anthropic  # lazy import so the fallback still works when it is not installed / no key

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    model = model or os.getenv("PARSER_MODEL", "claude-sonnet-5-5")
    system = build_system_prompt(schema)
    messages = [{"role": "user", "content": text}]
    last_err: Exception | None = None

    for _ in range(max_retries + 1):
        resp = client.messages.create(
            model=model, max_tokens=500, temperature=0,
            system=system, messages=messages, timeout=15,
        )
        raw = resp.content[0].text
        try:
            return parse_and_validate(raw, schema)
        except QueryValidationError as e:
            last_err = e
            messages += [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": f"Your JSON was invalid: {e}\nReturn corrected JSON only."},
            ]
    raise QueryValidationError(f"LLM output failed validation after retries: {last_err}")


# ----------------------------------------------------------------------
# 5. Rule-based fallback (no network needed)
# ----------------------------------------------------------------------
_NUM = r"(\d+(?:\.\d+)?)"
_AGE_RULES: list[tuple[str, str]] = [
    (rf"(?:older|greater|more|over|above)\s+than\s+{_NUM}|(?:over|above|>)\s*{_NUM}", ">"),
    (rf"(?:younger|less|under|below|fewer)\s+than\s+{_NUM}|(?:under|below|<)\s*{_NUM}", "<"),
    (rf"{_NUM}\s*\+|{_NUM}\s+and\s+(?:older|above|over)|at\s+least\s+{_NUM}", ">="),
    (rf"at\s+most\s+{_NUM}|up\s+to\s+{_NUM}", "<="),
]
_SYNONYMS = {"women": ("sex", "female"), "woman": ("sex", "female"), "men": ("sex", "male"),
             "man": ("sex", "male"), "female": ("sex", "female"), "male": ("sex", "male"),
             # "女"/"男" (Chinese for female/male) are kept on purpose so Chinese questions still work
             "女": ("sex", "female"), "男": ("sex", "male")}


def rule_parse(text: str, schema: dict = DATASET_SCHEMA) -> ParsedQuery:
    t = text.lower()
    filters: list[dict] = []

    # Age: "between 40 and 60" / "aged 40-60"
    m = re.search(rf"(?:between|aged?)\s+{_NUM}\s*(?:and|to|-|–)\s*{_NUM}", t)
    if m and "age" in schema:
        filters += [{"column": "age", "op": ">=", "value": float(m[1])},
                    {"column": "age", "op": "<=", "value": float(m[2])}]
    elif "age" in schema:
        for pat, op in _AGE_RULES:
            m = re.search(pat, t)
            if m:
                num = next(g for g in m.groups() if g)
                filters.append({"column": "age", "op": op,
                                "value": int(float(num)) if float(num).is_integer() else float(num)})
                break

    # Synonyms (sex)
    for word, (col, val) in _SYNONYMS.items():
        if col in schema and re.search(rf"\b{re.escape(word)}\b" if word.isascii() else word, t):
            if not any(f["column"] == col for f in filters):
                filters.append({"column": col, "op": "==", "value": val})

    # Categorical values: find group_by first, then treat any values that appear as filters
    group_by = None
    # "按" / "分组" (Chinese for "by" / "group") are kept on purpose so Chinese questions still work
    m = re.search(r"(?:by|group(?:ed)?\s+by|split\s+by|across|按|分组)\s*([a-z_\u4e00-\u9fa5]+)", t)
    if m:
        cand = m[1].rstrip("s")
        for name in schema:
            if name.lower() == cand or name.lower() == m[1]:
                group_by = name
    if group_by is None and "by treatment" in t:
        group_by = "treatment"

    for name, c in schema.items():
        if c["type"] != "categorical" or name == group_by or name == "sex":
            continue
        hits = [v for v in c["values"]
                if len(v) > 1 and re.search(rf"\b{re.escape(v.lower())}\b", t)
                or re.search(rf"\b(?:{name}|arm|stage)\s+{re.escape(v.lower())}\b", t)]
        hits = list(dict.fromkeys(hits))
        if len(hits) == 1:
            filters.append({"column": name, "op": "==", "value": hits[0]})
        elif len(hits) > 1:
            filters.append({"column": name, "op": "in", "value": hits})

    outcome = ("progression_free_survival"
               if re.search(r"progression|pfs", t) else "overall_survival")
    plot = ("boxplot" if "box" in t else "histogram" if re.search(r"distribution|histogram", t)
            else "bar" if "bar" in t else "km_curve")

    return parse_and_validate(
        {"filters": filters, "outcome": outcome, "group_by": group_by, "plot_type": plot}, schema)


# ----------------------------------------------------------------------
# 6. Unified entry point
# ----------------------------------------------------------------------
def parse_query(text: str, schema: dict = DATASET_SCHEMA) -> tuple[ParsedQuery, str]:
    """Return (ParsedQuery, source 'llm' | 'rules'); the UI can display the source."""
    try:
        return llm_parse(text, schema), "llm"
    except Exception as e:  # network/key/timeout/validation failures all fall back to rules
        print(f"[parser] LLM failed ({type(e).__name__}: {e}); using rule-based fallback")
        return rule_parse(text, schema), "rules"


if __name__ == "__main__":
    tests = [
        "Show survival curves for patients older than 65 by treatment",
        "women with stage III or IV, progression-free survival, by treatment",
        "age distribution of patients under 50",
        "treatment A patients between 40 and 60 grouped by stage",
    ]
    for s in tests:
        q = rule_parse(s)
        print(s, "\n  ->", q.model_dump_json(), "\n")
