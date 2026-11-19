"""
nl_parser_v2.py
自然语言 -> Parsed Query(JSON),与 query_engine.py 配套。

流程:
  1. LLM 解析:prompt 里放 列名/类型/取值范围(自动从 DataFrame 生成)+ JSON schema + few-shot,只要求返回 JSON
  2. 校验:复用 query_engine.validate_query(列名存在、op 白名单、取值合法)
     不合格 -> 把错误信息喂回 LLM 重试(最多 MAX_RETRIES 次)
  3. 兜底:LLM 出错 / 超时 / 重试耗尽 -> 规则解析(正则,中英文),保证现场演示不翻车

用法:
    import anthropic
    from nl_parser_v2 import parse_query
    out = parse_query("patients older than 65, survival by treatment", df, client=anthropic.Anthropic())
    out.query   # 校验通过的 dict,直接传给 query_engine.run_query
    out.source  # "llm" | "rules"
    out.notes   # 重试 / 兜底原因,可显示在 Parsed Query 面板

不传 client(或 use_llm=False)则直接走规则解析,完全离线可用。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from query_engine import (
    ALLOWED_OPS,
    ALLOWED_PLOTS,
    OUTCOME_MAP,
    QueryError,
    validate_query,
)

MODEL = "claude-sonnet-5-5"   # 按需替换
MAX_RETRIES = 2
MAX_CATEGORIES_LISTED = 15
MAX_CATEGORICAL_UNIQUE = 30   # 规则解析时,取值数超过它的列(如 patient_id)不参与匹配

# ----------------------------------------------------------------------------
# JSON schema(放进 prompt)
# ----------------------------------------------------------------------------
QUERY_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "filters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "column": {"type": "string", "description": "Must be a column that exists in the data"},
                    "op": {"type": "string", "enum": sorted(ALLOWED_OPS)},
                    "value": {"description": "Number or string; a list when op is 'in'"},
                },
                "required": ["column", "op", "value"],
            },
        },
        "outcome": {"type": ["string", "null"], "enum": sorted(OUTCOME_MAP) + [None]},
        "group_by": {"type": ["string", "null"], "description": "Categorical column name or null"},
        "plot_type": {"type": "string", "enum": sorted(ALLOWED_PLOTS)},
    },
    "required": ["filters", "outcome", "group_by", "plot_type"],
}

# few-shot 里的列名/取值要和你的真实数据一致;不一致时 LLM 会被示例带偏
FEW_SHOTS: list[tuple[str, dict[str, Any]]] = [
    (
        "Show overall survival for patients older than 65, split by treatment",
        {"filters": [{"column": "age", "op": ">", "value": 65}],
         "outcome": "overall_survival", "group_by": "treatment", "plot_type": "km_curve"},
    ),
    (
        "女性、III 或 IV 期患者的无进展生存,按治疗方案比较",
        {"filters": [{"column": "sex", "op": "==", "value": "Female"},
                     {"column": "stage", "op": "in", "value": ["III", "IV"]}],
         "outcome": "progression_free_survival", "group_by": "treatment", "plot_type": "km_curve"},
    ),
    (
        "distribution of age by sex",
        {"filters": [], "outcome": None, "group_by": "sex", "plot_type": "histogram"},
    ),
    (
        "compare survival time between Drug A and Placebo as a boxplot",
        {"filters": [{"column": "treatment", "op": "in", "value": ["Drug A", "Placebo"]}],
         "outcome": "overall_survival", "group_by": "treatment", "plot_type": "boxplot"},
    ),
    (
        "How does mean survival time change with age in patients under 80?",
        {"filters": [{"column": "age", "op": "<", "value": 80}],
         "outcome": "overall_survival", "group_by": None, "plot_type": "trend"},
    ),
]


# ----------------------------------------------------------------------------
# 数据描述(列名 / 类型 / 取值范围)
# ----------------------------------------------------------------------------
def describe_schema(df: pd.DataFrame) -> str:
    lines = []
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            lines.append(f"- {col} (numeric): min={s.min():g}, max={s.max():g}")
        else:
            vals = sorted(map(str, s.dropna().unique()))
            if len(vals) > MAX_CATEGORIES_LISTED:
                # 高基数列(patient_id、日期等)只给个数,绝不列出取值:这份描述会发给 LLM,不能带上标识符
                lines.append(f"- {col} (categorical, {len(vals)} distinct values; not listed)")
            else:
                lines.append(f"- {col} (categorical): {vals}")
    return "\n".join(lines)


def _current_schema() -> dict[str, Any]:
    """outcome 的可选值随数据集变化(OUTCOME_MAP 会被 configure_outcomes 改写),所以每次现算。"""
    schema = json.loads(json.dumps(QUERY_JSON_SCHEMA))
    schema["properties"]["outcome"]["enum"] = sorted(OUTCOME_MAP) + [None]
    return schema


def build_system_prompt(df: pd.DataFrame) -> str:
    outcomes = "\n".join(f"- {k} (uses columns {t}, {e})" for k, (t, e) in OUTCOME_MAP.items())
    shots = "\n\n".join(
        f"Question: {q}\nJSON: {json.dumps(a, ensure_ascii=False)}" for q, a in FEW_SHOTS
    )
    return f"""You convert a clinician's natural-language question into a structured query over a patient table.

Dataset columns:
{describe_schema(df)}

Available outcomes:
{outcomes}

Rules:
- Return ONLY one JSON object. No prose, no markdown fences.
- "column" must be one of the dataset columns above. Never invent columns.
- "op" must be one of: {", ".join(sorted(ALLOWED_OPS))}. Use "in" with a list for multiple values.
- For categorical columns use values exactly as listed above (normalize synonyms, e.g. "women" -> "Female").
- "between X and Y" becomes two filters on the same column (>= X and <= Y).
- "outcome" is one of the outcomes above, or null if the question is not about survival.
- "group_by" is a categorical column or null.
- "plot_type" is one of: {", ".join(sorted(ALLOWED_PLOTS))}. Use km_curve for survival curves.
- If something in the question cannot be expressed with these columns, omit it rather than guess.

JSON schema:
{json.dumps(_current_schema(), ensure_ascii=False)}

Examples:

{shots}"""


# ----------------------------------------------------------------------------
# 结果容器
# ----------------------------------------------------------------------------
@dataclass
class ParseOutput:
    query: dict[str, Any]
    source: str                      # "llm" | "rules"
    notes: list[str] = field(default_factory=list)


def _to_public(clean: dict[str, Any]) -> dict[str, Any]:
    """validate_query 把数值转成了 float;对外展示时把整数值还原成 int(65.0 -> 65)。"""
    def fix(v):
        return int(v) if isinstance(v, float) and v.is_integer() else v

    out = dict(clean)
    out["filters"] = [
        {**f, "value": [fix(x) for x in f["value"]] if isinstance(f["value"], list) else fix(f["value"])}
        for f in clean["filters"]
    ]
    return out


# ----------------------------------------------------------------------------
# 1) LLM 解析 + 校验重试
# ----------------------------------------------------------------------------
def _extract_json(text: str) -> dict[str, Any]:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)   # 去掉可能的代码围栏
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise QueryError("The response contains no JSON object")
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError as e:
        raise QueryError(f"Could not parse JSON: {e}")


def llm_parse(question: str, df: pd.DataFrame, client: Any, model: str = MODEL,
              max_retries: int = MAX_RETRIES) -> ParseOutput:
    """调用 LLM;校验失败则带着错误信息重试。最终失败抛 QueryError。"""
    system = build_system_prompt(df)
    messages = [{"role": "user", "content": question}]
    notes: list[str] = []
    last_err: Exception | None = None

    for attempt in range(max_retries + 1):
        resp = client.messages.create(model=model, max_tokens=500, system=system,
                                      messages=messages, timeout=15)
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        try:
            clean = validate_query(df, _extract_json(text))
            if attempt:
                notes.append(f"LLM output passed validation on attempt {attempt + 1}")
            return ParseOutput(query=_to_public(clean), source="llm", notes=notes)
        except (QueryError, TypeError, AttributeError) as e:
            last_err = e
            notes.append(f"Attempt {attempt + 1} failed validation: {e}")
            messages += [
                {"role": "assistant", "content": text},
                {"role": "user", "content": f"That JSON is invalid: {e}\nReturn a corrected JSON object only."},
            ]
    raise QueryError(f"LLM output still invalid after {max_retries} retries: {last_err}")


# ----------------------------------------------------------------------------
# 2) 规则解析兜底(中英文)
# ----------------------------------------------------------------------------
# 用户说法 -> 数据列名候选
COLUMN_SYNONYMS = {
    "treatment": ["treatment", "arm", "drug", "therapy", "治疗", "疗法", "药物", "用药", "方案"],
    "sex": ["sex", "gender", "性别"],
    "stage": ["stage", "分期", "期别"],
}

# 用户说法 -> 规范取值(会再去 df 里按大小写不敏感匹配真实取值)
VALUE_SYNONYMS = {
    "women": "female", "woman": "female", "females": "female", "女性": "female", "女": "female",
    "men": "male", "man": "male", "males": "male", "男性": "male", "男": "male",
}

_NUM = r"(\d+(?:\.\d+)?)"
# (正则, op);按顺序匹配,命中即取
_AGE_PATTERNS: list[tuple[str, str]] = [
    (rf"(?:between|aged?)\s*{_NUM}\s*(?:and|to|-|–)\s*{_NUM}", "between"),
    (rf"(?:older than|over|above|more than|greater than|age\s*>)\s*{_NUM}", ">"),
    (rf"(?:aged?|age)\s*(?:>=|≥)\s*{_NUM}", ">="),
    (rf"{_NUM}\s*(?:years?(?: old)?)?\s*(?:or older|and older|\+)", ">="),
    (rf"at least\s*{_NUM}", ">="),
    (rf"(?:younger than|under|below|less than|age\s*<)\s*{_NUM}", "<"),
    (rf"(?:aged?|age)\s*(?:<=|≤)\s*{_NUM}", "<="),
    (rf"(?:{_NUM}\s*(?:years?(?: old)?)?\s*(?:or younger|and younger)|at most\s*{_NUM}|up to\s*{_NUM})", "<="),
    (rf"{_NUM}\s*岁\s*(?:以上|及以上)", ">="),
    (rf"(?:大于|超过|高于|年龄大于){_NUM}\s*岁?", ">"),
    (rf"{_NUM}\s*岁\s*(?:以下|及以下)", "<="),
    (rf"(?:小于|低于|年龄小于){_NUM}\s*岁?", "<"),
]

_OUTCOME_KEYWORDS = [
    (r"progression|pfs|无进展|进展", "progression_free_survival"),
    (r"surviv|\bos\b|生存|存活", "overall_survival"),
]
_PLOT_KEYWORDS = [
    (r"box\s*plot|boxplot|箱线", "boxplot"),
    (r"histogram|distribution|分布|直方", "histogram"),
    (r"trend|change[sd]? with|趋势|变化", "trend"),
    (r"scatter|散点", "scatter"),
    (r"\bbar\b|\bcounts?\b|柱状|计数|人数", "bar"),
    (r"\bkm\b|kaplan|survival curve|生存曲线", "km_curve"),
]


def _find_column(df: pd.DataFrame, key: str) -> str | None:
    """把 key 映射到 df 里真实存在的列:先精确(大小写不敏感),再按下划线分词匹配。"""
    for c in df.columns:
        if c.lower() == key:
            return c
    for c in df.columns:
        if key in re.split(r"[_\W]+", c.lower()):
            return c
    return None


def _age_filters(q: str, age_col: str) -> list[dict[str, Any]]:
    def num(x: str):
        f = float(x)
        return int(f) if f.is_integer() else f

    for pat, op in _AGE_PATTERNS:
        m = re.search(pat, q)
        if not m:
            continue
        if op == "between":
            return [{"column": age_col, "op": ">=", "value": num(m.group(1))},
                    {"column": age_col, "op": "<=", "value": num(m.group(2))}]
        val = next(g for g in m.groups() if g)
        return [{"column": age_col, "op": op, "value": num(val)}]
    return []


def _category_hits(q: str, col: str, series: pd.Series) -> list[str]:
    actual = sorted(map(str, series.dropna().unique()), key=len, reverse=True)
    lower_map = {v.lower(): v for v in actual}
    hits: list[str] = []

    # 1) 取值本身出现在问题里
    for v in actual:
        vl = v.lower()
        if len(vl) <= 2:   # "I"/"II" 这类短取值,必须写成 "stage I" / "I期" 才算
            pat = rf"(?:{re.escape(col.lower())}|分期|期别)\s*[:=]?\s*{re.escape(vl)}\b|\b{re.escape(vl)}\s*期"
        else:
            pat = rf"(?<![a-z]){re.escape(vl)}(?![a-z])"
        if re.search(pat, q):
            hits.append(v)

    # 2) 同义词(women -> Female, 女 -> Female ...)
    for word, canon in VALUE_SYNONYMS.items():
        pat = rf"(?<![a-z]){re.escape(word)}(?![a-z])" if word.isascii() else re.escape(word)
        if canon in lower_map and re.search(pat, q):
            hits.append(lower_map[canon])

    # 3) 英文 "stage III or IV" 这种裸罗马数字列表
    if "stage" in col.lower():
        m = re.search(r"stage\s+((?:iv|iii|ii|i)(?:\s*(?:,|or|and|/|和|或)\s*(?:iv|iii|ii|i))*)", q)
        if m:
            romans = [lower_map[r] for r in re.findall(r"iv|iii|ii|i", m.group(1)) if r in lower_map]
            if romans:
                hits = [h for h in hits if h not in actual or h in romans] + romans

    return list(dict.fromkeys(hits))


def _find_group_by(q: str, df: pd.DataFrame) -> str | None:
    m = re.search(r"(?:\bby\b|\bsplit by\b|\bgrouped by\b|\bstratified by\b|按|根据|依据)\s*([a-z_ ]+|[\u4e00-\u9fa5]+)", q)
    if not m:
        return None
    phrase = m.group(1)
    best: tuple[int, str] | None = None
    for key, words in COLUMN_SYNONYMS.items():
        pos = [phrase.find(w) for w in words if w in phrase]
        col = _find_column(df, key)
        if pos and col and (best is None or min(pos) < best[0]):
            best = (min(pos), col)
    if best:
        return best[1]
    for c in df.columns:   # 直接用列名
        if c.lower() in phrase:
            return c
    return None


def rule_parse(question: str, df: pd.DataFrame) -> ParseOutput:
    q = question.lower()
    notes: list[str] = ["Parsed with the rule-based fallback"]
    filters: list[dict[str, Any]] = []

    age_col = _find_column(df, "age")
    if age_col:
        filters += _age_filters(q, age_col)

    for col in df.columns:
        s = df[col]
        if col == age_col or pd.api.types.is_numeric_dtype(s) or s.nunique() > MAX_CATEGORICAL_UNIQUE:
            continue
        hits = _category_hits(q, col, s)
        if len(hits) == 1:
            filters.append({"column": col, "op": "==", "value": hits[0]})
        elif len(hits) > 1:
            filters.append({"column": col, "op": "in", "value": hits})

    group_by = _find_group_by(q, df)
    if group_by is None and re.search(r"compare|比较|对比", q):
        multi = next((f["column"] for f in filters if f["op"] == "in"), None)
        group_by = multi or _find_column(df, "treatment")

    outcome = next((o for pat, o in _OUTCOME_KEYWORDS if re.search(pat, q)), None)
    plot_type = next((p for pat, p in _PLOT_KEYWORDS if re.search(pat, q)), None)
    if plot_type is None:
        plot_type = "km_curve" if outcome else "histogram"
    if plot_type == "km_curve" and outcome is None:
        outcome = "overall_survival"
    if plot_type in ("boxplot", "trend") and outcome is None and re.search(r"time|时间", q):
        outcome = "overall_survival"

    raw = {"filters": filters, "outcome": outcome, "group_by": group_by, "plot_type": plot_type}
    try:
        clean = validate_query(df, raw)
    except QueryError as e:
        # 规则结果也要过校验;失败就退化成最保守的全量 KM,而不是让界面报错
        notes.append(f"Rule-based result failed validation ({e}); fell back to the default query")
        clean = validate_query(df, {"filters": [], "outcome": "overall_survival",
                                    "group_by": None, "plot_type": "km_curve"})
    return ParseOutput(query=_to_public(clean), source="rules", notes=notes)


# ----------------------------------------------------------------------------
# 总入口
# ----------------------------------------------------------------------------
def parse_query(question: str, df: pd.DataFrame, client: Any | None = None,
                use_llm: bool = True, model: str = MODEL) -> ParseOutput:
    question = (question or "").strip()
    if not question:
        raise QueryError("The question is empty")

    if use_llm and client is not None:
        try:
            return llm_parse(question, df, client, model=model)
        except Exception as e:   # 校验耗尽、网络、超时、鉴权……一律兜底
            out = rule_parse(question, df)
            out.notes.insert(0, f"LLM parsing failed, fell back to rules: {type(e).__name__}: {e}")
            return out
    return rule_parse(question, df)


# ----------------------------------------------------------------------------
# 自测:python nl_parser_v2.py(不需要网络和 API key)
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np

    rng = np.random.default_rng(0)
    n = 200
    demo = pd.DataFrame({
        "age": rng.integers(30, 90, n),
        "sex": rng.choice(["Male", "Female"], n),
        "treatment": rng.choice(["Drug A", "Drug B", "Placebo"], n),
        "stage": rng.choice(["I", "II", "III", "IV"], n),
        "os_time": rng.exponential(24, n).round(1),
        "os_event": rng.integers(0, 2, n),
        "pfs_time": rng.exponential(12, n).round(1),
        "pfs_event": rng.integers(0, 2, n),
    })

    print("=== 规则解析 ===")
    for q in [
        "show overall survival for patients older than 65 by treatment",
        "65岁以上女性患者按治疗方案分组的总生存曲线",
        "progression-free survival, stage III or IV, split by sex",
        "compare Drug A and Placebo survival as a boxplot",
        "distribution of age by sex",
        "patients younger than 50",
        "women aged 40 to 60, survival grouped by stage",
    ]:
        out = rule_parse(q, demo)
        print(f"\nQ: {q}\n -> {json.dumps(out.query, ensure_ascii=False)}")

    print("\n=== 假 LLM:第一次返回非法列,第二次修正 ===")

    class _Block:
        type = "text"
        def __init__(self, text): self.text = text

    class _Resp:
        def __init__(self, text): self.content = [_Block(text)]

    class FakeClient:
        def __init__(self, replies):
            self.replies, self.calls = list(replies), 0
            self.messages = self
        def create(self, **kw):
            self.calls += 1
            return _Resp(self.replies.pop(0))

    fake = FakeClient([
        '```json\n{"filters":[{"column":"agee","op":">","value":65}],"outcome":"overall_survival","group_by":"treatment","plot_type":"km_curve"}\n```',
        '{"filters":[{"column":"age","op":">","value":65}],"outcome":"overall_survival","group_by":"treatment","plot_type":"km_curve"}',
    ])
    out = parse_query("older than 65 survival by treatment", demo, client=fake)
    print(out.source, fake.calls, "calls", json.dumps(out.query), out.notes)

    print("\n=== 假 LLM 一直出错 -> 回退规则 ===")
    class BrokenClient:
        def __init__(self): self.messages = self
        def create(self, **kw): raise TimeoutError("api down")
    out = parse_query("older than 65 survival by treatment", demo, client=BrokenClient())
    print(out.source, json.dumps(out.query), out.notes)

    print("\n=== Prompt 预览(前 700 字符)===")
    print(build_system_prompt(demo)[:700])
