"""query_normalize —— 轻量 NLP 预处理（查询进向量库前只做“轻量清洗”，不重写语义）。

职责边界（见 ../aigc_docs/06-会话上下文检索与工具组装机制.md 第 2 节）：
- LLM 不改写查询；这里只做确定性、可测试的机械清洗：
  1. 错别字/别名校正（本地小词典，命中即替换）
  2. 语序/停用词（去语气词/无意义停顿）
  3. 按“独立子意图”切成 2~3 条 raw 子句（每条仍是原话片段）
  4. （可选）拼音/形近容错开关，留给后端 lexicon 兜底
- 明确不做：扩写背景、补全省略、把“上次那个”脑补成具体对象。

清洗后输出仍是“原样/轻清洗后”的查询串；多条子句 = 多路原样查询并行进库（multi-query raw）。
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

# 语气词/无意义停顿：对检索无益，可安全去掉（“帮我”这类祈使助词也算杂质）。
DEFAULT_STOPWORDS_ZH = {
    "请帮我", "麻烦帮我", "帮我", "帮我们", "请你", "麻烦你", "能不能", "可不可以",
    "请问", "那一个", "那个", "这个", "就是", "的话", "一下吧", "一下", "一下下",
    "我想", "我要", "我们需要", "给我们", "把", "将", "给",
    "你好", "嗨", "好的", "嗯", "呃", "然后", "接着", "接下来", "顺便", "还有",
}
DEFAULT_STOPWORDS_EN = {"please", "could", "can", "you", "i", "we", "the", "a", "an"}

# 子意图切分：遇到这些连接词/标点就把句子拆成独立子句。
# 注意顺序：长的连接词在前，避免被短词先截断。
DEFAULT_SPLIT_TERMS = [
    "并且", "同时", "然后呢", "然后再", "另外再", "另外", "同时再", "以及",
    "还有", "再帮", "顺便", "然后", "接着", "再",
    "；", "。", "？", "！", "?", "!", "\n",
    "，", ",", "和", "与", "跟",
]

# 每条子句保留原始片段，最多拆出这几条（避免过度切分把单意图拆散）。
DEFAULT_MAX_CLAUSES = 3

# 切分时用的临时分隔符：内容中几乎不可能出现的 Unicode 控制符，避免与正文冲突。
_SPLIT_MARK = "\u0001"


class QueryNormalizer:
    """确定性预处理管道：alias 校正 -> 去停用词 -> 拆子意图。

    可传入本地词典（项目/工具/文件名/人名/术语表）。词典把“错误写法/别名”映射到
    正确写法（输入法/形近/拼音错都可放这里，命中即替换）。
    """

    def __init__(
        self,
        aliases: Optional[Dict[str, str]] = None,
        stopwords: Optional[set] = None,
        split_terms: Optional[List[str]] = None,
        max_clauses: int = DEFAULT_MAX_CLAUSES,
        lowercase_en: bool = True,
        remove_fillers: bool = True,
    ) -> None:
        self.aliases: Dict[str, str] = dict(aliases or {})
        self.stopwords = set(stopwords) if stopwords is not None else (
            DEFAULT_STOPWORDS_ZH | DEFAULT_STOPWORDS_EN
        )
        self.split_terms = list(split_terms or DEFAULT_SPLIT_TERMS)
        self.max_clauses = max(1, int(max_clauses))
        self.lowercase_en = lowercase_en
        self.remove_fillers = remove_fillers
        # 别名按长度降序，避免短 key 抢先替换长 key 的一部分
        self._alias_keys = sorted(self.aliases, key=len, reverse=True)

    # ---------- 清洗 ----------

    def clean(self, text: str) -> str:
        """单条查询清洗：别名校正 -> 去停用词 -> 归一空白 -> 英文小写。不拆句。"""
        if text is None:
            return ""
        t = str(text).strip()
        for key in self._alias_keys:
            t = t.replace(key, self.aliases[key])
        if self.remove_fillers:
            for word in sorted(self.stopwords, key=len, reverse=True):
                if not word:
                    continue
                if re.fullmatch(r"[a-zA-Z0-9_]+", word):
                    t = re.sub(r"\b" + re.escape(word) + r"\b", " ", t)
                else:
                    t = t.replace(word, " ")
        t = re.sub(r"\s+", " ", t).strip(" ，,。；;！!？?、")
        if self.lowercase_en:
            t = re.sub(r"[A-Za-z][A-Za-z0-9_]*", lambda m: m.group(0).lower(), t)
        return t

    # ---------- 切分 + 归一化 ----------

    def _split_on_any(self, text: str) -> List[str]:
        ordered = sorted([x for x in self.split_terms if x], key=len, reverse=True)
        tmp = text
        for term in ordered:
            tmp = tmp.replace(term, _SPLIT_MARK)
        return [p for p in tmp.split(_SPLIT_MARK)]

    def normalize(self, text: str) -> List[str]:
        """返回 1~max_clauses 条“轻清洗后 raw 子句”。

        - 空输入 -> []（调用方据此走“最近优先 fallback”）
        - 清洗后只剩一条 -> 返回该条
        - 含多子意图 -> 拆开各自清洗，最多 max_clauses 条
        """
        if not text or not str(text).strip():
            return []
        clauses: List[str] = []
        for piece in self._split_on_any(str(text)):
            cleaned = self.clean(piece)
            if cleaned:
                clauses.append(cleaned)
            if len(clauses) >= self.max_clauses:
                break
        # 去重且保序（同一条子句被标点切成重复片段的场景）
        seen = set()
        unique: List[str] = []
        for c in clauses:
            if c not in seen:
                seen.add(c)
                unique.append(c)
        return unique

    def __call__(self, text: str) -> List[str]:
        return self.normalize(text)


# 顶层函数入口（供 controller / API 直接调用，也可离线单测）
def query_normalize(
    text: str,
    aliases: Optional[Dict[str, str]] = None,
    max_clauses: int = DEFAULT_MAX_CLAUSES,
) -> List[str]:
    """把一条 user 请求清洗成 2~3 条 raw 子句；不改写语义。"""
    return QueryNormalizer(aliases=aliases, max_clauses=max_clauses).normalize(text)
