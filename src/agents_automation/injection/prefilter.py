"""Layer 2: a deliberately broad net. It decides whether the (costly) judge runs at all.

False positives here are cheap (one judge call); false negatives skip the judge entirely, so
the vocabulary errs on the wide side. Plain workflow text without any authority words is
delivered without a judge call.
"""

from __future__ import annotations

import re

BROAD_AUTHORITY = re.compile(
    r"\b(?:approv\w*|pre-?approv\w*|authori[sz]\w*|admin\w*|system\w*|ignore\w*|disregard\w*|"
    r"override\w*|bypass\w*|skip\w*|waive\w*|exempt\w*|valid\w*|verif\w*|certif\w*|"
    r"instruction\w*|polic\w*|compliance|complian\w*|mandat\w*|ceo|cfo|cto|director\w*|"
    r"official\w*|urgent\w*|priority|automatic\w*|immediate\w*|review\w*|security|"
    r"sign(?:ed)?[\s-]+off|signed|sanction\w*|endors\w*|vetted|cleared|forget\w*|"
    r"green[\s-]?lit|ok(?:ay)?['\u2019]?d|final|as\s+is|straight\s+through|"
    r"no\s+(?:further\s+|human\s+|manual\s+)?(?:checks?|review)|"
    r"do\s+not\s+(?:flag|hold|check|review)|"
    r"rubber[\s-]?stamp\w*|(?:passed|confirmed|ok(?:ay)?|okayed)\s+by|unchanged|"
    r"dear\s+(?:ai|assistant|model|system|reviewers?|checkers?)|"
    r"nothing\s+(?:more|left|else)\s+to\s+(?:check|look|review)|"
    r"send\s+(?:it\s+)?(?:on|directly|straight)|pay\s+out|"
    r"note\s+(?:to|for)\s+(?:the\s+)?(?:ai|assistant|model|reviewers?|checkers?|system))\b",
    re.IGNORECASE,
)


def needs_judge(text: str) -> bool:
    return BROAD_AUTHORITY.search(text) is not None
