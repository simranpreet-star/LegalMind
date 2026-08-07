"""
Rule-based clause risk triage.

This is a heuristic linguistic filter for deciding which clauses deserve
human attention first. It is not a legal judgment and must not be presented
as one.

One design point that materially changes how the output should be read: a
clause where no pattern matched is reported as "Unassessed", not "Medium".
Collapsing those two into one bucket makes the risk distribution
uninterpretable, because you can no longer tell a genuinely middling clause
from one the rulebook simply does not cover. Any reported risk breakdown
should be accompanied by the coverage figure from `assess_document_risk`,
which says what share of clauses any rule fired on at all.
"""

import re

RISK_PATTERNS = {
    "Liability": {
        "high": [
            r"\bunlimited liability\b",
            r"\bno limitation of liability\b",
            r"\buncapped\b",
            r"\bgross negligence\b",
            r"\bwillful misconduct\b",
        ],
        "medium": [
            r"\bindirect\b",
            r"\bconsequential damages\b",
            r"\bcap on liability\b",
            r"\baggregate liability\b",
            r"\bshall not exceed\b",
        ],
        "low": [r"\bmutual limitation\b", r"\bexcept for gross negligence\b"],
    },
    "Termination": {
        "high": [
            r"\bsole discretion\b",
            r"\bwithout cause\b",
            r"\bimmediate termination\b",
            r"\bno notice\b",
        ],
        "medium": [
            r"\b30 days\b",
            r"\b60 days\b",
            r"\b90 days\b",
            r"\bfor convenience\b",
            r"\bmaterial breach\b",
            r"\bfails to cure\b",
        ],
        "low": [r"\bmutual agreement\b", r"\b180 days\b"],
    },
    "Payment": {
        "high": [
            r"\bnon-refundable\b",
            r"\bpenalty\b",
            r"\blate fee\b",
            r"\bliquidated damages\b",
        ],
        "medium": [
            r"\bminimum commitment\b",
            r"\bvolume\b",
            r"\bdiscount\b",
            r"\brevenue share\b",
            r"\bmonthly fee\b",
            r"\bshall pay\b",
        ],
        "low": [r"\bstandard rate\b", r"\bnet 30\b", r"\bnet 60\b"],
    },
    "Governing Law": {
        "high": [
            r"\bwaive.*jury\b",
            r"\bexclusive jurisdiction\b",
            r"\bwithout regard to conflict\b",
        ],
        "medium": [r"\barbitration\b", r"\bgoverned by\b", r"\blaws of\b"],
        "low": [r"\bmutual choice of law\b"],
    },
    "Exclusivity": {
        "high": [
            r"\bperpetual\b",
            r"\bworldwide exclusiv",
            r"\bsole and exclusive\b",
            r"\bshall not\b.*\bcompeting\b",
        ],
        "medium": [
            r"\bnon-compete\b",
            r"\bnon-solicit\b",
            r"\bterritory\b",
            r"\bexclusiv(e|ely)\b",
        ],
        "low": [r"\bnon-exclusive\b", r"\blimited exclusivity\b"],
    },
}

UNASSESSED = "Unassessed"
RISK_LEVELS = ("High", "Medium", "Low", UNASSESSED)
RISK_ORDER = {"Low": 0, "Medium": 1, "High": 2, UNASSESSED: -1}

# Compiled once at import; the patterns are static and re-compiling them per
# clause dominated the runtime on document-sized inputs.
_COMPILED = {
    category: {level: [re.compile(p, re.IGNORECASE) for p in patterns]
               for level, patterns in levels.items()}
    for category, levels in RISK_PATTERNS.items()
}


def assess_clause_risk(text, category):
    """
    Returns {"risk", "matched_patterns", "rule_fired"}.

    risk is High/Medium/Low when a pattern matched, and "Unassessed" when
    none did. rule_fired makes that distinction machine-readable so callers
    don't have to infer it from an empty match list.
    """
    compiled = _COMPILED.get(category)
    if compiled is None:
        return {
            "risk": UNASSESSED,
            "matched_patterns": [],
            "rule_fired": False,
            "note": f"no rules defined for category '{category}'",
        }

    matched = {level: [p.pattern for p in patterns if p.search(text)]
               for level, patterns in compiled.items()}

    if matched["high"]:
        risk = "High"
    elif matched["medium"]:
        risk = "Medium"
    elif matched["low"]:
        risk = "Low"
    else:
        return {"risk": UNASSESSED, "matched_patterns": [], "rule_fired": False}

    return {
        "risk": risk,
        "matched_patterns": matched["high"] + matched["medium"] + matched["low"],
        "rule_fired": True,
    }


def assess_document_risk(clauses):
    """
    clauses: list of {"text", "category"} dicts, typically classifier output.

    Returns the per-clause assessments plus a summary carrying the coverage
    rate -- the share of clauses any rule matched. A high Unassessed count
    means the rulebook is thin for this document, not that the document is
    low risk, and the summary is written so that reading it the wrong way
    takes effort.
    """
    assessed = [
        {**clause, **assess_clause_risk(clause["text"], clause.get("category", ""))}
        for clause in clauses
    ]

    summary = {level: 0 for level in RISK_LEVELS}
    for c in assessed:
        summary[c["risk"]] += 1

    total = len(assessed)
    n_fired = sum(1 for c in assessed if c["rule_fired"])

    return {
        "clauses": assessed,
        "summary": summary,
        "coverage": {
            "n_clauses": total,
            "n_rules_fired": n_fired,
            "pct_covered": round(n_fired / total * 100, 1) if total else 0.0,
        },
    }
