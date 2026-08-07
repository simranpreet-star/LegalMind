"""
Rewrite suggestions for clauses flagged as risky.

Providers are tried in order: Groq, then Anthropic, then an offline template.
Previously the notebook used Groq and the app used Anthropic, so the two
produced different output for the same clause; they now share this one path.

Every provider returns the same shape -- {"rewrite": str, "changes": list,
"source": str} -- because a caller that has to branch on the type of
`changes` will eventually get it wrong, and mixed types made the exported
CSV unusable.

These are drafting suggestions for a human to review, not legal advice, and
nothing here has been validated by a lawyer.
"""

import os
import time

PROMPT = """You are assisting with contract clause review.

Category: {category}
Risk level: {risk}
Flagged language: {matched_patterns}

Original clause:
{clause_text}

Rewrite the clause to reduce legal risk while preserving the commercial
intent. Respond in exactly this format and nothing else:

REWRITE:
<rewritten clause text>

CHANGES:
- <change 1>
- <change 2>
"""

SOFTEN_MAP = {
    "sole discretion": "reasonable discretion, with prior written notice",
    "without cause": "for material breach or with 30 days written notice",
    "unlimited liability": "liability capped at fees paid in the preceding 12 months",
    "no limitation of liability": "liability limited as set out in this clause",
    "non-refundable": "refundable on a pro-rated basis in specified circumstances",
    "perpetual": "term-limited, renewable by mutual agreement",
}


def _parse_llm_output(text):
    rewrite_parts, changes = [], []
    mode = None
    for line in text.strip().split("\n"):
        stripped = line.strip()
        if stripped.upper().startswith("REWRITE:"):
            mode = "rewrite"
            remainder = stripped[len("REWRITE:"):].strip()
            if remainder:
                rewrite_parts.append(remainder)
        elif stripped.upper().startswith("CHANGES:"):
            mode = "changes"
        elif mode == "changes" and stripped.startswith(("-", "*", "•")):
            changes.append(stripped.lstrip("-*• ").strip())
        elif mode == "rewrite" and stripped:
            rewrite_parts.append(stripped)
    return {"rewrite": " ".join(rewrite_parts).strip(), "changes": changes}


def _call_groq(prompt, model="llama-3.3-70b-versatile", timeout=60):
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None
    import requests

    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": "You are an expert commercial contract lawyer."},
                {"role": "user", "content": prompt},
            ],
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _call_anthropic(prompt, model="claude-sonnet-4-6", max_tokens=800, timeout=60):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    import requests

    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return "".join(b.get("text", "") for b in resp.json().get("content", []))


def _template_rewrite(clause_text, category):
    """
    Deterministic offline placeholder. Substitutes the most aggressive
    phrases where a known softer form exists. This is a smoke test so the
    pipeline runs without an API key -- it is not an evaluation output and
    should not appear in a results table.
    """
    rewritten = clause_text
    changes = []
    lowered = clause_text.lower()
    for phrase, replacement in SOFTEN_MAP.items():
        idx = lowered.find(phrase)
        if idx != -1:
            rewritten = rewritten[:idx] + replacement + rewritten[idx + len(phrase):]
            lowered = rewritten.lower()
            changes.append(f"replaced '{phrase}' with '{replacement}'")

    if not changes:
        changes.append(
            f"no offline substitution available for this {category} clause; "
            "set GROQ_API_KEY or ANTHROPIC_API_KEY for a generated rewrite"
        )
    return {"rewrite": rewritten, "changes": changes}


PROVIDERS = (("groq", _call_groq), ("anthropic", _call_anthropic))


def generate_rewrite(clause_text, category, risk, matched_patterns, retries=1):
    """
    Returns {"rewrite", "changes", "source"}, or None if the clause is not
    flagged Medium/High. `source` records which provider produced it, so a
    results table can separate real generations from template fallbacks.
    """
    if risk not in ("Medium", "High"):
        return None

    prompt = PROMPT.format(
        category=category,
        risk=risk,
        matched_patterns=", ".join(matched_patterns) or "none",
        clause_text=clause_text,
    )

    for name, call in PROVIDERS:
        for attempt in range(retries + 1):
            try:
                output = call(prompt)
                if output:
                    parsed = _parse_llm_output(output)
                    if parsed["rewrite"]:
                        return {**parsed, "source": name}
                break  # provider not configured; move to the next one
            except Exception as exc:  # noqa: BLE001 - provider errors are expected
                message = str(exc)
                print(f"[{name}] attempt {attempt + 1} failed: {message}")
                if "429" in message or "rate" in message.lower():
                    break  # backing off further won't help inside one call
                if attempt < retries:
                    time.sleep(2)

    return {**_template_rewrite(clause_text, category), "source": "template-fallback"}
