"""dejavu — prompt caching cost simulator.

No API key needed. Reproduces the README math: same chatbot, same answers,
with vs without prompt caching. Prove the 9x to yourself.

    python demo.py
    python demo.py --fixed 20000 --questions 100 --in 2.50 --cached 0.25
"""

from __future__ import annotations

import argparse


def cost(
    fixed_tokens: int,
    n_questions: int,
    price_in: float,
    price_cached: float,
    avg_question_tokens: int = 0,
) -> tuple[float, float]:
    """Return (cost_without_cache, cost_with_cache) in dollars.

    Prices are dollars per 1M input tokens. The fixed block (system prompt +
    document) is what caching saves; the per-question tokens are always fresh.
    """
    per_m = 1_000_000
    fresh = n_questions * avg_question_tokens * price_in / per_m

    # Without cache: the fixed block is re-read on every single question.
    without = n_questions * fixed_tokens * price_in / per_m + fresh

    # With cache: first question pays full price, the rest hit the cache.
    with_cache = (
        fixed_tokens * price_in / per_m
        + (n_questions - 1) * fixed_tokens * price_cached / per_m
        + fresh
    )
    return without, with_cache


def main() -> None:
    p = argparse.ArgumentParser(description="prompt caching cost simulator")
    p.add_argument("--fixed", type=int, default=20_000,
                   help="fixed tokens that never change (system + doc)")
    p.add_argument("--questions", type=int, default=100,
                   help="number of questions in the session")
    p.add_argument("--in", dest="price_in", type=float, default=2.50,
                   help="input price $/1M tokens")
    p.add_argument("--cached", dest="price_cached", type=float, default=0.25,
                   help="cached input price $/1M tokens")
    p.add_argument("--qtokens", type=int, default=0,
                   help="avg tokens per question (the part that changes)")
    args = p.parse_args()

    without, with_cache = cost(
        args.fixed, args.questions, args.price_in, args.price_cached, args.qtokens
    )
    saving = without / with_cache if with_cache else float("inf")

    print(f"  fixed block : {args.fixed:,} tokens  (goes at the START)")
    print(f"  questions   : {args.questions}")
    print(f"  input       : ${args.price_in:.2f} / 1M    cached: ${args.price_cached:.2f} / 1M")
    print()
    print(f"  WITHOUT cache : ${without:6.2f}")
    print(f"  WITH cache    : ${with_cache:6.2f}")
    print(f"  ---> {saving:.1f}x cheaper. Same chatbot, same answers.")


if __name__ == "__main__":
    main()
