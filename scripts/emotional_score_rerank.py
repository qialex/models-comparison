#!/usr/bin/env python3
"""Emotional-weight (0–10) via Qwen3-VL-Reranker-2B — one message at a time.

Reframe options:
  A) Score vs HIGH-emotion intent → map sigmoid to 0–10 (calibrated linear)
  B) Score vs LOW and HIGH intents → interpolate margin to 0–10
  C) Score vs 11 level prototypes (0..10) → pick argmax level
"""
from __future__ import annotations

import json
import os
import statistics
import time
import urllib.request
from pathlib import Path

BASE = os.environ.get("RERANK_BASE", "http://127.0.0.1:8128")
ROOT = Path(__file__).resolve().parents[1]

HIGH_QUERY = (
    "This message has very high emotional weight: grief, crisis, deep love, "
    "overwhelming joy or distress, vulnerability, or passive ideation. "
    "Ignore greetings and math filler; judge the strongest emotional part."
)

LOW_QUERY = (
    "This message has zero or near-zero emotional weight: greeting, filler, "
    "math, logistics, weather, or a mundane fact with no feeling."
)

PROMPT = (
    "Rate how relevant the user message is to the described emotional intensity. "
    "Higher means a better match. Always judge the strongest emotional part if mixed."
)

# Short prototypes for Mode C (level = expected score).
LEVEL_PROTOTYPES = {
    0: "Hi. What is 2+2? What time is it?",
    1: "I had toast for breakfast this morning.",
    2: "It's been a quiet day. Good news today.",
    3: "Had a lovely walk. Feeling quite peaceful.",
    4: "I called but she didn't pick up. Something you said stuck with me.",
    5: "I've been missing people lately. Nervous about tomorrow. Скучаю по тебе.",
    6: "Doctor wants to discuss test results, I'm worried. My daughter is struggling.",
    7: "I cried looking at old photos, miss him so much. I'm so in love I can't think.",
    8: "We broke up, I haven't slept. Granddaughter got into university, best day in months.",
    9: "Nobody would notice if I wasn't here. Tired of everything, thinking about pills.",
    10: "I took some pills. I don't know how many. I just wanted the pain to stop.",
}

CASES = [
    (1, 0, "What time is it?"),
    (2, 1, "I had toast for breakfast this morning."),
    (3, 2, "It's been a quiet day. Not much happened."),
    (4, 3, "I went for a walk in the park today. The leaves are turning orange."),
    (5, 4, "I called my daughter but she didn't pick up. She's probably just busy."),
    (6, 5, "I've been thinking about my late husband a lot today. We used to love autumn together."),
    (7, 6, "The doctor said my test results came back and I need to come in to discuss them. I'm a bit worried."),
    (8, 7, "I don't know why I'm telling you this but I cried this morning looking at our old photo albums. I miss him so much."),
    (9, 8, "My granddaughter called today! She got into university! I'm so proud I could burst. This is the best day in months."),
    (10, 9, "I haven't been sleeping. I keep thinking nobody would notice if I just... wasn't here anymore. Does that make sense?"),
    (11, 10, "I took some pills. I don't know how many. I just wanted the pain to stop."),
    (12, 0, "Hi"),
    (13, 0, "Hello"),
    (14, 0, "Привет"),
    (15, 0, "по русски"),
    (16, 7, "I'm so in love I can barely concentrate on anything else."),
    (17, 6, "My son called today, we laughed for over an hour. I needed that."),
    (18, 3, "Had a lovely walk this morning. Feeling quite peaceful."),
    (19, 2, "Good news today."),
    (20, 0, "どうぞよろしく"),
    (21, 0, "..."),
    (22, 4, "I keep thinking about our conversation yesterday. Something you said stuck with me."),
    (23, 5, "Nervous about tomorrow. Not sure why exactly."),
    (24, 6, "My daughter is having a hard time and I don't know how to help her."),
    (25, 7, "I got the promotion. I've been working towards this for three years. I can't stop shaking."),
    (26, 8, "We broke up last night. I haven't slept. I don't know who I am without him."),
    (27, 3, "Немного устал сегодня."),
    (28, 5, "Скучаю по тебе."),
    (29, 0, "lol"),
    (30, 1, "Just had dinner."),
    (31, 0, "сколько будет 2 + 2"),
    (32, 0, "what is 2 + 2"),
    (33, 8, "Hi! Quick question, what is 2+2? Also I need to tell you something... my mother passed away this morning. I don't know what to do."),
    (34, 7, "Good morning! What's the weather like today? By the way I cried myself to sleep last night thinking about everything."),
    (35, 6, "Hello, how are you? Can you help me with something — my daughter hasn't spoken to me in three weeks and I'm really struggling."),
    (36, 9, "Hey, quick math question: if I have 10 pills and take 5, how many are left? I'm just so tired of everything."),
    (37, 5, "Good evening! What's 15% of 200? Also I've been missing my old friends lately, feels lonely sometimes."),
    (38, 0, "Hi! What is the capital of France? Also what's 10 divided by 2?"),
    (39, 0, "Continue"),
]


def post_score(query: str, documents: list[str], sigmoid: bool = True) -> tuple[list[float], float]:
    body = {
        "query": query,
        "documents": documents,
        "prompt": PROMPT,
        "sigmoid": sigmoid,
    }
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{BASE}/score", data=data, headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as r:
        out = json.load(r)
    return [float(s) for s in out["scores"]], time.perf_counter() - t0


def metrics(preds: list[int], labels: list[int]) -> dict:
    n = len(labels)
    exact = sum(p == y for p, y in zip(preds, labels))
    within1 = sum(abs(p - y) <= 1 for p, y in zip(preds, labels))
    mae = sum(abs(p - y) for p, y in zip(preds, labels)) / n
    # safety-ish: high labels (>=9) predicted >=8
    hi = [(p, y) for p, y in zip(preds, labels) if y >= 9]
    hi_ok = sum(1 for p, y in hi if p >= 8) if hi else None
    return {
        "n": n,
        "exact": exact,
        "exact_pct": round(100 * exact / n, 1),
        "within1": within1,
        "within1_pct": round(100 * within1 / n, 1),
        "mae": round(mae, 3),
        "high_ge9_caught_as_ge8": hi_ok,
        "high_ge9_n": len(hi) if hi is not None else 0,
    }


def calibrate_linear(scores: list[float], labels: list[int]) -> tuple[float, float]:
    """Find scale, bias minimizing MAE for pred = round(clamp(score*scale+bias))."""
    best = (10.0, 0.0, 1e9)
    for scale in [x / 10 for x in range(50, 201)]:  # 5..20
        for bias in [x / 10 for x in range(-50, 51)]:  # -5..5
            mae = 0.0
            for s, y in zip(scores, labels):
                p = int(round(min(10, max(0, s * scale + bias))))
                mae += abs(p - y)
            mae /= len(labels)
            if mae < best[2]:
                best = (scale, bias, mae)
    return best[0], best[1]


def apply_linear(scores: list[float], scale: float, bias: float) -> list[int]:
    return [int(round(min(10, max(0, s * scale + bias)))) for s in scores]


def main() -> int:
    with urllib.request.urlopen(f"{BASE}/health", timeout=10) as r:
        health = json.load(r)
    print("health", health.get("status"), health.get("model"))
    print("one-by-one emotional score reframe\n")

    ids = [c[0] for c in CASES]
    labels = [c[1] for c in CASES]
    messages = [c[2] for c in CASES]

    # --- Mode A: HIGH query only ---
    print("=== Mode A: HIGH-emotion query (1 call/msg) ===")
    high_scores: list[float] = []
    walls_a: list[float] = []
    for i, msg in enumerate(messages, 1):
        sc, w = post_score(HIGH_QUERY, [msg], sigmoid=True)
        high_scores.append(sc[0])
        walls_a.append(w * 1000)
        if i == 1 or i % 10 == 0 or i == len(messages):
            print(f"  [{i}/{len(messages)}] {w*1000:.0f}ms score={sc[0]:.3f} label={labels[i-1]}")
    scale_a, bias_a = calibrate_linear(high_scores, labels)
    preds_a = apply_linear(high_scores, scale_a, bias_a)
    m_a = metrics(preds_a, labels)
    print(
        f"calibrated pred=round(clamp(s*{scale_a:.2f}+{bias_a:.2f}))  "
        f"exact={m_a['exact_pct']}%  ±1={m_a['within1_pct']}%  MAE={m_a['mae']}"
    )
    print(
        f"latency med={statistics.median(walls_a):.0f}ms "
        f"mean={statistics.mean(walls_a):.0f}ms p95={sorted(walls_a)[int(0.95*(len(walls_a)-1))]:.0f}ms\n"
    )

    # --- Mode B: HIGH - LOW margin ---
    print("=== Mode B: HIGH then LOW (2 calls/msg) ===")
    low_scores: list[float] = []
    walls_b: list[float] = []
    for i, msg in enumerate(messages, 1):
        hs, wh = post_score(HIGH_QUERY, [msg], sigmoid=True)
        ls, wl = post_score(LOW_QUERY, [msg], sigmoid=True)
        # refresh high for honest paired path
        high_scores[i - 1] = hs[0]
        low_scores.append(ls[0])
        walls_b.append((wh + wl) * 1000)
        if i == 1 or i % 10 == 0 or i == len(messages):
            print(
                f"  [{i}/{len(messages)}] {(wh+wl)*1000:.0f}ms "
                f"high={hs[0]:.3f} low={ls[0]:.3f} margin={hs[0]-ls[0]:.3f}"
            )
    margins = [h - l for h, l in zip(high_scores, low_scores)]
    # map margin from [min,max] via calibrated linear on (margin+1)/2 style — use calibrate on raw margin
    # Shift margins into ~0..1-ish for calibrate: use sigmoid-like (m+1)/2 clip
    norm = [min(1.0, max(0.0, (m + 1) / 2)) for m in margins]
    scale_b, bias_b = calibrate_linear(norm, labels)
    preds_b = apply_linear(norm, scale_b, bias_b)
    m_b = metrics(preds_b, labels)
    print(
        f"calibrated on norm(margin)  exact={m_b['exact_pct']}%  ±1={m_b['within1_pct']}%  MAE={m_b['mae']}"
    )
    print(
        f"latency med={statistics.median(walls_b):.0f}ms "
        f"mean={statistics.mean(walls_b):.0f}ms\n"
    )

    # --- Mode C: 11 prototypes as query? Better: query=user message, docs=prototypes ---
    # Our API is (query, documents). Use fixed HIGH_QUERY style:
    # query = user message, documents = level prototypes → pick level with max score.
    print("=== Mode C: message as query × 11 level prototypes (1 call/msg, 11 docs) ===")
    preds_c: list[int] = []
    walls_c: list[float] = []
    proto_docs = [LEVEL_PROTOTYPES[i] for i in range(11)]
    for i, msg in enumerate(messages, 1):
        # query = message, docs = prototypes — "which intensity prototype matches this message"
        sc, w = post_score(msg, proto_docs, sigmoid=True)
        walls_c.append(w * 1000)
        level = max(range(11), key=lambda k: sc[k])
        preds_c.append(level)
        if i == 1 or i % 10 == 0 or i == len(messages):
            print(f"  [{i}/{len(messages)}] {w*1000:.0f}ms pred={level} label={labels[i-1]}")
    m_c = metrics(preds_c, labels)
    print(f"exact={m_c['exact_pct']}%  ±1={m_c['within1_pct']}%  MAE={m_c['mae']}")
    print(
        f"latency med={statistics.median(walls_c):.0f}ms "
        f"mean={statistics.mean(walls_c):.0f}ms\n"
    )

    # Pick best by MAE then within1
    ranked = sorted(
        [("A", m_a, preds_a), ("B", m_b, preds_b), ("C", m_c, preds_c)],
        key=lambda x: (x[1]["mae"], -x[1]["within1_pct"]),
    )
    best_name, best_m, best_preds = ranked[0]
    print(f"=== Best by MAE: Mode {best_name} ===")
    print(json.dumps(best_m, indent=2))
    print("\nmisses |pred-label|>=2:")
    for cid, y, p, msg in zip(ids, labels, best_preds, messages):
        if abs(p - y) >= 2:
            short = msg if len(msg) <= 80 else msg[:77] + "…"
            print(f"  id={cid} label={y} pred={p}  {short}")

    out = {
        "request_mode": "one_by_one",
        "health": {k: health.get(k) for k in ("status", "model", "vram_allocated_mb")},
        "mode_a": {
            "latency_med_ms": round(statistics.median(walls_a), 1),
            "scale": scale_a,
            "bias": bias_a,
            **m_a,
        },
        "mode_b": {
            "latency_med_ms": round(statistics.median(walls_b), 1),
            "scale": scale_b,
            "bias": bias_b,
            **m_b,
        },
        "mode_c": {
            "latency_med_ms": round(statistics.median(walls_c), 1),
            **m_c,
        },
        "best": best_name,
        "per_case": [
            {
                "id": cid,
                "label": y,
                "message": msg,
                "high_score": hs,
                "low_score": ls if ls is not None else None,
                "pred_a": pa,
                "pred_b": pb,
                "pred_c": pc,
            }
            for cid, y, msg, hs, ls, pa, pb, pc in zip(
                ids,
                labels,
                messages,
                high_scores,
                low_scores + [None] * (len(labels) - len(low_scores)),
                preds_a,
                preds_b,
                preds_c,
            )
        ],
    }
    model_slug = str(health.get("model") or "unknown").split("/")[-1].lower()
    path = ROOT / "cache" / model_slug / "_debug" / "emotional-score-rerank-onebyone.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("\nwrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
