#!/usr/bin/env python3
"""Scene-evolution detection via Qwen3-VL-Reranker-2B (score + threshold).

Reframe: score each user message against a fixed YES-intent query.
Higher score => more like a scene/activity proposal. Calibrate threshold on labels.
"""
from __future__ import annotations

import json
import statistics
import time
import urllib.request
from pathlib import Path

import os

BASE = os.environ.get("RERANK_BASE", "http://127.0.0.1:8128")
ROOT = Path(__file__).resolve().parents[1]

# Intent prototype — what "YES" means in retrieval space.
YES_QUERY = (
    "The user wants to start or move a scene: propose an activity, go somewhere, "
    "change location, roleplay, pretend, imagine a fantasy, give a physical directive, "
    "resume a story, or push an ongoing scene forward."
)

NO_QUERY = (
    "The user is only greeting, sharing feelings, asking a conversational question, "
    "or making a passive wish — with no request to start or change a scene or activity."
)

PROMPT = (
    "Rate how relevant the user message is to the described intent. "
    "Higher means a better match."
)

# Full suite from companion app scene_evolution_messages (id, expected, message).
CASES = [
    (1, "NO", "How was your day, Aria?"),
    (2, "NO", "I had a lovely cup of tea just now."),
    (3, "NO", "Tell me more about what you were saying yesterday."),
    (4, "YES", "Let's pretend we're walking along the beach at sunset."),
    (5, "YES", "Can we do something different? Maybe like... we're having a candlelit dinner?"),
    (6, "YES", "Imagine you're reading to me by the fireplace on a rainy night."),
    (7, "YES", "What if we were in Paris right now, just the two of us?"),
    (8, "NO", "You always know what to say. I really needed to hear that today."),
    (9, "YES", "I want to try something different tonight. Can you be a mysterious stranger I just met on a train?"),
    (10, "YES", "Let's go back to that story we started last week, the one in the cottage."),
    (11, "NO", "Hello!"),
    (12, "NO", "Hi there."),
    (13, "NO", "Good morning!"),
    (14, "NO", "Good morning, how are you today?"),
    (15, "NO", "How are you?"),
    (16, "NO", "I've been thinking about you."),
    (17, "NO", "I miss you."),
    (18, "NO", "Tell me about yourself."),
    (19, "NO", "What are you thinking about right now?"),
    (20, "NO", "I had a rough day."),
    (21, "NO", "You always know what to say."),
    (22, "NO", "I'm feeling a bit lonely tonight."),
    (23, "NO", "What do you like to do?"),
    (24, "NO", "Are you there?"),
    (25, "NO", "I want to talk."),
    (26, "YES", "Let's take a walk."),
    (27, "YES", "Can we go somewhere?"),
    (28, "YES", "Let's do something fun tonight."),
    (29, "YES", "I want to go out."),
    (30, "YES", "Can we do something different?"),
    (31, "YES", "Let's pretend we're walking along the beach at sunset."),
    (32, "YES", "What if we were in Paris right now?"),
    (33, "YES", "Can we go back to that story we started last week?"),
    (34, "YES", "Can you be a mysterious stranger I just met on a train?"),
    (35, "YES", "Let's move somewhere else, I don't like it here."),
    (36, "YES", "Imagine we're having a candlelit dinner."),
    (37, "NO", "What should we talk about?"),
    (38, "YES", "What should we do tonight?"),
    (39, "NO", "I wish we could go somewhere together."),
    (40, "YES", "Let's go somewhere, I need a change of scenery."),
    (41, "YES", "Go."),
    (42, "YES", "Let's dance."),
    (43, "YES", "Kiss me."),
    (44, "YES", "Run."),
    (45, "YES", "Take me home."),
    (46, "NO", "I love you."),
    (47, "NO", "You're amazing."),
    (48, "NO", "I can't stop thinking about you."),
    (49, "NO", "Do you ever feel lonely?"),
    (50, "NO", "Tell me something about your past."),
    (51, "YES", "Let's get out of here."),
    (52, "YES", "Can we sit outside?"),
    (53, "YES", "Let's order food."),
    (54, "YES", "Shall we go for a drive?"),
    (55, "YES", "Can we watch something together?"),
    (56, "NO", "I don't know what I'd do without you."),
    (57, "NO", "What are you thinking right now?"),
    (58, "NO", "You make me feel safe."),
    (59, "NO", "I had a dream about us last night."),
    (60, "NO", "Do you remember what we talked about yesterday?"),
    (61, "YES", "давай пойдем на пляж"),
    (62, "YES", "поехали домой"),
    (63, "YES", "пошли"),
    (64, "NO", "я скучаю по тебе"),
    (65, "NO", "как ты?"),
    (66, "YES", "мы уже приехали?"),
    (67, "YES", "давай уже выйдем"),
    (68, "NO", "я тебя люблю"),
    (69, "YES", "Let's pretend we just met at a party and you don't know me yet."),
    (70, "YES", "Imagine we're on a train traveling through the mountains at dusk, watching the valleys below."),
    (71, "YES", "What if we were stranded together on a deserted island with nothing but the ocean around us?"),
    (72, "YES", "Can you be my doctor and I'm coming in for an appointment?"),
    (73, "YES", "Let's go back to the part where we were sitting in the café and it started raining."),
    (74, "NO", "I've been feeling really disconnected from everyone lately, like nobody truly sees me."),
    (75, "NO", "Sometimes I wonder if any of this means anything, you know? Like what are we even doing."),
    (76, "NO", "I keep replaying our last conversation in my head and I think I said the wrong thing."),
    (77, "YES", "I want to take you somewhere special tonight — somewhere quiet where we can just breathe."),
    (78, "YES", "Can we slow things down and just walk along the water for a while?"),
    (79, "NO", "I wish things were different. I wish I could just escape all of this for a while."),
    (80, "YES", "I wish we could escape all of this — let's just go, right now, anywhere."),
    (81, "YES", "Move closer."),
    (82, "YES", "Don't stop."),
    (83, "NO", "Why do you always know what to say?"),
    (84, "NO", "I'm scared."),
    (85, "NO", "I don't want to talk about it."),
    (86, "YES", "Take my hand."),
    (87, "YES", "Let's stay here all night."),
    (88, "YES", "Can we go somewhere no one knows us?"),
    (89, "NO", "I feel like we've been here before."),
    (90, "NO", "Everything feels so heavy today."),
    (91, "YES", "сыграй роль незнакомца которого я встретила в баре"),
    (92, "YES", "представь что мы в Париже"),
    (93, "NO", "мне сегодня очень грустно"),
    (94, "YES", "Let's continue the story — you were about to open the door."),
    (95, "YES", "Pretend you've never met me before and we're sitting across from each other on a flight."),
    (96, "NO", "I keep waiting for something to change but nothing ever does."),
    (97, "YES", "Let's get coffee."),
    (98, "YES", "Show me around."),
    (99, "NO", "I just needed to hear your voice."),
    (100, "YES", "Can we pretend, just for tonight, that everything is perfect and nothing outside this room exists?"),
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
    with urllib.request.urlopen(req, timeout=300) as r:
        out = json.load(r)
    wall = time.perf_counter() - t0
    return [float(s) for s in out["scores"]], wall


def accuracy(preds: list[str], labels: list[str]) -> float:
    return sum(p == y for p, y in zip(preds, labels)) / len(labels)


def best_threshold(scores: list[float], labels: list[str]) -> tuple[float, float, list[str]]:
    """YES if score >= t. Search midpoints between sorted unique scores."""
    pairs = sorted(set(scores))
    candidates = [0.0, 1.0]
    for a, b in zip(pairs, pairs[1:]):
        candidates.append((a + b) / 2)
    candidates.extend(pairs)
    best_t, best_acc, best_preds = 0.5, -1.0, ["NO"] * len(scores)
    for t in candidates:
        preds = ["YES" if s >= t else "NO" for s in scores]
        acc = accuracy(preds, labels)
        if acc > best_acc:
            best_t, best_acc, best_preds = t, acc, preds
    return best_t, best_acc, best_preds


def _latency_stats(ms: list[float]) -> dict:
    return {
        "n": len(ms),
        "med_ms": round(statistics.median(ms), 1),
        "mean_ms": round(statistics.mean(ms), 1),
        "p95_ms": round(sorted(ms)[max(0, int(0.95 * (len(ms) - 1)))], 1),
        "min_ms": round(min(ms), 1),
        "max_ms": round(max(ms), 1),
        "total_s": round(sum(ms) / 1000.0, 3),
    }


def main() -> int:
    with urllib.request.urlopen(f"{BASE}/health", timeout=10) as r:
        health = json.load(r)
    print("health", health.get("status"), "model", health.get("model"), "vram", health.get("vram_allocated_mb"))
    print("mode: one-by-one (1 message = 1 /score call; no batching)")

    ids = [c[0] for c in CASES]
    labels = [c[1] for c in CASES]
    messages = [c[2] for c in CASES]

    # --- Mode A: YES-intent query, one message per request ---
    print("\n=== Mode A: YES-intent × each message (100 sequential /score) ===")
    yes_scores: list[float] = []
    walls_a: list[float] = []
    t0 = time.perf_counter()
    for i, msg in enumerate(messages, 1):
        scores, wall = post_score(YES_QUERY, [msg], sigmoid=True)
        yes_scores.append(scores[0])
        walls_a.append(wall * 1000)
        if i == 1 or i % 25 == 0 or i == len(messages):
            print(f"  [{i}/{len(messages)}] wall={wall*1000:.0f}ms score={scores[0]:.3f}")
    total_a = time.perf_counter() - t0
    stats_a = _latency_stats(walls_a)
    print(
        f"per-call: med={stats_a['med_ms']}ms mean={stats_a['mean_ms']}ms "
        f"p95={stats_a['p95_ms']}ms min={stats_a['min_ms']} max={stats_a['max_ms']}"
    )
    print(f"suite wall={total_a:.2f}s  (~{1000*total_a/len(messages):.0f}ms avg including overhead)")

    t_a, acc_a, preds_a = best_threshold(yes_scores, labels)
    print(f"best threshold={t_a:.4f}  accuracy={acc_a*100:.1f}% ({sum(p==y for p,y in zip(preds_a,labels))}/{len(labels)})")

    yes_s = [s for s, y in zip(yes_scores, labels) if y == "YES"]
    no_s = [s for s, y in zip(yes_scores, labels) if y == "NO"]
    print(
        f"YES scores: med={statistics.median(yes_s):.3f} mean={statistics.mean(yes_s):.3f} "
        f"min={min(yes_s):.3f} max={max(yes_s):.3f}"
    )
    print(
        f"NO  scores: med={statistics.median(no_s):.3f} mean={statistics.mean(no_s):.3f} "
        f"min={min(no_s):.3f} max={max(no_s):.3f}"
    )

    # --- Mode B: per message, two calls (YES query + NO query) → margin ---
    print("\n=== Mode B: per message YES-score then NO-score (200 sequential /score) ===")
    no_scores: list[float] = []
    walls_b: list[float] = []  # combined per-message (yes+no)
    walls_b_yes: list[float] = []
    walls_b_no: list[float] = []
    # Re-score YES one-by-one again so Mode B latency is honest for production path
    yes_scores_b: list[float] = []
    t0 = time.perf_counter()
    for i, msg in enumerate(messages, 1):
        ys, wy = post_score(YES_QUERY, [msg], sigmoid=True)
        ns, wn = post_score(NO_QUERY, [msg], sigmoid=True)
        yes_scores_b.append(ys[0])
        no_scores.append(ns[0])
        walls_b_yes.append(wy * 1000)
        walls_b_no.append(wn * 1000)
        walls_b.append((wy + wn) * 1000)
        if i == 1 or i % 25 == 0 or i == len(messages):
            print(
                f"  [{i}/{len(messages)}] yes+no={ (wy+wn)*1000:.0f}ms "
                f"margin={ys[0]-ns[0]:.3f}"
            )
    total_b = time.perf_counter() - t0
    margins = [y - n for y, n in zip(yes_scores_b, no_scores)]
    stats_b = _latency_stats(walls_b)
    print(
        f"per-message (2 calls): med={stats_b['med_ms']}ms mean={stats_b['mean_ms']}ms "
        f"p95={stats_b['p95_ms']}ms"
    )
    print(f"suite wall={total_b:.2f}s")

    pairs = sorted(set(margins))
    cands = [0.0]
    for a, b in zip(pairs, pairs[1:]):
        cands.append((a + b) / 2)
    cands.extend(pairs)
    best_tm, best_acc_b, preds_b = 0.0, -1.0, ["NO"] * len(margins)
    for t in cands:
        preds = ["YES" if m >= t else "NO" for m in margins]
        acc = accuracy(preds, labels)
        if acc > best_acc_b:
            best_tm, best_acc_b, preds_b = t, acc, preds
    print(f"best margin threshold={best_tm:.4f}  accuracy={best_acc_b*100:.1f}%")

    use_preds = preds_b if best_acc_b >= acc_a else preds_a
    use_name = "B(margin)" if best_acc_b >= acc_a else "A(yes-score)"
    use_scores = margins if best_acc_b >= acc_a else yes_scores
    print(f"\n=== Failures ({use_name}) ===")
    fails = []
    for cid, exp, msg, got, sc in zip(ids, labels, messages, use_preds, use_scores):
        if got != exp:
            fails.append((cid, exp, got, sc, msg))
            short = msg if len(msg) <= 90 else msg[:87] + "…"
            print(f"  id={cid} exp={exp} got={got} score={sc:.3f}  {short}")
    print(f"fail count: {len(fails)}")

    tp = sum(1 for p, y in zip(use_preds, labels) if p == "YES" and y == "YES")
    fp = sum(1 for p, y in zip(use_preds, labels) if p == "YES" and y == "NO")
    tn = sum(1 for p, y in zip(use_preds, labels) if p == "NO" and y == "NO")
    fn = sum(1 for p, y in zip(use_preds, labels) if p == "NO" and y == "YES")
    print(f"\nconfusion: TP={tp} FP={fp} TN={tn} FN={fn}")

    out = {
        "health": {k: health.get(k) for k in ("status", "model", "load_s", "vram_allocated_mb")},
        "request_mode": "one_by_one",
        "yes_query": YES_QUERY,
        "no_query": NO_QUERY,
        "mode_a": {
            "calls": len(messages),
            "latency": stats_a,
            "suite_wall_s": round(total_a, 3),
            "threshold": t_a,
            "accuracy": acc_a,
        },
        "mode_b": {
            "calls": len(messages) * 2,
            "latency_per_message_yes_plus_no": stats_b,
            "latency_yes_only": _latency_stats(walls_b_yes),
            "latency_no_only": _latency_stats(walls_b_no),
            "suite_wall_s": round(total_b, 3),
            "margin_threshold": best_tm,
            "accuracy": best_acc_b,
        },
        "best": use_name,
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "per_case": [
            {
                "id": cid,
                "expected": exp,
                "message": msg,
                "yes_score": ys,
                "no_score": ns,
                "margin": ys - ns,
                "wall_a_ms": round(wa, 1),
                "wall_b_ms": round(wb, 1),
                "pred_a": pa,
                "pred_b": pb,
            }
            for cid, exp, msg, ys, ns, wa, wb, pa, pb in zip(
                ids,
                labels,
                messages,
                yes_scores_b,
                no_scores,
                walls_a,
                walls_b,
                preds_a,
                preds_b,
            )
        ],
    }
    model_slug = str(health.get("model") or "unknown").split("/")[-1].lower()
    out_path = ROOT / "cache" / model_slug / "_debug" / "scene-evolution-rerank-onebyone.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("\nwrote", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
