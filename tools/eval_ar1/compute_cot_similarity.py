#!/usr/bin/env python3
"""Score similarity between gt_cot and gen_cot in CoC evaluation JSONL files.

Reads a *_coc_eval.jsonl (output of eval_ckpt_coc.py), sends each gt_cot / gen_cot
pair to a local vLLM for similarity scoring, and writes a new *_score.jsonl with an
additional ``sim_score`` field per entry.

Usage:
    python tools/eval_ar1/compute_cot_similarity.py \
        --input outputs/ckpts_coc_eval.jsonl

    # Custom vLLM endpoint / model:
    python tools/eval_ar1/compute_cot_similarity.py \
        --input outputs/output_coc_stage1_2b_coc_eval.jsonl \
        --base-url http://127.0.0.1:8080/v1 \
        --model ckpts/Qwen3.6-27B-FP8

    # Dry run (print entries, no API calls):
    python tools/eval_ar1/compute_cot_similarity.py \
        --input outputs/ckpts_coc_eval.jsonl --dry-run
"""

import argparse
import asyncio
import json
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

MAX_CONCURRENCY = 128
MAX_RETRIES = 3

# ---------------------------------------------------------------------------
# Prompt — adapted for comparing generated CoC vs ground-truth CoC
# ---------------------------------------------------------------------------

SIMILARITY_PROMPT = """You are an expert at evaluating the semantic alignment between two driving-scene chain-of-thought (CoC) sentences.

Both sentences follow the pattern: [action] due to / to / because of [reason].

Your task: compare the **underlying driving concern** in each sentence — specifically whether the actions and the reasons refer to the same driving situation, hazard, or environmental factor.

Score on a 0–1 scale with four tiers:

Tier 1 (0.8–1.0): Nearly identical meaning.
Both the action and the reason point to the same driving concern, even if worded differently.
Example:
  A: "Nudge to the left due to a stopped vehicle blocking the lane ahead"
  B: "Nudge to the left to clear the stopped vehicle blocking the right side of our lane"
  Score: 0.9

Tier 2 (0.6–0.8): Same general action, overlapping reason.
The actions are the same or very similar, and the reasons share a clear common factor but differ in specificity or detail.
Example:
  A: "Nudge to the left due to a stopped vehicle blocking the lane ahead"
  B: "Change lanes to avoid an obstacle in the road ahead"
  Score: 0.7

Tier 3 (0.4–0.6): Inferred overlap after reasoning.
The explicit elements differ, but a reasonable person would infer they belong to the same general scenario.
Example:
  A: "Nudge to the left due to a stopped vehicle blocking the lane ahead"
  B: "Steer left to go around road work cones blocking the right side"
  Score: 0.5

Tier 4 (0.0–0.4): Little to no similarity.
The actions and/or reasons refer to completely different situations.
Example:
  A: "Nudge to the left due to a stopped vehicle blocking the lane ahead"
  B: "Keep lane since the lane is clear ahead"
  Score: 0.1

--- INPUT ---
Sentence A: {gt_cot}
Sentence B: {gen_cot}

Reply with ONLY a JSON object with a single key "score" and the similarity value as a float.
Example: {{"score": 0.75}}"""


# ---------------------------------------------------------------------------
# VLM call
# ---------------------------------------------------------------------------

def parse_vlm_json_response(text: str) -> dict:
    text = text.strip()
    m = re.search(r'```(?:json)?\s*(.*?)```', text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    return json.loads(text)


async def call_vlm(prompt: str, client: AsyncOpenAI, model: str,
                   enable_thinking: bool) -> dict:
    """Single VLM API call with retry."""
    extra_body = {
        "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    if enable_thinking:
        extra_body["thinking_token_budget"] = 4000
    for attempt in range(MAX_RETRIES):
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=8192,
                temperature=0.6,
                top_p=0.95,
                presence_penalty=1.5,
                extra_body=extra_body,
            )
            result_text = response.choices[0].message.content.strip()
            return parse_vlm_json_response(result_text)
        except json.JSONDecodeError:
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(1 * (attempt + 1))
            else:
                raise
        except Exception:
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(2 * (attempt + 1))
            else:
                raise


# ---------------------------------------------------------------------------
# Entry-level scoring
# ---------------------------------------------------------------------------

async def score_entry(
    entry: dict,
    idx: int,
    client: AsyncOpenAI,
    model: str,
    enable_thinking: bool,
    semaphore: asyncio.Semaphore,
) -> dict:
    gt_cot = entry.get("gt_cot", "")
    gen_cot = entry.get("gen_cot", "")
    prompt = SIMILARITY_PROMPT.format(gt_cot=gt_cot, gen_cot=gen_cot)

    async with semaphore:
        try:
            result = await call_vlm(prompt, client, model, enable_thinking)
            entry["sim_score"] = result.get("score", None)
        except Exception as e:
            print(f"  [FAIL] entry {idx} (clip={entry.get('clip_id','?')} "
                  f"ts={entry.get('ts','?')}): {e}", flush=True)
            entry["sim_score"] = None
            entry["sim_score_error"] = str(e)
    return entry


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main_async(args):
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return

    # Output: same directory, insert _score before .jsonl
    output_path = input_path.parent / input_path.name.replace(".jsonl", "_score.jsonl")

    with open(input_path, "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]

    print(f"Input:  {input_path} ({len(entries)} entries)")
    print(f"Output: {output_path}")
    print(f"Model:  {args.model}")
    print(f"Max concurrency: {args.max_concurrency}")

    if args.dry_run:
        for idx, entry in enumerate(entries):
            gt = entry.get("gt_cot", "")[:80]
            gen = entry.get("gen_cot", "")[:80]
            print(f"  [{idx}] gt:  \"{gt}...\"")
            print(f"        gen: \"{gen}...\"")
        return

    client = AsyncOpenAI(api_key=args.api_key, base_url=args.base_url)
    semaphore = asyncio.Semaphore(args.max_concurrency)

    coros = [
        score_entry(entry, idx, client, args.model, args.enable_thinking, semaphore)
        for idx, entry in enumerate(entries)
    ]
    scored_entries = await tqdm_asyncio.gather(*coros, desc="Scoring CoT similarity")

    with open(output_path, "w", encoding="utf-8") as f:
        for entry in scored_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    ok = sum(1 for e in scored_entries if e.get("sim_score") is not None)
    fail = sum(1 for e in scored_entries if e.get("sim_score_error"))
    avg_score = (
        sum(e["sim_score"] for e in scored_entries if e.get("sim_score") is not None)
        / ok
        if ok > 0
        else 0
    )
    print(f"\nDone. {ok} scored, {fail} failed, avg sim_score={avg_score:.4f}")
    print(f"Written to {output_path}")


def main():
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    parser = argparse.ArgumentParser(
        description="Score gt_cot vs gen_cot similarity in CoC eval JSONL"
    )
    parser.add_argument("--input", type=str, required=True, help="Input JSONL path")
    parser.add_argument("--api-key", type=str, default="EMPTY")
    parser.add_argument("--base-url", type=str, default="http://0.0.0.0:8080/v1")
    parser.add_argument("--model", type=str, default="ckpts/Qwen3.6-27B-int4-AutoRound")
    parser.add_argument("--max-concurrency", type=int, default=MAX_CONCURRENCY)
    parser.add_argument("--no-thinking", action="store_true",
                        help="Disable thinking mode for VLM")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.enable_thinking = not args.no_thinking

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
