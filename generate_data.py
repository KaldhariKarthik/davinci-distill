"""
generate_data.py  —  PHASE A: teacher generates the training data.

For each seed (expanded with slot fillers) it asks the 32B teacher to:
  1. produce a CLEAN, generic retrieval query (privacy: no personal data in it)
  2. produce a realistic RETRIEVED CONTEXT bundle (mixed quality, some junk)
  3. produce the GROUNDED ANSWER by reasoning over that context

The saved {request, query, context, answer} tuples are the training targets.
The student later learns to reproduce steps 1-3 (reasoning over retrieved context).

Run this FIRST, on the pod. It writes data/teacher_data.jsonl.
Teacher is loaded here; it's loaded AGAIN in distill.py for live logit/hidden signal.
That double-load is expected — generation is one-time, training reuses the saved data.
"""
import json, random, itertools
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoTokenizer
import config
from seeds import SEEDS, SLOT_FILLERS

random.seed(0)


def fill_slots(template: str) -> str:
    """Replace every {slot} with a random filler."""
    out = template
    for slot, opts in SLOT_FILLERS.items():
        if "{" + slot + "}" in out:
            out = out.replace("{" + slot + "}", random.choice(opts))
    return out


def build_gen_prompt(request: str, source: str) -> str:
    """Ask the teacher to produce query + context + grounded answer as JSON."""
    return f"""You are generating training data for a small on-device assistant whose
skill is REASONING OVER RETRIEVED CONTEXT. Given a user request, produce a JSON object
with exactly these keys:

- "query": a clean, generic search query to retrieve info for this request. It must
  contain NO personal names or private details (privacy: personal data never leaves the
  device). If the request needs no external info, use "".
- "context": a realistic bundle of 3-5 retrieved snippets as one string, separated by
  "---". Make it REALISTIC: include 1-2 slightly irrelevant or low-quality snippets so
  the assistant must filter. For personal 'files'/'calendar' sources, write plausible
  note/calendar snippets instead of web results.
- "answer": the ideal assistant response, produced by reasoning ONLY over the context
  above. Ground every claim in the context. Be concise and spoken-friendly.

User request: "{request}"
Knowledge source: {source}

Respond with ONLY the JSON object, no preamble."""


def main():
    print("[gen] loading teacher (32B) — this takes a few minutes...")
    tok = AutoTokenizer.from_pretrained(config.TEACHER_MODEL, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.TEACHER_MODEL, dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True)
    model.eval()

    # expand seeds into (request, source, skill) examples
    examples = []
    for seed in SEEDS:
        for _ in range(config.VARIATIONS_PER_SEED):
            examples.append((fill_slots(seed["request"]), seed["source"], seed["skill"]))
    random.shuffle(examples)
    print(f"[gen] expanded {len(SEEDS)} seeds -> {len(examples)} examples")

    written = 0
    with open(config.GENERATED_DATA_PATH, "w") as f:
        for i, (request, source, skill) in enumerate(examples):
            prompt = build_gen_prompt(request, source)
            msgs = [{"role": "user", "content": prompt}]
            text_in = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                              tokenize=False)
            inputs = tok(text_in, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=config.GEN_MAX_NEW_TOKENS,
                                     temperature=config.GEN_TEMPERATURE, do_sample=True)
            text = tok.decode(out[0][inputs["input_ids"].shape[1]:],
                              skip_special_tokens=True).strip()

            # parse the teacher's JSON; skip malformed ones rather than crash
            try:
                text = text[text.index("{"): text.rindex("}") + 1]
                rec = json.loads(text)
                rec.update({"request": request, "source": source, "skill": skill})
                f.write(json.dumps(rec) + "\n")
                written += 1
            except Exception:
                pass  # teacher occasionally emits bad JSON; just drop it

            if (i + 1) % 25 == 0:
                print(f"[gen] {i+1}/{len(examples)} processed, {written} saved")

    print(f"[gen] done. {written} examples -> {config.GENERATED_DATA_PATH}")
    print("[gen] hold out ~200 lines as data/eval.jsonl before training "
          "(tail -n 200 ... > eval.jsonl; then remove them from the train file).")


if __name__ == "__main__":
    main()