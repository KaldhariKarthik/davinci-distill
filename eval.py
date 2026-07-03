"""
eval.py  —  score the student on held-out reasoning-over-context examples.

Two signals:
  1. eval loss (CE) on held-out data — a cheap proxy, tracks improvement across stages.
  2. sample generations — prints the student's actual answers so you can READ quality.

For a real "% of teacher" number, add LLM-as-judge: have the 32B teacher rate the
student's answer vs the reference (1-10) and average. That needs the teacher loaded;
run it as a separate pass to keep this script light. Stub marked below.
"""
import json, sys
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import config

MODEL = sys.argv[1] if len(sys.argv) > 1 else config.STUDENT_OUT


def main():
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True)
    model.eval()

    rows = [json.loads(l) for l in open(config.EVAL_DATA_PATH)]
    print(f"[eval] {len(rows)} held-out examples\n")

    # --- sample generations (read these) ------------------------------------
    for r in rows[:5]:
        ctx = r.get("context", "")
        user = f"{r['request']}\n\nRetrieved context:\n{ctx}" if ctx else r["request"]
        ids = tok.apply_chat_template([{"role": "user", "content": user}],
                                      add_generation_prompt=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=256, do_sample=False)
        ans = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
        print(f"REQUEST : {r['request']}")
        print(f"STUDENT : {ans}")
        print(f"REFERENCE: {r['answer'][:200]}...")
        print("-" * 70)

    # --- eval loss proxy -----------------------------------------------------
    total, n = 0.0, 0
    for r in rows:
        ctx = r.get("context", "")
        user = f"{r['request']}\n\nRetrieved context:\n{ctx}" if ctx else r["request"]
        p = tok.apply_chat_template([{"role": "user", "content": user}],
                                    add_generation_prompt=True, return_tensors="pt")[0]
        a = tok(r["answer"], add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        ids = torch.cat([p, a])[:config.MAX_SEQ_LEN].unsqueeze(0).to(model.device)
        labels = ids.clone(); labels[0, :len(p)] = -100
        with torch.no_grad():
            loss = model(ids, labels=labels).loss
        total += loss.item(); n += 1
    print(f"\n[eval] mean eval loss: {total/n:.4f}  (lower = better; compare across stages)")


if __name__ == "__main__":
    main()
