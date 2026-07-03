"""
distill.py  —  CO-RESIDENT distillation training loop.

Both models resident on one 96GB card:
  - TEACHER (32B): frozen, eval, no grad. Provides LIVE logits + hidden states.
  - STUDENT (3B):  trainable, full-weight, 8-bit Adam.

Loss = W_CE * CE(answer)  +  W_KL * KL(student||teacher logits)  +  W_HIDDEN * hidden_match

The hidden-state match needs a PROJECTION layer because the student's hidden width
(~2048) != teacher's (~5120). We learn a Linear per matched layer to bridge them.

Memory discipline (this is why it fits 96GB): gradient checkpointing ON, batch 1 +
grad accum, 8-bit Adam, capped seq len. If you OOM, lower MAX_SEQ_LEN in config first.
"""
import json, math, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, get_cosine_schedule_with_warmup
import bitsandbytes as bnb
import config

random.seed(0); torch.manual_seed(0)


# ------------------------------------------------------------------ dataset ---
class ReasoningDataset(Dataset):
    """Each item: input = request + retrieved context; target = grounded answer."""
    def __init__(self, path, tok):
        self.tok = tok
        self.rows = [json.loads(l) for l in open(path)]

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        # the input the student sees: the request plus the retrieved context
        ctx = r.get("context", "")
        user = (f"{r['request']}\n\nRetrieved context:\n{ctx}" if ctx else r["request"])
        answer = r["answer"]

        prompt_text = self.tok.apply_chat_template(
            [{"role": "user", "content": user}],
            add_generation_prompt=True, tokenize=False)
        prompt_ids = self.tok(prompt_text, add_special_tokens=False,
                              return_tensors="pt")["input_ids"][0]
        answer_ids = self.tok(answer, add_special_tokens=False,
                              return_tensors="pt")["input_ids"][0]

        input_ids = torch.cat([prompt_ids, answer_ids])[:config.MAX_SEQ_LEN]
        # labels: -100 on the prompt (don't train on it), real ids on the answer
        labels = input_ids.clone()
        labels[:len(prompt_ids)] = -100
        return input_ids, labels


def collate(batch, pad_id):
    ids, labels = zip(*batch)
    maxlen = max(x.size(0) for x in ids)
    def pad(seqs, val):
        return torch.stack([F.pad(s, (0, maxlen - s.size(0)), value=val) for s in seqs])
    input_ids = pad(ids, pad_id)
    labels    = pad(labels, -100)
    attn      = (input_ids != pad_id).long()
    return input_ids, labels, attn


# ------------------------------------------------------- hidden-state bridge ---
def matched_layer_pairs(n_student, n_teacher, n_matches=4):
    """Evenly spaced (student_layer, teacher_layer) pairs to align."""
    pairs = []
    for k in range(1, n_matches + 1):
        s = round(k * n_student / (n_matches + 1))
        t = round(k * n_teacher / (n_matches + 1))
        pairs.append((s, t))
    return pairs


def main():
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(config.STUDENT_MODEL, trust_remote_code=True)
    pad_id = tok.pad_token_id or tok.eos_token_id

    print("[distill] loading teacher (32B, frozen)...")
    teacher = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.TEACHER_MODEL, dtype=torch.bfloat16,
        device_map={"": 0}, trust_remote_code=True)
    teacher.eval()
    for p in teacher.parameters(): p.requires_grad_(False)

    print("[distill] loading student (3B, trainable)...")
    student = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.STUDENT_MODEL, dtype=torch.bfloat16,
        device_map={"": 0}, trust_remote_code=True)
    if config.GRADIENT_CHECKPOINTING:
        student.gradient_checkpointing_enable()
    student.train()

    # width bridge for hidden-state matching.
    # VL configs sometimes nest text dims under .text_config — handle both.
    def cfg_get(cfg, attr):
        if hasattr(cfg, attr):
            return getattr(cfg, attr)
        return getattr(cfg.text_config, attr)

    s_hidden = cfg_get(student.config, "hidden_size")
    t_hidden = cfg_get(teacher.config, "hidden_size")
    pairs = matched_layer_pairs(cfg_get(student.config, "num_hidden_layers"),
                                cfg_get(teacher.config, "num_hidden_layers"))
    projections = nn.ModuleList([
        nn.Linear(s_hidden, t_hidden, bias=False).to(dev).to(torch.bfloat16)
        for _ in pairs])
    print(f"[distill] hidden match layers (student->teacher): {pairs}")

    # data
    ds = ReasoningDataset(config.GENERATED_DATA_PATH, tok)
    dl = DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=True,
                    collate_fn=lambda b: collate(b, pad_id))
    print(f"[distill] {len(ds)} training examples")

    # optimizer: 8-bit Adam over student + projection params (co-resident memory saver)
    params = list(student.parameters()) + list(projections.parameters())
    opt = bnb.optim.Adam8bit(params, lr=config.LEARNING_RATE) if config.USE_8BIT_ADAM \
          else torch.optim.AdamW(params, lr=config.LEARNING_RATE)
    total_steps = (len(dl) // config.GRAD_ACCUM_STEPS) * config.NUM_EPOCHS
    sched = get_cosine_schedule_with_warmup(opt, config.WARMUP_STEPS, total_steps)

    step = 0
    best_eval = float("inf"); best_epoch = -1

    def eval_heldout():
        """Mean CE loss on the held-out eval set — the number that matters."""
        student.eval()
        rows = [__import__("json").loads(l) for l in open(config.EVAL_DATA_PATH)]
        tot, n = 0.0, 0
        with torch.no_grad():
            for r in rows:
                ctx = r.get("context", "")
                u = (f"{r['request']}\n\nRetrieved context:\n{ctx}" if ctx else r["request"])
                ptxt = tok.apply_chat_template([{"role": "user", "content": u}],
                                               add_generation_prompt=True, tokenize=False)
                p = tok(ptxt, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
                a = tok(r["answer"], add_special_tokens=False,
                        return_tensors="pt")["input_ids"][0]
                ids = torch.cat([p, a])[:config.MAX_SEQ_LEN].unsqueeze(0).to(dev)
                lab = ids.clone(); lab[0, :len(p)] = -100
                tot += student(ids, labels=lab).loss.item(); n += 1
        student.train()
        return tot / max(n, 1)

    for epoch in range(config.NUM_EPOCHS):
        for bi, (input_ids, labels, attn) in enumerate(dl):
            input_ids, labels, attn = input_ids.to(dev), labels.to(dev), attn.to(dev)

            # teacher forward — live signal, no grad
            with torch.no_grad():
                t_out = teacher(input_ids, attention_mask=attn, output_hidden_states=True)

            # student forward
            s_out = student(input_ids, attention_mask=attn, output_hidden_states=True)

            # 1) CE on the answer tokens
            ce = F.cross_entropy(
                s_out.logits[:, :-1].reshape(-1, s_out.logits.size(-1)),
                labels[:, 1:].reshape(-1), ignore_index=-100)

            # 2) KL between softened logits (only where we have real labels)
            # teacher and student can have slightly different vocab sizes
            # (32B has a few extra special tokens) — align to the shared minimum.
            T = config.KL_TEMPERATURE
            mask = (labels[:, 1:] != -100)
            V = min(s_out.logits.size(-1), t_out.logits.size(-1))
            s_logits = s_out.logits[:, :-1, :V]
            t_logits = t_out.logits[:, :-1, :V]
            s_logp = F.log_softmax(s_logits / T, dim=-1)
            t_p    = F.softmax(t_logits / T, dim=-1)
            kl = (F.kl_div(s_logp, t_p, reduction="none").sum(-1) * mask)
            kl = kl.sum() / mask.sum().clamp(min=1) * (T * T)

            # 3) hidden-state matching through projections
            hid = torch.tensor(0.0, device=dev)
            for proj, (si, ti) in zip(projections, pairs):
                s_h = proj(s_out.hidden_states[si])
                t_h = t_out.hidden_states[ti].detach()
                hid = hid + F.mse_loss(s_h.float(), t_h.float())
            hid = hid / len(pairs)

            loss = (config.W_CE * ce + config.W_KL * kl + config.W_HIDDEN * hid)
            (loss / config.GRAD_ACCUM_STEPS).backward()

            if (bi + 1) % config.GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sched.step(); opt.zero_grad()
                step += 1
                if step % 5 == 0:
                    print(f"e{epoch} step{step} | loss {loss.item():.3f} "
                          f"(ce {ce.item():.3f} kl {kl.item():.3f} hid {hid.item():.3f})")

        # --- end of epoch: eval on held-out set, save this epoch's checkpoint ---
        ev = eval_heldout()
        ckpt = f"{config.CHECKPOINT_DIR}/epoch{epoch}"
        student.save_pretrained(ckpt); tok.save_pretrained(ckpt)
        flag = ""
        if ev < best_eval:
            best_eval, best_epoch = ev, epoch
            student.save_pretrained(config.STUDENT_OUT); tok.save_pretrained(config.STUDENT_OUT)
            flag = "  <-- new best, saved to outputs"
        print(f"[eval] epoch {epoch}: held-out loss {ev:.4f}{flag}")

    print(f"[distill] done. best epoch {best_epoch} @ {best_eval:.4f} -> {config.STUDENT_OUT}")


if __name__ == "__main__":
    main()