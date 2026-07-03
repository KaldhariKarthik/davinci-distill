# DaVinci Distillation — Run Guide

Co-resident distillation: Qwen2.5-VL-32B teacher -> Qwen2.5-VL-3B student.
Target skill: **reasoning over retrieved context**. Text-first.

---

## Before you spend a rupee

1. **Edit `seeds.py`** — this is the one file that's yours. Make the seeds match what
   DaVinci actually does. Add real slot fillers. More variety = better student.
2. **Deploy the pod** (RunPod RTX PRO 6000, 96GB): PyTorch 2.4 CUDA 12.4 template,
   SSH on, **Volume disk 150GB** (32B weights alone are ~65GB — 40GB will fail).
3. **Set a phone alarm.** The pod bills every minute. Guard it.

## Setup on the pod
```bash
pip install -r requirements.txt
huggingface-cli login          # if the Qwen weights need auth
```

## THE DRY RUN — do this first, it's your go/no-go gate (~₹150)
Prove everything loads and moves before committing the full run.
```bash
# tiny smoke test: 1 seed, 2 variations, 2 steps
python - <<'PY'
import config
config.VARIATIONS_PER_SEED = 2
import generate_data; generate_data.main()
PY
head -n 4 data/teacher_data.jsonl > data/eval.jsonl   # fake tiny eval
python distill.py    # let it run ~10 steps, confirm loss prints and drops, Ctrl-C
```
If loss prints and the numbers move: **the machine works.** Commit the real run.
If it OOMs: lower `MAX_SEQ_LEN` (2048 -> 1024) in `config.py`, retry. Still OOM:
drop `W_HIDDEN` to 0 (skips hidden-state buffers) as a fallback.

## THE REAL RUN
```bash
# 1) teacher generates training data (~2-3 hrs)
python generate_data.py

# 2) hold out an eval set, remove those lines from the training file
tail -n 200 data/teacher_data.jsonl > data/eval.jsonl
head -n -200 data/teacher_data.jsonl > data/train.tmp && mv data/train.tmp data/teacher_data.jsonl

# 3) baseline: score the RAW student before training (your floor)
python eval.py Qwen/Qwen2.5-VL-3B-Instruct

# 4) distill (the long pole, ~3-4 hrs)
python distill.py

# 5) score the distilled student — compare to the baseline floor
python eval.py
```

## After training — DO THIS BEFORE KILLING THE POD
The volume disk is **deleted when the pod is terminated.** Download your model first:
```bash
# from your LOCAL machine:
runpodctl receive <pod-id>:/workspace/davinci-distill/outputs/student_distilled ./
# or use the RunPod file browser / scp
```
Then stop the pod.

## Next stages (later runs)
- `prune.py` + heal (trim 3B -> ~2.5B if the Jetson needs it)
- `quantize.py` (4-bit for the 8GB device)
- add vision: swap text inputs for image+text using the same VL models

## The three OOM knobs (co-resident is tight on 96GB)
1. `MAX_SEQ_LEN` down (biggest lever)
2. `GRADIENT_CHECKPOINTING` stays True
3. `W_HIDDEN = 0` removes teacher hidden-state buffers (last resort — loses a signal)

## Money discipline
- Dry run before real run, always.
- Stop the pod the instant a stage ends.
- Download the model before terminating (volume disk dies with the pod).
