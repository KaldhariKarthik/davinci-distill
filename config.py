"""
Central configuration for DaVinci distillation.
Everything you'd tune lives here so you never edit the training code on a live pod.

Setup: Qwen2.5-VL-32B teacher + Qwen2.5-VL-3B student, CO-RESIDENT, full-weight,
8-bit Adam. Target skill = reasoning over retrieved context. Text-first
(we use the VL models but feed text only; vision is added in a later run).
"""

# ---- models (same family so hidden-state distillation is clean) --------------
TEACHER_MODEL = "Qwen/Qwen2.5-VL-32B-Instruct"
STUDENT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"

# ---- co-resident memory discipline (96GB RTX PRO 6000 is TIGHT) --------------
# These three knobs are what keep you under 96GB. If you OOM, lower them first.
MAX_SEQ_LEN        = 2048     # cap context length. Lower = less activation memory.
BATCH_SIZE         = 1        # per-step batch. Keep at 1 co-resident; use grad accum.
GRAD_ACCUM_STEPS   = 8        # effective batch = BATCH_SIZE * GRAD_ACCUM_STEPS
GRADIENT_CHECKPOINTING = True # trades compute for big activation-memory savings. Keep ON.
USE_8BIT_ADAM      = True     # NON-NEGOTIABLE co-resident. Regular Adam OOMs.

# ---- distillation loss weights ----------------------------------------------
# Total loss = CE*w_ce + KL*w_kl + hidden_state*w_hidden
W_CE     = 1.0    # cross-entropy on teacher's generated answer tokens
W_KL     = 1.0    # soft-logit (KL) matching — the core distillation signal
W_HIDDEN = 0.02   # hidden-state matching (needs the projection layer)
KL_TEMPERATURE = 2.0          # softens distributions for richer signal

# which student/teacher layers to align for hidden-state matching.
# 3B and 32B have different depths, so we map evenly-spaced layers.
HIDDEN_MATCH_LAYERS = "even"  # 'even' = spread matches across depth

# ---- training ----------------------------------------------------------------
LEARNING_RATE = 2e-5
NUM_EPOCHS    = 2
WARMUP_STEPS  = 20
SAVE_EVERY    = 100           # checkpoint frequency (steps)

# ---- data --------------------------------------------------------------------
GENERATED_DATA_PATH = "data/teacher_data.jsonl"   # written by generate_data.py
EVAL_DATA_PATH      = "data/eval.jsonl"           # held-out, for scoring
VARIATIONS_PER_SEED = 30    # how many variations to expand each seed into

# ---- paths -------------------------------------------------------------------
STUDENT_OUT   = "outputs/student_distilled"
CHECKPOINT_DIR = "checkpoints"

# ---- generation (teacher answering during data gen) --------------------------
GEN_MAX_NEW_TOKENS = 512
GEN_TEMPERATURE    = 0.7
