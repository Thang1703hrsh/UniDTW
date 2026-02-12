# DTW + DistiLLM Training Optimization

## Problem Diagnosis

### Symptoms
- RougeL score starts at **21.8068** (threshold=0.0)
- RougeL drops to **~6.0** when threshold increases (0.1-0.8)
- Total loss explodes from ~8 to ~47 over training
- Exact match score drops to 0.0%

### Root Cause
The **DistiLLM adaptive threshold mechanism** was gradually replacing high-quality teacher data with low-quality student-generated samples:

1. **threshold=0.0**: Model uses original teacher-quality training data → Good performance
2. **threshold>0.0**: Model starts sampling from student generations stored in ReplayBuffer → Poor performance
3. With `--loss-eps 0.1`, the threshold increases aggressively, filling the replay buffer with garbage

The student model at early training stages generates very poor quality text, and the adaptive mechanism creates a vicious cycle:
- Student generates poor samples → Added to buffer → Student trains on poor samples → Generates worse samples

## Solution Applied

### Key Changes in `train_dtw.sh`

#### 1. Disabled Student Generation (Lines 119-134)
```bash
# BEFORE:
OPTS+=" --type adaptive-srkl-dtw"
OPTS+=" --student-gen"
OPTS+=" --init-threshold 0.0"
OPTS+=" --loss-eps 0.1"

# AFTER:
OPTS+=" --type srkl-dtw"  # Removed "adaptive" prefix
# Commented out all student-gen related flags
```

**Rationale**: DTW already provides powerful sequence-level alignment between student and teacher hidden states. The student generation mechanism is redundant and counterproductive at early training stages.

#### 2. Added Training Warmup (Line 83)
```bash
OPTS+=" --warmup-iters 500"  # Was 0
```

**Rationale**: Gradual learning rate warmup prevents early training instability.

#### 3. Enabled Periodic Evaluation (Lines 98-99)
```bash
OPTS+=" --save-interval 2"    # Was -1 (never save)
OPTS+=" --eval-interval 1"    # Was -1 (never eval)
```

**Rationale**: Monitor training progress and catch issues early.

## Expected Results

### Before Optimization
- RougeL: ~6.0-6.5 (poor)
- Exact Match: 0.0%
- Loss: Exploding to 40+

### After Optimization (Expected)
- RougeL: ~21-22 (good) - consistent with threshold=0.0 performance
- Exact Match: ~2.8%
- Loss: Stable, decreasing over time

## When to Use Student Generation

If you want to use DistiLLM's student generation in the future, only enable it when:

1. **Student model is already well-trained** (e.g., after 10+ epochs of standard distillation)
2. **Use strict quality filters**:
   ```bash
   OPTS+=" --init-threshold 0.7"      # High threshold
   OPTS+=" --loss-eps 0.01"           # Very slow increase
   OPTS+=" --gen-num-beams 4"         # Beam search for better quality
   OPTS+=" --gen-top-p 0.95"          # Nucleus sampling
   OPTS+=" --temperature 0.7"         # Lower temperature
   ```
3. **Monitor RougeL closely** - if it drops, disable immediately

## Architecture Notes

### DTW Loss Component
The DTW loss aligns student and teacher hidden states at the sequence level:
- **Cost matrix**: Cosine distance between all token pairs
- **Soft-DTW**: Differentiable approximation with gamma=0.1
- **Window**: Band constraint of 32 tokens for efficiency
- **Unitization**: Pools tokens by semantic units before alignment

This is already a very powerful distillation mechanism and doesn't need augmentation from DistiLLM's replay buffer in early training.

### DistiLLM Original Purpose
DistiLLM was designed for scenarios where:
- You don't have parallel teacher-student data
- You want to distill from teacher generations
- The student needs to learn the teacher's generation distribution

In this setup, you already have aligned hidden states via DTW, making the student generation mechanism unnecessary.

## Monitoring Training

Watch for these healthy signs:
- ✅ `ds_loss` (distillation loss) decreasing: 1.8 → 1.5 → 1.2
- ✅ `dtw_loss` stable or slightly decreasing: ~0.52
- ✅ RougeL stable or increasing: ~21-22
- ✅ Total loss decreasing or stable

Warning signs:
- ❌ Total loss increasing significantly
- ❌ RougeL dropping below 15
- ❌ Exact match dropping to 0%

## File Modified
- `scripts/gpt2/distillm/train_dtw.sh`

## Date
2026-02-11
