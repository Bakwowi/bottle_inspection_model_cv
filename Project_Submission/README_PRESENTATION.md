# 10-Minute Presentation Guide — Krones Bottle-Base Inspection

Built to match the grading rubric (50 pts):

| Section | Max | Suggested time |
|---|---|---|
| Exploratory data analysis | 5 | ~1.5 min |
| Methodology / background | 15 | ~3.5 min |
| Progress & competition results | 10 | ~2.5 min |
| Lessons learned | 5 | ~1.5 min |
| Q&A | 15 | ~ (live) |

The single thread that ties the whole talk together — say it early, return to it at the end:

> **The per-defect-category labels were the key the whole time.** A plain GOOD/FAULTY classifier
> throws that information away. Every gain we made came from *using* it — first by localizing
> defects (detection), then by predicting defect categories as an auxiliary task, then by letting a
> stacker combine that per-category evidence. Richer supervision beat architectural cleverness.

---

## Slide-by-slide

### Slide 1 — Title (10 sec)
Group number, project: *Bottle-base defect inspection for Krones (70,000 bottles/hour line)*.
One line: "Binary GOOD/FAULTY, but scored on F1 **and** speed **and** insight."

### Slide 2 — The problem & the scoring (40 sec)
- Inspect the base of each bottle: GOOD or FAULTY.
- ~35k grayscale images, 1280×1024, confirmed grayscale.
- Why it's hard: defects are tiny/faint, and the **defect categories are wildly imbalanced**.
- Scoring drives everything: **50% F1 + 30% efficiency + 20% insight**. A fast model that meets
  line speed (19.4 img/s) can beat a slower, slightly-more-accurate one. Keep this in view all talk.

---

## EDA (5 pts) — Slide 3 (~1.5 min)

Show three plots straight from the training notebook:
1. **Class balance** — imbalanced toward FAULTY (~58/42). → motivates `pos_weight` in the loss.
2. **Defect-category frequency** (horizontal bar) — a handful of defect types dominate; many are
   rare. → a binary label hides this; rare faint defects contribute almost nothing to a binary
   gradient. This plot *is* the motivation for our core idea.
3. **Example GOOD vs FAULTY bottles** — visually, defects are small/faint vs the whole base. →
   motivates (a) ROI crop to zoom in, (b) per-category supervision to learn subtle appearances.

One-sentence takeaway to say aloud: *"The EDA told us the signal is small, faint, and
category-imbalanced — so we needed to preserve fine detail and exploit the category labels."*

---

## Methodology / background (15 pts) — Slides 4–7 (~3.5 min)

This is the most heavily weighted section — spend real time here and tell it as a *story of
iterations*, not a list.

### Slide 4 — Starting point and the ceiling we hit
- Began with a straightforward binary classifier (EfficientNet/ConvNeXt, 224–320px).
- It plateaued (~0.84 F1 at recall ≥ 0.99) and the **same ~100 hard bottles** failed every time.
- Diagnosis: a **resolution + label-ambiguity ceiling** — tiny faint defects, and a binary label
  that says *that* a bottle is bad but not *why*.

### Slide 5 — Two levers that broke the ceiling
1. **Resolution + ROI crop.** Crop each image to its base (ROI box from COCO) and train at 448–512.
   Small defects get many more pixels. This alone lifted the classifier substantially.
2. **The key idea — auxiliary multi-task head.** Add a second head that predicts **which of the 26
   defect categories** are present, trained from the COCO annotations alongside the binary label.
   This forces the backbone to learn defect-specific features. *This is the main F1 driver.*

Explain *why* it works in one breath: under a binary loss, a rare faint defect is swamped; giving
it its own supervision signal makes the network actually represent it.

### Slide 6 — The stacker (learned combiner)
- After the network, a small **LightGBM stacker** takes `[logit(main prob), 26 aux probs,
  bottle-type]` and outputs the final probability.
- It's effectively a *learned* version of a hand-written rule ("FAULTY prob is borderline but a
  serious defect category is active → FAULTY").
- Trained **fold-safe on out-of-fold predictions** (explain OOF briefly: predictions from models
  that never saw that image — the only leakage-free way to train a second-stage model).
- We pick raw vs stack vs blend by OOF F1.

### Slide 7 — Why this also respects efficiency
- Backbone is EfficientNetV2-S, **grayscale 1-channel**, small and fast.
- The stacker is almost free (trees on ~30 features).
- The one real cost is the 3-fold ensemble — and we *measured* it (next section) to confirm we
  still clear line speed. Mention we can drop to a single fold if efficiency needs to win.

> If you also ran the detection approach, add one slide: "We separately framed it as **defect
> detection** (YOLO) — localize defects, measure box area, apply a 3-tier area rule. It hit
> ~0.95 F1 at ~2.7× line speed. Crucially, it exploits the *same per-category labels* — which is
> what convinced us category information was the real lever, independent of paradigm."

---

## Progress & competition results (10 pts) — Slide 8 (~2.5 min)

Lead with the headline number, then back it with evidence.

- **Final model: F1 = 0.9783** (multi-task EfficientNetV2-S + LightGBM stacker) — our best of the
  whole project.
- **Efficiency: meets line speed.** Single model forward ≈ 13 ms (≈ 76 img/s); deployed 3-fold
  ensemble ≈ 39.5 ms/img ≈ **25.3 img/s**, which clears the 19.4 img/s (70k bottles/hour)
  requirement with **1.30× headroom**.
- Show the **confusion matrix** and the **score-separation histogram** from the notebook — GOOD and
  FAULTY scores form two clean clusters with the threshold in the valley.
- Show the **progression table** to demonstrate method, not luck:

  | Step | What changed | F1 (recall≥0.99 unless noted) |
  |---|---|---|
  | Baseline classifier @320 | binary only | ~0.84 |
  | + 512px + ROI crop | resolution & zoom | ~0.95 |
  | + lighter augmentation | preserve faint defects | +~0.004 |
  | + aux head + stacker | per-category supervision + learned combiner | **0.9783** |

- State the trade-off honestly: most of our F1 gain is cheap (aux head, stacker); the ensemble is
  the expensive part. Under 50/30/20, we land on a config that is both top-F1 **and** above line
  speed.

---

## Lessons learned (5 pts) — Slide 9 (~1.5 min)

Make these genuine and specific — graders reward honest reflection:

1. **Richer supervision beats architectural cleverness.** We tried CBAM, MSFF, HRNet — attention
   and high-resolution machinery — and they plateaued. The aux head won not by being fancier but by
   *using labels we already had and were discarding*.
2. **Diagnose the ceiling before fighting it.** The same ~100 bottles failed every model. That told
   us it was a *signal* limit (tiny faint defects, ambiguous labels), so the fix was preprocessing +
   supervision, not a bigger net.
3. **Trust your eyes on the data.** Inspecting augmentation previews revealed heavy noise/blur was
   erasing the faint defects we needed; lighter augmentation measurably helped. Also caught a
   grayscale-display bug (a channel/normalization artifact, not real data).
4. **Respect the scoring function.** Because efficiency is 30%, we *measured* runtime as a
   first-class result and kept the model fast, rather than chasing the last decimal of F1.

---

## Q&A (15 pts) — prepare for these (~the biggest single bucket)

Anticipate and rehearse crisp answers:

- **"Why does the auxiliary head help if you only need GOOD/FAULTY?"** It shapes the features during
  training to be defect-aware; at inference we can even ignore it. Richer gradient signal,
  especially for rare/faint defects.
- **"Isn't the stacker just overfitting?"** No — it's trained fold-safe on OOF predictions, and we
  pick raw/stack/blend by OOF F1, so the choice is leakage-free. We can show the OOF comparison.
- **"How do you handle the class/category imbalance?"** `pos_weight` on the main loss; the aux head
  gives rare categories dedicated supervision; stratified CV by label × bottle type.
- **"Why is your model efficient enough?"** Show the timing cell: per-image ms and 1.30× headroom
  over line speed. Note the single-fold option for more speed.
- **"What would you do with more time?"** Port the stacker onto the detector's per-class outputs
  (replace the hand-written area rule) to get the F1 lift *without* the ensemble's speed cost;
  cross-architecture ensemble (EfficientNet + ConvNeXt) into the stacker.
- **"Why not just a bigger model / higher resolution?"** We showed diminishing returns — the limit
  is signal (faint defects indistinguishable from benign texture), not model capacity.

---

## Delivery tips

- **Open and close with the one insight** (per-category labels were the lever). Repetition of a
  single clear idea is what graders remember.
- Lead every results claim with a number, then the plot.
- Keep methodology as a *narrative of iterations* — it shows scientific process, which is exactly
  what "methodology/background" (15 pts) and "lessons learned" (5 pts) reward.
- Watch the clock: EDA and lessons are short; methodology and results are the meat. Don't overrun
  into Q&A — it's worth the most points, so leave it room.
