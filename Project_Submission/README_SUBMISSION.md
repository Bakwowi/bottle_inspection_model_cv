# Submission & Setup Guide — Krones Vision AI Challenge

This guide walks through turning the two notebooks into a valid final submission. **Read it
fully** — several steps here are the difference between a real score and an automatic **0** under
the competition rules.

---

## What you submit

The competition requires, for your group:
1. **One Training Notebook** — titled `Group XX - Training Notebook`
2. **EXACTLY ONE Evaluation Notebook** — titled `Group XX - Evaluation Notebook`
3. **The trained model**, attached to the evaluation notebook as a Kaggle Dataset and **explicitly
   shared with BOTH organizers**

> Replace `XX` with your real group number in both notebook titles. The group number must be
> clearly visible in the title — this is a stated requirement.

---

## Step 1 — Run the training notebook

1. Open `Group XX - Training Notebook` on Kaggle with the competition data attached and GPU on.
2. Run all cells top to bottom. It will do EDA, train 3 folds, train the stacker, and save these
   files to `/kaggle/working/`:
   - `model_fold0.pt`, `model_fold1.pt`, `model_fold2.pt`
   - `stacker.txt`
   - `best_threshold.json`
   - `oof.csv`, `oof_aux.npy` (analysis only — not needed by evaluation, but harmless to include)
3. Note the final OOF F1 printed in stage 10 — that's your offline estimate of the leaderboard F1.

---

## Step 2 — Package the model as a Kaggle Dataset

The evaluation notebook loads the model from `/kaggle/input/...`, so the trained files must become
a Kaggle Dataset.

1. After training, in the notebook's **Output** tab, select the artifact files
   (`model_fold*.pt`, `stacker.txt`, `best_threshold.json`) and **Create a Dataset** from them
   (or download them and upload as a new Dataset).
2. Give it a clear name, e.g. `group-xx-krones-model`.
3. Keep it **Private**.

> **CRITICAL — sharing:** A private dataset that isn't shared cannot be read by the organizers,
> which means **score = 0**. After creating it, open the dataset's **Settings → Sharing/Collaborators**
> and **add both organizers explicitly**. Sharing the *notebook* is **not** enough — the *dataset
> (model)* itself must be shared too, because of how Kaggle permissions work.

---

## Step 3 — Wire up the evaluation notebook

1. Open `Group XX - Evaluation Notebook`.
2. **Attach** your model dataset (Add Input → your `group-xx-krones-model` dataset) **and** the
   competition data.
3. In cell 1, set `MODEL_DIR` to the attached dataset's path. It will look like:
   ```python
   MODEL_DIR = Path("/kaggle/input/group-xx-krones-model")
   ```
   (Check the exact path in the right-hand "Input" panel — Kaggle slugifies the name.)
4. Run all cells top to bottom. Confirm it:
   - loads the config and all 3 fold models + stacker,
   - writes `submission.csv`,
   - prints the **inference runtime** block (total time, per-image ms, throughput, headroom).

The evaluation notebook is **self-contained**: it redefines the model, redoes the exact
preprocessing, loads the saved weights, runs inference, and times itself. It does **not** retrain
and does **not** depend on the training notebook at run time. This is exactly the
"one fully executable evaluation notebook" the rules demand.

---

## Step 4 — Test in a clean environment (do not skip)

The organizers run your notebook in a **fresh** Kaggle session and **fix nothing**. To be sure it
works for them:

1. In the evaluation notebook: **Factory reset / Restart & Run All** (fresh kernel).
2. Better: **share notebook + model with a teammate or another group** and have them
   *Restart & Run All from scratch*. If they can't run it, the organizers can't either.
3. Verify `submission.csv` is produced and the timing cell prints without error.

---

## Common pitfalls that cause score = 0 (from the rules) — and how we avoid them

| Pitfall | How we avoid it |
|---|---|
| Model attached but not shared | Step 2 — share the **dataset** with both organizers, not just the notebook |
| Missing `.onnx.data` for large models | We ship plain PyTorch `.pt` weights (no split ONNX), so this can't happen |
| External downloads (Drive/URLs/APIs) | The notebooks only read `/kaggle/input` — no internet fetches for the model |
| Notebook works for you but not others | Step 4 — fresh-kernel Restart & Run All + a teammate test |
| Multiple disconnected notebooks | Inference is **fully integrated** in one evaluation notebook |
| Manual path edits / preprocessing steps | All config is in `best_threshold.json`; the only edit is `MODEL_DIR` and the organizers' test-data switch |

---

## About the efficiency (timing) cell

Efficiency is **30%** of the final score, measured by the organizers running your timing cell
multiple times (≈3–10) and taking your **best (fastest) runtime**, then normalizing by percentile
across all teams (`t_p10 → 1`, `t_p90 → 0`).

Our timing cell (evaluation stage 7) times the **full pipeline** on the test set — image load, ROI
crop, preprocessing, all fold forward passes, and the stacker — after a warmup pass (so one-time
CUDA/cuDNN setup isn't counted). It reports per-image latency and throughput vs the 70,000
bottles/hour (19.4 img/s) line-speed requirement.

**Efficiency lever (optional):** the biggest cost is the **3-fold ensemble** (3× forward passes).
If you want to trade a little F1 for a much better efficiency score, you can run a **single fold**:
- set `train_folds` to e.g. `[0]` in training (still trains fine; the stacker just stacks on one
  fold's OOF), or
- keep all 3 folds trained but, in the evaluation notebook, load only `model_fold0.pt`.
The single model alone runs ~3× faster than the ensemble. Compare the OOF F1 you'd lose against the
efficiency you'd gain before deciding. Our default ships the 3-fold ensemble (best F1, and it still
clears line speed comfortably).

---

## Final pre-submission checklist

- [ ] Both notebook titles contain your real group number
- [ ] Training notebook runs top-to-bottom and saves all artifacts
- [ ] Model dataset created, **Private**, and **shared with both organizers**
- [ ] Evaluation notebook attached to the model dataset + competition data
- [ ] `MODEL_DIR` set correctly
- [ ] Evaluation notebook **Restart & Run All** in a fresh kernel → produces `submission.csv` + timing
- [ ] A teammate/another group confirmed it runs from scratch
- [ ] Exactly ONE evaluation notebook is shared
