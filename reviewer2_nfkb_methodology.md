# Reviewer #2 — NF-κB Methodology Depth: Corrected Analysis Scripts

## What changed and why

Reading the actual pipeline (github.com/rjone143/NFKB) surfaced one serious issue and two
methodological weaknesses beyond what better prose alone could fix:

1. **`phase_nfkb.py` never applies a perturbation.** It computes a local finite difference of
   the unperturbed mean trajectory. Any pulse-shaped curve produces a "non-flat" result this
   way — it can't distinguish genuine phase-dependent sensitivity from just having a rising and
   falling phase. This is the manuscript's own strongest claimed evidence (Sections 6.1 and 10),
   so it's worth fixing rather than just describing more carefully.
2. **`residence_nfkb.py` pools all cells and replicates into one trajectory** before extracting
   dwell times, which is why only 3 dwell segments were ever found — most of the data's
   statistical power is discarded before the analysis even starts.
3. **The "Markov null" is a single random draw of 3 samples**, not a reference distribution.

## How to run

1. Put `nfkb_load_percell.py`, `coherence_nfkb_v2.py`, `residence_nfkb_v2.py`, and
   `phase_nfkb_v2.py` in the same directory as your local `AllDataSets.mat`.
2. Run each script (`python coherence_nfkb_v2.py`, etc.) — no changes needed unless your file
   isn't named `AllDataSets.mat` or isn't in the same folder (edit `MAT_FILE` at the top).
3. Each prints its results to the console and saves PNGs. Send me the printed output (numbers,
   not the 2GB file) and I'll fold the real results into the manuscript text.

**Runtime note:** `coherence_nfkb_v2.py`'s permutation null (1000 draws) and
`residence_nfkb_v2.py`'s Monte Carlo null (10,000 draws) are the slow parts. If it's too slow
on the full dataset, drop `N_PERMUTATIONS` / `N_NULL_DRAWS` in the script headers — a few
hundred is usually enough to see whether the effect is anywhere near significance before
committing to the full run.

## What to do about Method 2 in phase_nfkb_v2.py either way

Whatever the natural-phase-sensitivity result comes back as (significant or not), it's honest
either way and directly answers Reviewer #2's question — regardless of outcome, report it:

- **If significant:** this becomes your real phase-sensitivity result, replacing the current
  claim. It's correlational (activation timing predicts a cell's own later decay behavior),
  which is a *weaker* claim than "response to a controlled perturbation" — say so explicitly
  rather than overstating it as equivalent evidence.
- **If not significant:** report that too. Combined with the already-honest treatment of the
  residence-time signature as underpowered, this would mean none of the three signatures are
  strongly supported in the NF-κB case — which is a substantive finding worth stating plainly
  (see Section 10's existing "partial but meaningful support" framing, which would need revising
  down further).

## Draft replacement text for Section 6.1 (finalize numbers once you've run the scripts)

> Coherence was assessed using individual single-cell trajectories (n=[X] cells across the
> LPS_R1 replicate(s)) rather than replicate-level means, giving substantially more statistical
> power than the original replicate-pooled analysis. Peak coherence occurred at time index [Y].
> To test whether this transient reflects genuine cross-cell synchrony rather than an artifact
> of each cell's own pulse shape, we compared the observed peak against a circular-shift
> permutation null (1000 permutations) in which each cell's trajectory was independently
> time-shifted, preserving its own shape while destroying real cross-cell alignment. [Report
> p-value once run.]
>
> Residence-time analysis was revised to extract dwell times from each cell's own
> threshold-crossing behavior rather than from a single trajectory pooled across all cells,
> yielding [Z] dwell segments (compared to 3 in the original replicate-pooled analysis). The
> memoryless null was constructed from 10,000 Monte Carlo draws rather than a single random
> draw, giving an empirical p-value of [P] for the observed tail ratio.
>
> The original phase-response analysis computed a local finite difference of the unperturbed
> mean trajectory, which cannot distinguish phase-dependent perturbation sensitivity from the
> ordinary shape of an activation-relaxation pulse; we do not report it as evidence of
> phase-dependent response. Instead, we tested whether natural cell-to-cell variation in
> activation timing predicts a genuine downstream feature of each cell's own trajectory
> (post-peak decay rate), using a permutation test (Spearman ρ=[R], p=[P2]). [State conclusion
> once run: this is a correlational test of natural variation, not a controlled perturbation
> experiment, and should be interpreted accordingly.]

## Rebuttal letter paragraph (Reviewer #2, NF-κB methodology comment)

> We thank the reviewer for pressing on this point. On close re-examination of our analysis
> code, we identified that our original phase-response measure did not in fact test response to
> a perturbation — it computed the local slope of the unperturbed mean trajectory, which cannot
> distinguish genuine phase-dependent sensitivity from the ordinary shape of an
> activation-relaxation pulse. We have replaced this with [a test of whether natural cell-to-cell
> variation in activation timing predicts downstream trajectory behavior / an honest description
> of the original measure as a diagnostic rather than a discriminating test — pending script
> results]. We also revised the residence-time analysis to extract dwell times per cell rather
> than from a single trajectory pooled across all cells and replicates, and replaced the
> single-draw memoryless comparison with a Monte Carlo null distribution (10,000 draws),
> yielding a proper empirical significance test. Coherence was recomputed using individual
> single-cell trajectories rather than replicate-level means, with a permutation-based
> significance test against a circular-shift null. We believe this is a substantially more
> rigorous treatment than the original submission and addresses the reviewer's concern about
> statistical significance directly rather than only through expanded prose.

## One more thing worth doing before resubmission

The repository is currently informal (working comments, hardcoded paths, no environment file).
Since it may end up cited or reviewed directly as supporting material, consider before final
resubmission: adding a `requirements.txt` / `environment.yml`, a top-level `README` describing
the four analyses and how they map to manuscript sections, and removing exploratory artifacts
that aren't meant for a reader (e.g., informal inline comments). I can draft this if useful once
the analysis itself is finalized.
