# Production-run registry

Policy (from 2026-09-02, retroactively reconstructed before that date from
the project journal): every production-instrument launch is recorded here
with its outcome. A "discarded" run is one whose output was not used as
evidence; the reason is stated. Gate failures crash the run (hard
assertions), so a failed gate leaves no results file — this registry is the
audit trail the assertion design cannot provide by itself.

## Discards and incidents (complete list, reconstructed from the journal)

| date (2026) | run | outcome |
|---|---|---|
| Aug | w3c instrument versions v1–v5 | superseded, not evidence. Each version retired for a diagnosed estimator pathology (recorded as failure modes (i)–(vii) in the paper's measurement-pipeline appendix: sqrt-noise pole splitting, exceptional-point misidentification, through-origin fits, beat-node ratio blowup, out-of-window probing, scalar single-pole branch mixing, 1D companion locking). v6 is the production instrument. |
| Aug | co-streaming L3 variant | not a gate discard: measurement showed the co-streaming environment destroys the node (exceptional-point transition). Kept as evidence FOR the cross-streaming mandate, not as cone evidence. |
| Aug 31 | w3c v6 with hard 5%-tolerance gate | GATE TRIPPED on the annealed r2 channel (ratio 0.949, 1.2 sigma of its own jackknife). Diagnosis: fixed tolerance too tight for the low-orbit channel, not an instrument fault. Gate redesigned to significance-based tolerance max(0.02, 3 sigma); rerun passed. The trip and redesign are recorded in the journal (J41-era) and the tolerance change is visible in git history. |
| Aug 31 | overnight run_chain.sh (m8+w4+i3 chain) | shell chain died silently (orphaned nohup); no output produced, nothing discarded. All long runs since are harness-tracked background tasks. |
| Aug 31 | duplicate w4 process | a dead chain restart raced a parallel w4 copy on one log; the redundant process was killed; the surviving run's JSON verified intact (atomic end-of-run write). |
| Sep 1 | q=0.05 rows (old instrument) | transcription of superseded-instrument numbers removed from the manuscript in red-team round 1; never re-measured with v6. |
| Sep 2 | i3 first relaunch with streamed iota | crashed on GPU OOM (int64 iota); no gate involved, no data; rerun with int8 iota passed. |
| Sep 2 | m8, w4 with t=0 in fit window | superseded by reruns with the post-launch window T0=1 (review round 3, F23); pre-fix outputs replaced in place, visible in git history. |
| Sep 2 | q=0.15 cone row exclusion | the row was excluded from evidence at the ANALYSIS level (probe radii outside the node's linear window, slope change 0.30 between radii); at the time of exclusion no automated rule encoded this — the linear-range flag (LIN_MAX = 0.15, within_linear_range in the JSON) was added to w3c_corner.py afterwards, prospectively. The committed 4-row JSON predates the flag. |
| Sep 2 | m8 (T0=1) stdout logs, twice | log files lost while the JSON landed intact: git stash/rebase operations executed against the repo during the live runs replaced the tracked log's inode, orphaning the writer's file descriptor. The two runs' JSONs are byte-identical (determinism certificate). Lesson adopted: long runs write logs outside the repo and copy in on completion; no git operations against a repo with live writers. Final log from run 3. |

| Sep 3 | w3c_fresh first launch (gate row) | killed mid-run: launched at TCYC=7 before the sharp torus horizon T <= ceil(L/8) = 6 at L=48 was proven; relaunched at TCYC=6. No output written. |
| Sep 3 | w3c_control_fresh at R=1200 | CONTROL GATE TRIPPED: broken-triple (control B) r1 anisotropy not resolved at 3 sigma with R=1200 (ratio 0.727 +- 0.140 — 2.0 sigma; the same numbers are recorded in the instrument's docstring); a positive control must resolve its known answer. Relaunched at R=3600; anisotropy resolved at 8.6/11.6 sigma. No JSON was written by the failed run; its log is retained outside the repo. |
| Sep 3 | i3_fresh first launch | crashed on an over-strict amplitude-gate assertion (gate.sum() >= 3) at the starvation-prone last cycle; the gate was made tolerant (retained-momentum accounting) before rerun. No data involved. |
| Sep 4 | w4_fresh replication (R=6000, seed 91011) | GATE TRIPPED on the ANNEALED known-answer row: helicity-map isotropy statistic 0.102 +- 0.019 against the hard 0.10 threshold. The statistic is a positively biased max over the nine bilinear entries and measures the instrument's noise floor on the exactly known operator; the floor fluctuated 1.5 sigma above the committed campaign's 0.065 +- 0.017 under the independent seed stream. Run discarded per policy (no JSON written); not rerun with another seed (that would be seed selection). Consequence for the paper's claim: none reversed — the helicity isotropy statistic is already reported only as an instrument-floor upper bound (~0.1), and this trip confirms the floor's scale and seed dependence. |

## Evidence runs (current)

Every results/*.json and *.log file in this directory is the output of the
run that produced it; instruments crash on gate failure before writing.

The runs backing the paper's quoted numbers (fresh-tape campaign, Sep 3):
w3c_fresh (annealed gate + quenched F1 rows, TCYC=6 = ceil(48/8)),
w4_fresh (helicity), m8_fresh (massive, TCYC=6), i3_fresh (interaction
law, L=40, T=3), w3c_control_fresh (known-answer controls A and B at
R=3600), inout_fresh (exact two-boundary check), plus the
theory/certificate scripts (assert before write): freshtape_proof,
reread_kinematics, freshtape_interaction, theory_3d_certs (both
schedules), bridge_check, orbit_flow.

The generic-schedule campaign (w3c_corner 4 rows, m8_corner, w4_helicity,
i3_quenched3d, w3c_positive_control) remains evidence for the paper's
generic-schedule subsection (re-read corrections) and for the
schedule-comparison figure; it no longer backs the headline rows.

Formalization note: the linear-range rule (probe radii inside the node's
measured linear window) is encoded in w3c_corner.py (LIN_MAX /
within_linear_range). w3c_fresh.py computes no linear-range flag of its
own; its compliance is verified from the committed per-delta values
(worst per-delta slope variation 0.091 < 0.15 across both rows), not
enforced in code.

## Replication runs (Sep 4, doubled R, independent seed streams)

Purpose: close the observation that the committed fresh campaign's three
quenched central values drift positive in the same direction
(+1.8/+1.1/+0.7 sigma), and the caveat that w3c_fresh and w4_fresh share
a media seed stream. Outputs go to *_r2.json; the committed campaign
remains the paper's evidence.

| run | outcome |
|---|---|
| w3c_fresh_r2 (R=6000, seed 90011) | BOTH GATES PASSED. Quenched node lam0 = 0.6180 +- 0.0038 (-0.36% vs closed form 0.6202), om0 = 0.3029 +- 0.0303 (-0.4 sigma); the committed run's +1.6%/1.8 sigma excursion did NOT replicate under the independent stream — a fluctuation, as the statistics indicated. z_diamond (new fields): weakest channel r1 = 18.4 sigma, cubic channels 594/780 sigma. |
| w4_fresh (R=6000, seed 91011) | discarded — see the gate-trip entry above. |
| m8_fresh_r2 (R=6000, seed 92011) | BOTH GATES PASSED. Quenched m = 0.05569 +- 0.00412 (+0.76 sigma vs closed form 0.05258; campaign row was +1.1 sigma on the shared stream), omega_c = 0.3089 +- 0.0059; dispersion on the exact branches. Together with w3c_fresh_r2 the same-sign-drift observation is resolved: independent streams scatter both ways. |
