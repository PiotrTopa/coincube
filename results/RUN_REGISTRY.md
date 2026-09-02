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
| Sep 2 | m8 (T0=1) stdout logs, twice | log files lost while the JSON landed intact: git stash/rebase operations executed against the repo during the live runs replaced the tracked log's inode, orphaning the writer's file descriptor. The two runs' JSONs are byte-identical (determinism certificate). Lesson adopted: long runs write logs outside the repo and copy in on completion; no git operations against a repo with live writers. Final log from run 3. |

## Evidence runs (current)

Every results/*.json and *.log file in this directory is the output of the
run that produced it; instruments crash on gate failure before writing.
The runs backing the paper's quoted numbers: w3c_corner (4 rows, per-q
gates), m8_corner (2 rows, gated), w4_helicity (gated), i3_quenched3d
(g=0 exactness gate), w3c_positive_control (anisotropic known-answer
controls A and B), plus the theory/certificate scripts (assert before
write).
