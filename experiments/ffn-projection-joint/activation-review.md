# Authorized startup activation

Root authorized activation after all11 joint hardware cases (`r1060`–`r1070`)
passed. The frozen candidate and control both use the current exact stationary
sliding/full O kernels. Their263 vocabulary comparisons cover68,943,872 values,
and their repeated raw-cache/sentinel comparisons cover27,179,089,920 bytes;
all match exactly. Actual reported available HBM never falls below3,520,462,080B.
Candidate-first reset mean saving is2.875344ms across these11 paired model runs;
this is not an HTTP throughput measurement.

The activation changes only startup preparation and prepared-state defaults.
One call in `Worker::new_with_storage` creates immutable joined weights before
serving canonical grouped requests. No request-time join, scheduler policy,
credit, reservation or cursor changes occur. Constructors use storage presence,
so manually constructed diagnostic models without joined storage keep the
original core. Joined storage is published only after all layer copies and its
output allocation succeed; startup errors propagate before serving.

`round2_path` independently reviewed the complete activation patch. The three
real constructors select the candidate only when storage exists; every wrapper
inherits their real states. Disabling composition/global preload or any of the
seven older norm/core/residual routes clears projection selection. Prompt512
and pair512 wrappers delegate to those fallback methods. The existing early
compatibility guard and both exact O flags/producers remain unchanged. No
ownership, fallback or startup-source blocker was found.

The attached `activation-draft.patch` and `activation-plan.md` record the patch
prepared before authorization; it is now applied after the hardware gate.
Root will build and evaluate the activated server. The model/kernel evidence
alone does not establish a new public-task score, HTTP SLO or leaderboard rank.
