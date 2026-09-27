# Optimization work

- Workspace: `/home/jihwan38/moa-workspace`; private GitHub repository:
  `jihwan38/moa-workspace`. Read `baseline/current.json` for the adopted runtime
  and `docs/experiments/current-work.md` for the latest branches and decisions.
- Primary objective: workload-specialized RNGD dataflow and memory-mapping
  design for Gemma's single-token quantized projections and FFN. The adopted
  kernel's execution structure may itself be poorly matched to the workload or
  hardware; it is not a proven foundation to which more techniques must be added.
  Choose and compare complete execution structures, with compatible optimization
  methods, before spending effort on local tuning.
- Prevent duplicate experiments: before a build, record the candidate ID, actual
  starting commit, prior related attempts, the materially different execution
  path, and its first stop condition, specifying whether it stops tuning, defers
  adoption validation, or rejects that prototype. A renamed candidate or tile-size
  sweep is not a new architecture. Retry a rejected family only with a specific
  change to the cost or correctness mechanism that caused its rejection. A
  justified first NPU test after static-only closure is not a retry of a measured
  hardware failure.
- Keep a named branch per candidate. Preserve each candidate's source snapshots,
  build logs/hashes, static-versus-NPU results and closure reason before moving
  to the next branch. Carry records forward without carrying an unadopted kernel
  into the next candidate. Push completed records to the private remote.
- Integrate completed records, verified tooling and the adopted runtime into
  main before a new experiment series. Check branch ancestry instead of merging
  every old branch separately; retain historical branch tips and evidence. Verify
  `baseline/current.json` against the live sources before integration. New work
  branches may start from main, but their candidate source recipes remain free
  to select another architecture or historical commit.
- E55 is an immutable historical control; E104 or any later adopted runtime is
  also a performance comparison, not a required source or starting architecture.
  Explore replacements, removals, different layouts and coupled changes to decode,
  contraction, scale, tiling and communication. A slow intermediate design does
  not rule out a coherent combination. Record the actual candidate starting point.
- The exploration unit is a coherent architecture-and-method combination. Use
  historical recipes and rejected snapshots as options, not only as records to
  compare against the current winner. Select the architecture, compatible prior
  methods, conflicting methods to remove, and changed producer/consumer layouts
  together. A method that loses in one combination is not globally rejected.
  Ground a small set of distinct designs in the actual tensor shapes, input and
  weight formats, reuse opportunities, and numerical contract. Reconsider row/
  column and cluster/slice ownership, resident versus streamed operands, memory
  layout, intermediate representation, and stage dependencies together. A new
  Git parent or candidate name alone does not demonstrate broader exploration.
- The source recipe and performance control must be separate explicit inputs.
  Use `structural_build.py --plan` to check the selected commit and file origins;
  never hard-code the adopted source into the builder. A variant `recipe.json`
  can choose a different origin and source bundle from another variant. Baseline
  method IDs are descriptions, not implemented independent switches. Compose
  methods sharing a file explicitly and verify all intended changes survive;
  JSON metadata alone does not prove source-level or numerical compatibility.
  Before an Arena comparison, verify its actual control package matches the
  declared comparison recipe; source planning does not configure a batch runner.
- While the gap to the leading teams remains large, prioritize materially
  different execution architectures and coupled designs over local parameter
  sweeps. The current winner is a comparison control, not a design constraint.
- Allocate research across QKV, O and FFN using their proportional contribution
  to the geometric-mean score; FFN's larger absolute cycle count is not a larger
  score weight. Record current evidence and the expected complete-path change,
  not a ranking promise or a sum of historical percentage gains.
- As of 2026-09-27 after E127, provisional research questions are O's repeated
  lookup setup and next-weight readiness, QKV's table preparation and Main
  execution dependencies, and FFN's
  decoded-weight/partial storage plus the packed copies introduced by direct
  decode. These are evidence-based questions, not mandated implementations.
  Reuse the recorded profiles and refresh affected assumptions when the path
  changes. A 4/8-stage sweep or another input-broadcast tweak needs a concrete
  explanation of which limiting work it removes and which new costs it adds.
  M03/C09 adoption recovers modest gains; it does not establish a large redesign.
  Follow `baseline/current.json` and the latest `docs/experiments/current-work.md`
  evidence when these provisional priorities become stale.
- E128 subsequently adopted native up/gate dot/scale/F32-reduction fusion after
  three positive allocations, extra6 and a boundary oracle. Its up/gate block
  partial DM is now absent; down partials and full FP8 weight materialization
  remain. The modest gain is recovered, and six-row tile tuning stops. The next
  larger FFN question is projection/cluster ownership that removes FP8 weight
  materialization without E123's joint packed assembly; public transfer support
  and direct-decode correctness remain feasibility questions, not proven fixes.
- E129–E131 subsequently checked complete paths: projection-owned cluster ingress
  hit an explicit target-lowering limitation despite a public `cluster_tile` API;
  QKV Main/Sub separation and joint direct-FP4 fused FFN regressed on hardware.
  Preserve those exact boundaries: public API availability is not target support,
  logical storage removal is not latency reduction, and a separate probe timeout
  is not an observed arithmetic failure of a passing whole-kernel image. Stop the
  frozen E131 row-size series. E93/E95 native-down positives remain incompletely
  validated, not demonstrated down failures. A post-GeGLU E5/native-down proposal
  must include actual quantization/broadcast and F32 partial/reduction costs and
  establish accuracy anew; it is an open question, not a proven next optimization.
- The E131 recovery follow-up separates a rejected implementation from an
  exhausted architecture. When substantial work was removed but the added
  dependencies have not been measured, use one bounded whole-path diagnostic
  if its result can select an implementable repair. For E131, compare frozen
  E128/E131v2 internal NPU spans before choosing a new copy/setup/ownership
  combination. Do not treat the end of row-size tuning as a ban on such repairs,
  or treat a profiled interval as fully reclaimable time. Native down remains
  a parallel hypothesis; its ranking value depends on whole-FFN improvement and
  accompanying QKV/O gains, not on the amount of logical storage removed.
- Classify the scope before implementation: parameter tuning, local dataflow
  replacement, or whole-projection/FFN architecture redesign. C09 row staging
  and C11 partial VRF are local dataflow replacements; do not count them as
  evidence that broad architecture alternatives have been explored. C10 changes
  operand ownership/mapping, but its lowering failure is not a performance test.
- E133 direct E5 native down subsequently improved whole FFN in three allocations
  (paired median -3.8264%) and passed nine standard input cases. Its seven exact
  down mapping cases passed, but an adversarial tiny-value case lost 50% of the
  raw down output. A separate CPU construction places a potential final-FFN
  tolerance failure within the generator's component value ranges; it is not an
  observed PRNG seed. The later E134 same-input full-FFN NPU comparison reproduced
  1920 E133 tolerance failures and passed all3840 outputs on E128 and E134. Preserve
  the earlier CPU-only records at their original evidence scope. Do not discard
  native down as a slow family or promote it as generally accurate because the
  standard cases pass. Residual representation and native term restoration must
  change together; changing only quantization invalidates an unweighted sum.
- E134 adopted weighted native down after three positive allocations (paired
  FFN median -4.451846%), nine standard inputs, nine actual-down reference cases,
  and the full-FFN regression above. Its hi/lo16/third256 terms use coefficients
  [s,s/16,s/256] before F32 term reduction. Full decoded down-weight DM is now
  absent; up/gate FP8 materialization and down F32 block partials remain. This is
  a precision repair and adoption of the E133 family, not a new architecture.
  Stop this parameter series. The finite validation and normalized CPU error
  bound do not establish arbitrary-input or end-to-end model correctness.
  Use the E134 source and measured complete path when choosing new bottleneck
  experiments; do not continue using E128 BF16-down profiles as current facts.
- E135 independent compact FP4 -> six-row fused up/gate passed descriptor and
  register checks but both whole FFN and the production projection oracle timed
  out on their first launch. No candidate output or valid FFN timing exists.
  Preserve this as unresolved runtime liveness, not arithmetic or performance
  failure of every independent-decode design. Packet-count parity (225 versus
  450 units of 256B) alone does not change the documented 128B bank phase and
  does not justify padding retries. E136 original-E4 weight-resident QKV passed
  three standard and two cancellation inputs but regressed 11.9973% against
  contemporary E134 controls. Its precision success does not validate its cost
  structure. The next selected O combination keeps small activation terms in
  TRF, streams original weights, preserves C09 ingress staging, and fuses term
  and K16 reduction before global input-scale restoration. It must include real
  four-term quantization, global amax, replication, TRF and coefficient costs.
  This is a precision/dataflow repair of the native-O family, not another newly
  invented family, and requires its own whole-kernel and boundary results.
- E137 subsequently recovered that native-O family with consumer-local four-term
  quantization. Frozen v2 regressed14.0551%; one internal NPU diagnostic showed
  its faster native stage waiting for global-max/quantization/term distribution.
  Frozen v3 moves actual BF16 shards to all consumer rowgroups, computes the
  same global K4096 scale via K16 Max in each group, and prepares four terms
  locally. It removes512KiB term distribution while increasing original-input
  replication16KiB to256KiB; local term storage and TRF loads remain. Three
  independent whole-O allocations improved14.1496/12.2978/12.2654%, with nine
  standard inputs, nine actual-projection cases/34560 exact BF16 outputs, and
  the original whole-O case7/3840 outputs passing. Adopt the frozen measured
  combination after source/image binding; stop its parameter series. Preserve
  v2's failure and v3's added costs, not just the removed bytes. This is a
  concrete recovered implementation, not proof that every failed architecture
  can be repaired. O's K256 residual fits a different local representation from
  QKV's H3840, and direct Sub Switch-to-TRF still lacks a successful lowering
  example after E55's bitmap failure. Revalidate producer/consumer ownership
  and whole-path costs before transferring this method to another kernel.
- E138 subsequently passed standard whole QKV, original cancellation/cache and
  nine raw Q/K/V cases, but its K4/H960 consumer-local native path regressed
  12.9006% against contemporary E137 controls. Correctness success does not
  validate this cost structure. One frozen readiness diagnostic may select a
  supported input-routing repair; no arbitrary H/tile/term sweep is selected.
  E139 measured the actual adopted FFN: GeGLU-to-down-TRF median36680cycles,
  followed by native1cycle later in every run/cluster. This supports E140's
  consumer-local weighted-down preparation review. Keep canonical BF16 gather
  before K1920 replication to avoid direct small-fragment ingress, preserve
  same-cluster K7680 scale via K4 Max and E134 arithmetic, and include added
  local quantization, copies, F32 residual storage and actual VRF lifetime.
  The measured readiness window is about11.7% of FFN, not a promised saving;
  do not add the overlapping gate-gather/down-weight envelope to it or treat
  this repair as a complete30th-place solution. Runtime remains E137 until a
  new complete path passes performance and changed-boundary adoption checks.
- E140 local-down v4 passed its isolated raw-down oracle but failed whole FFN.
  Actual upstream up-term descriptors copied1440B from480B sources and exceeded
  the packed destination; local-v4 overwrote the live down global scale. Three
  explicit480B Sub copies recovered local-v5 correctness, but whole FFN regressed
  3.0981%, so that local-down combination is not adopted. Copy-only E137 is adopted
  as E140 after three independent allocations, nine standard inputs and3840 exact
  complete-FFN regression outputs. Performance is effectively neutral; median
  paired -0.1847% is not a claimed breakthrough. Check actual source/destination
  byte extents and live neighbors for sliced/padded copies; logical view shapes
  and isolated helper correctness do not establish whole-memory correctness.
  The same legacy overrun overlaps a live ring32 custom_snoop_bitmap in BOTH
  E135's whole and production-oracle images. This is a concrete source repair
  justification for E141: frozen E135 plus exactly these three bounded copies,
  with E140 as independently declared contemporary performance control. Keep
  original timeouts, distinguish verified overwrite from unproved sole cause,
  and recheck actual whole/production oracle before judging the direct family.
- For a major candidate, write the proposed full execution graph and identify
  which inherited assumptions it removes: weight materialization, activation
  replication, cluster/row/column ownership, reduction/scale decomposition, or
  whole-stage completion dependencies. Trace HBM loads through conversion,
  contraction, scale/reduction and final stores; specify each intermediate's
  owner, HBM/DM/TRF/VRF placement, layout/alignment, lifetime, and consuming
  engine. Include all required conversion and transfer work in the comparison.
  Estimate both removed and newly added work along that graph. Internal helper
  boundaries, E104 term count and E104 internal layouts are design choices,
  not competition contracts. Preserve the
  actual public interface/input contract and validate changed arithmetic against
  the unchanged correctness criteria.
- Select large changes using cost evidence for the current control and the
  proposed complete path. Trace the actual producer, transfer and consumer;
  identify the dependency or shared resource that delays completion. Historical
  bottleneck analyses must be rechecked after adoption changes the path. A paper,
  faster arithmetic type or fewer source operations is a hypothesis, not evidence
  that the target kernel's limiting interval becomes shorter.
- Account for added work as explicitly as removed work: weight conversion and
  materialization, activation quantization/replication, partial write/read,
  scale/reduction, layout transfers/padding, lookup setup, smaller engine tasks
  and register lifetimes. Record which work can overlap and which waits remain.
  Do not add overlapping schedule durations into runtime savings, interpret
  logical bytes as measured bus traffic, or infer hardware stall fractions from
  a static schedule or a whole-kernel cycle count.
- When the next architecture choice depends on an unresolved cost mechanism,
  first run a bounded diagnostic that separates the competing explanations.
  State the specific question, fixed source/layout/input, measured quantity,
  added boundary costs and the outcome that changes the design decision. Reuse
  existing evidence when sufficient; do not require an exhaustive profiler or
  a new diagnostic before every experiment. A diagnostic is not an independent
  optimization family, and isolated timings are not additive kernel speedups.
  Diagnostic-only preconversion or partial execution must be labelled explicitly;
  adopted/submitted kernels must include all required work under the unchanged
  competition input and correctness contract. Validate a promising complete
  design on NPU with contemporary controls before claiming improvement.
- Keep the design broad and the feasibility check small. A small primitive probe
  may establish API support, but must not replace the proposed whole architecture
  with a convenient edit of one existing helper. Evaluate a coherent coupled
  design as such; count related parameter/build variants separately from distinct
  architecture families. Closing one prototype does not reject the whole family.
- Preserve the competition's public interfaces and correctness requirements.
  Keep source/build/package provenance and distinguish static estimates, Arena
  measurements and official submissions. Keep rejected and failed experiments.
- Use contemporary controls for performance claims. Do not promote a candidate
  based on one favorable sample or inherit another implementation's validation.
- Screen correctness and complete-kernel improvement early. For a modest positive
  result, freeze the implementation and complete bounded adoption validation;
  stop parameter exploration independently. Reserve extensive repeats or new
  diagnostics for a concrete unresolved question, and avoid prolonged polishing.
- When asked to find candidates, audit prior attempts and produce a shortlist
  before implementing. Distinguish an untried execution path from a new parameter
  choice or an already-rejected architecture. Ground paper-inspired proposals
  in the installed SDK and the competition's actual input/submission contract.
- Use bounded feasibility gates: one minimal prototype and at most one obvious
  API/shape correction by default. Compile-only closure, or a documented API/
  lowering failure, is a valid outcome. Do not keep repairing an unpromising
  candidate just to make it build; record why it ended and move to the next one.
  Candidates with a concrete benefit hypothesis proceed to Arena; an observed
  small positive gain also qualifies for bounded adoption validation. A coupled
  design may proceed on a concrete cost argument even if one isolated part loses.
- The user clarified that these gates must not become automatic abandonment of
  fixable implementation errors before any performance evaluation. An axis,
  padding, alignment or API-use error in our source is not proof that the SDK
  cannot express the architecture. Identify the failing producer/transfer/consumer
  contract and continue a concrete supported repair when its cause is understood;
  record the reason and each frozen attempt. Reassess after new evidence, rather
  than stopping solely because a compile count was reached. Distinguish a source
  bug, a demonstrated SDK limitation, an unresolved issue and a measured loss.
  Do not replace one arbitrary retry count with an unbounded parameter sweep.
- At candidate selection and after the first NPU screen, record three separate
  decisions in the existing recipe/decision record or clearly labelled sections:
  adoption (evidence, missing checks, adopt/validate/defer/reject), further tuning
  (continue/stop and the specific cost mechanism), and major research priority
  (which materialization, ownership or completion dependency changes). One
  overall "closed" flag must not silently decide all three. A candidate may be
  adopted while both parameter tuning and further research on it stop.
- The provisional 5% target-kernel cutoff limits additional parameter tuning;
  it is neither a significance threshold nor a minimum adoption gain. A single
  positive allocation calls for bounded independent contemporary controls and
  correctness checks at the changed arithmetic/layout boundaries. Adopt the
  frozen implementation when these support a reproducible net gain. If that
  validation must be deferred, record its actual cost or unresolved issue and
  the remaining checks; gain below 5% alone is not a deferral/rejection reason.
  Do not describe an unvalidated positive candidate, such as E123's initial
  screen, as a failed architecture or a proven worthless adoption option.
- Before running a new experiment, inspect its recipe, result-classification
  code and proposed next actions against these three decisions. Never copy a
  numeric branch such as "gain < 5% => no repeats/correctness/adoption" into a
  new runner. E126's frozen recipe/results.py/decision preserve that historical
  mistake for reproduction; their policy is superseded by E127 and these rules.
  Keep historical evidence immutable and write a separate follow-up decision.
  Include this strategy-consistency check when delegating candidate work; the
  parent agent owns the final adoption/tuning/research decision. AGENTS.md is an
  execution instruction, not an automatic enforcement mechanism in the tools.
- Apply measured percentage screens to NPU results. Static schedules can justify
  a budget-based early stop for a specific prototype, but are not a calibrated
  NPU performance oracle; small static differences do not establish a hardware
  regression or an architecture-wide upper bound. Report that limitation.
