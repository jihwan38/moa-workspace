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
- E141 subsequently repaired both E135 whole/oracle liveness with exactly the
  three bounded copies: whole45 comparisons and245760 selected original-BF16
  projection outputs pass. Actual bitmap addresses stayed unchanged. The complete
  direct six-row FFN nevertheless regressed3.151638% against E140; do not adopt it
  or continue tile tuning. A bounded same-image E140/E141 profile is selected to
  identify the cost of ten lookup preparations versus two before choosing the
  previously attempted direct30-row native/F32-block-partial combination. E103
  truly had oracle mismatches, but also retained the unsafe prefix and has no
  preserved final descriptors. Its cause is unresolved; neither inherit success
  from E141 nor treat its failure as proof that all direct partial paths fail.
- D-E141 paired diagnostic subsequently observed first direct native15,434cycles
  earlier but last up native10,888cycles later, with GeGLU10,810cycles later.
  Repeated lookup/table and direct native preparation are a concrete coupled
  cost question; table interval unions include overlap/waits and are not savings.
  E142 selects the historical full30 direct native/F32 block-partial helper with
  E141's corrected prefix/current suffix. It trades ten lookup/native streams for
  two plus ten separate scale/reduce tasks and28.125MiB partial write/read each.
  This revisits an existing combination after a source repair and measured cost
  evidence, not a new architecture or a proven speedup. Validate new actual
  partial bounds, original projection oracle and contemporary whole performance.
- E142 subsequently adopted that full30 direct/F32-partial combination after
  three independent whole FFN improvements3.3952/4.2150/3.6193percent (paired
  median3.6193), nine standard inputs,245760 exact first-BF16 projection outputs
  and3840 exact complete-FFN regression outputs. QKV/O images stay E140. All
  up/gate FP8 weight materialization is now absent; new up/gate F32 block partials
  and retained down partials remain. Stop this parameter series. This recovers
  a historical helper in a corrected/current surrounding path; it does not prove
  the sole cause of E103's old errors or general exactness of native arithmetic.
  Keep the added28.125MiB partial write/read each and separate scale tasks in
  future cost comparisons. Reassess QKV/O and remaining FFN readiness using the
  adopted image before selecting further major work; FFN3.62percent alone is
  insufficient for the current30th-place geometric-mean gap.
- E142 official submission e1f58b9e subsequently passed15 comparisons and became
  the team best at6.526159x,46/70. The current30th-place score7.896678x still needs
  a21.00percent score increase under a frozen leaderboard. Next discriminate
  E138-v2 QKV's remaining first-input readiness from later Q/K weight readiness
  using its frozen actual image; v1's trace does not establish v2's bottleneck.
  Do not assign the remaining4.10percent regression entirely to quantization.
  Four-term bit construction still lacks byte-equivalent formulas and a legal
  low-pass-count SDK pipeline. M06-v4 lacks the three unsafe packing DMAs, so
  E140's repair does not justify treating both M06 timeouts as solved. See
  docs/research/after-e142-priority-20260927.md before selecting another kernel.
- D-E143 subsequently measured frozen E138-v2 against actual E142 QKV in one
  allocation136455:81 comparisons pass, whole QKV regresses4.0916percent. Every
  run/cluster has first V native at TRF-end+1 and Q/K at their weight-end+2.
  E144 selects canonical full-H row-owned E4 ingress plus Main Switch into the
  same native K4 consumer, retaining four-term precision and all head arithmetic.
  This changes the weight routing boundary; no full transpose DM or weight TRF
  is intended. Include actual Switch/collect/reader costs, verify target lowering
  and all32768 row-quarter packet coordinates before NPU. Reuse E138 QKV source
  while retaining E142 FFN; historical source origin is not the performance
  control. Read docs/research/e144-qkv-fullh-switch-selection-20260930.md. The
  September30 public ranking is58/79 at the same E142 score; no new rank gain is
  implied by diagnostic readiness or the proposed ingress change.
- E144 subsequently adopted full-H row ingress plus Main Switch into native K4
  QKV after three independent paired improvements7.4094/8.0340/7.9907percent,
  nine standard inputs,73728 exact raw-projection outputs and16384 full-QKV
  outputs at original tolerances with8380416 untouched cache entries preserved.
  E142 O/FFN images are byte-identical. The v1 row-order lowering failure was a
  source contract error; v2 fixes low-two-bit row ownership before Switch. Keep
  both attempts. Weight HBM/DM logical30MiB remains and Switch costs are included.
  Adopt the frozen complete combination and stop parameter tuning. E143's old
  E138-v2 readiness trace is no longer current; one fixed E144 readiness profile
  should distinguish input/TRF, later weight, and head-gather completion before
  the next architecture selection. Do not infer exclusive DMA causality from
  the whole gain or claim30th place from a single-kernel improvement.
- E144 official submission2a66e430 subsequently passes15 comparisons, becomes
  BEEF best6.748109877x and ranks55/80 at2026-09-30T14:01Z (prior capture58/79).
  Current30th8.068829902x needs19.5717percent more score under the frozen board.
  Official unchanged O/FFN variation is not attributable to QKV optimization.
  Preserve paired QKV7.9907percent as the controlled gain and follow
  docs/research/after-e144-priority-20260930.md for the next bounded diagnostic.
- D-E145 subsequently measured actual E144 QKV: all six run/cluster samples have
  V native at activation TRF-end+1 and Q/K at own weight-end+2. Scalar DMA begins
  before V weight ends; do not label its whole readiness interval a pure bus stall.
  E146 selects two full-H Q/K row stages with the original single canonical gather,
  retaining V and the precision path. Weight30MiB stays, DMA/native starts3->5.
  First variants hit scheduler Scratchpad allocation failures; an actual vISA
  audit found constant-bool K/V Branches and tensor PhiFunctions. Unconditional
  K/V functions in v4 compile without v2 initialization or v3 assembly copies.
  This is a supported source-contract repair, not performance evidence or proof
  of the sole compiler cause. Check actual stage extents, packet ownership,
  disjoint output coverage, shared-register lifetime and whole NPU performance.
- E146 v4 subsequently passed three whole screens, raw73728 exact BF16 values
  and full16384 tolerance/cache8380416 checks, but paired QKV changes were
  -1.494967,+0.710594,+0.406876percent; median+0.406876percent. Do not adopt it or
  replace the paired outcome with the favorable aggregate ratio. Preserve the
  first positive and compiler repairs; no extra6 NPU was run after the reversal.
  Stop row/stage tuning. E147 selects E144 source/control with one-pass scale
  production and scalar Sub broadcast directly to VRF, preserving precision and
  original whole weight stages. Verify finite-BF16 scale equivalence, actual
  ring4 bitmap/valid masks/VRF lifetimes and full cost; D145 readiness is not a
  guaranteed saving or proof of exclusive TDMA causality.
- E147 subsequently adopted one-Main finite-BF16 scale and Sub ring4 direct-VRF
  scalar/coefficient supply after three paired improvements1.162207/2.262397/
  0.796883percent (median1.162207), nine standard inputs, raw73728 exact BF16
  values and full16384 tolerance/cache8380416 checks. All45 other source files
  and O/FFN images match E144; no E146 staging is included. Include the new
  shared16KiB bitmap DRAM load as well as removed24KiB scalar replication and
  increased pointwise Mul work. CPU scale equivalence covers65280 finite BF16
  patterns, not arbitrary NPU/model correctness. Stop this parameter series;
  modest reproducible gains are adopted independently of5percent tuning limits.
  E146 remains unadopted because its first gain failed to reproduce. Neither
  the score gap nor residual weight bottlenecks are solved by this small gain.
- E147 official aabfb8c6 subsequently passed15 comparisons with QKV94391,
  O42212 and FFN305502cycles; team best6.756274761x(+0.120995percent), rank55/80
  unchanged. QKV is1.591985percent below the prior official95918; O/FFN images
  are identical and their observed variation is not an E147 code change. The
  current30th-place score8.068829902x still requires19.4272percent more score.
  Keep the modest recovered gain without representing it as a ranking breakout.
- E148 O one-Main scale/Sub ring16 direct-VRF plus two direct cumulative sums
  subsequently passed exact descriptors/registers and45 standard comparisons,
  but whole O43945 versus E147 bracket42546 regressed3.288206percent in141077.
  Do not adopt or infer benefit from1MiB logical sum writes/reads removed. Preserve
  Main/Sub/VE coupling and16KiB shared bitmap cost. Raw9/fullcase7/extra6 packages
  were built but not executed after this screen. One fixed E147/E148 names-only
  O readiness comparison is selected to discriminate scalar/term/resource and
  weight/head-tail effects; it may justify a specific repair, not a tile sweep.
  The same diagnostic host can observe unchanged actual E142 FFN. Source-level
  duplicate up/gate TRF loads already share one actual TRF; do not count them as
  an unremoved5.625MiB transfer or select source-only commoning as a speedup.
- D-E148 subsequently confirms the frozen O prototype regresses again4.473631percent
  in141134(63comparisons pass). Quantization/TRF finish about12kcycles earlier,
  while last native waits for weight1-end+2 in all6 observations on both images.
  Do not call its removed storage inherently useless or attribute exclusive physical
  contention from trace envelopes. Stop local scalar/term tuning. E149 selects
  E147 source/control with unchanged O60+60 rows and native arithmetic, changing
  weight ownership to low2 row bits/K1024 ingress then Main InterTranspose4 into
  the original K256 consumer. Include routing cost and prove actual256B descriptor
  loops/row coverage; source1024B contiguous regions are not1024B DMA packets.
  Current E142 FFN also has gate at shared-TRF-end+1 and down at TRF-end+1. Its
  up-result gather finishes just after the long down-weight ingress; these may
  motivate a coupled supply/consumer change, not source-only TRF commoning.
- E149 K1024 O ingress/Main Switch4 subsequently passes45 standard comparisons
  and exact61,440 packet/15MiB/consumer/register gates but regresses34.616647percent
  whole O in141183. Do not adopt or repeat ring/row sweeps. The whole result does
  not isolate Switch service, HBM efficiency or bank contention. E150 selects
  current E147/E142 FFN weighted-down30+30 compact ingress/native into disjoint
  halves of the existing full60 F32 partial allocation; all five12-row scale/
  reduce consumers and quantizer stay unchanged. Current141134 evidence shows
  up gather finishing just after long down-weight ingress, then down native at
  activation-TRF-end+1. Splitting changes native dependencies and may let gather/
  GeGLU progress; source order does not enforce this. Include additional lookup/
  native/DMA/setup, actual28800B packed half extents/128B phase and cross-half
  rows24..35 consumer. No padding retry follows from odd128B packet counts alone.
- E150 subsequently adopts current E147/E142 weighted-down30+30 ingress/native
  after three paired FFN improvements2.193182/1.772074/2.206449percent, median
  2.1931816percent. Source/tag is ad1ecb3 / baseline-e150. All135 standard-stage
  comparisons plus extra6/30 comparisons pass: nine distinct standard inputs.
  Raw down9 passes34560 exact BF16 values (signed zero equivalent), and the
  original full-FFN regression passes3840 exact values with the original13input
  and official-reference checks. All46 compile inputs are bound; only mlp.rs
  changes and QKV/O images match E147. Two native streams write directly into
  the original partial root at0/14400B; five12-row consumers, including24..35,
  remain. Include the extra weight DMA/native/lookup setup: weight29491200B
  and partial write/read14745600B each remain unchanged, with no assembly copy.
  No exclusive DMA-contention cause or generic-input/model accuracy is proved.
  Stop stage/row tuning; fixed-image readiness diagnosis is running separately.
  Official submission is pending; do not claim a new official score or rank.
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

- E150 official ac53ad0a subsequently passed15 comparisons: FFN296854,
  score6.838931882,57/80 at2026-10-01 19:11 KST; the score improved1.223413%
  but rank did not change. QKV/O image timing variation is not an FFN code gain.
  Its current readiness141326 passed51 comparisons and showed last native to
  first reduce median10244cycles, with down scale DMA ending8965cycles after
  native and scale VRF to each reduce exactly1cycle in all6 observations.
  Select E151's row-low2 scale ingress/SubSwitch4-to-original-VRF combination
  for this measured supply boundary. Preserve weighted arithmetic and native
  partials; include the new14.0625MiB F32 routing/setup. These overlapping
  intervals are not a promised saving. No stage/row-count sweep is selected.

- E151 subsequently adopted the scale-ingress/Sub-to-original-VRF combination
  as baseline-e151/8adbbae. Three E150-bracketed allocations improve FFN by
  paired median0.716484859%; all candidates beat both individual controls.
  Keep the high first after-control308406 in the record; do not present its
  paired2.634532% as the reproducible gain. Nine standard inputs,34560 exact
  raw-down values and3840 exact full-FFN regression outputs pass. Source46
  and actual uploads/images are bound; QKV/O stay E150. Scale payload3,686,400B
  is unchanged and14,745,600B F32 Switch routing/setup is included. Preserve
  the weighted/native/partial/reduction arithmetic. Stop parameter tuning,
  recover the validated small gain, and do not infer that all10244cycles of
  the old readiness interval disappeared. The parallel E152 gate staging
  uses E150 independently; it does not inherit E151, and any combined source
  requires separate complete-path checks. Official submission is separate.

- E152 then passed45 original whole comparisons but regressed10.05543percent
  against contemporary E150 controls. Its gate15+15 ingress/native directly
  wrote original full30 partial halves with no assembly; static input gather
  did interleave between the two DMAs. This still did not establish whole gain.
  Preserve source/actual-descriptor evidence and built-but-unrun raw/full suites.
  Do not adopt or tune gate row counts. Reconsider initial activation producer
  ownership only with an actual complete-path mechanism; source-order or static
  earlier readiness alone is insufficient. E151 remains adopted independently.

- E151 official2ac6e446 passes15: QKV92705/O42977/FFN295105, score6.8348.
  Public best remains E150ac53ad0a6.838931882,57/80 at19:48KST October1.
  Distinguish latest final-evaluation source from best displayed score; unchanged
  QKV/O timing variation is not FFN code gain. Deadline is20:59KST October1;
  the official announcement selects the last pre-deadline submission as final.

- E153 first whole141557 passes45 but regresses1.145435percent against E151
  brackets(298332 versus294978/294929). Actual original RMS arithmetic through
  F32 DM is preserved, followed by local BF16H480, globalMax on8valid shards
  (emitted ratio256), late packed23040B and original ring/native/down. The
  first loss prevents adoption; raw8/fullFFN are built but NPU0. Stop parameter
  sweeps. One frozen E151/E153 names-only readiness comparison may identify a
  concrete added dependency or shifted suffix cost before a supported repair.
  Removing early gather does not itself prove speed or the old M06 timeout cause.
