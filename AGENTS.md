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
- Classify the scope before implementation: parameter tuning, local dataflow
  replacement, or whole-projection/FFN architecture redesign. C09 row staging
  and C11 partial VRF are local dataflow replacements; do not count them as
  evidence that broad architecture alternatives have been explored. C10 changes
  operand ownership/mapping, but its lowering failure is not a performance test.
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
