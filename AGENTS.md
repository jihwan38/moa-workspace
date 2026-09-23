# Optimization work

- Workspace: `/home/jihwan38/moa-workspace`; private GitHub repository:
  `jihwan38/moa-workspace`. Read `baseline/current.json` for the adopted runtime
  and `docs/experiments/current-work.md` for the latest branches and decisions.
- Prevent duplicate experiments: before a build, record the candidate ID, actual
  starting commit, prior related attempts, the materially different execution
  path, and its first stop condition. A renamed candidate or tile-size sweep is
  not a new architecture. Retry a rejected family only with a specific change to
  the cost or correctness mechanism that caused its rejection.
- Keep a named branch per candidate. Preserve each candidate's source snapshots,
  build logs/hashes, static-versus-NPU results and closure reason before moving
  to the next branch. Carry records forward without carrying an unadopted kernel
  into the next candidate. Push completed records to the private remote.
- E55 is an immutable comparison control, not a required starting architecture.
  Explore replacements, removals, different layouts and coupled changes to decode,
  contraction, scale, tiling and communication. A slow intermediate design does
  not rule out a coherent combination. Record the actual candidate starting point.
- While the gap to the leading teams remains large, prioritize materially
  different execution architectures and coupled designs over local parameter
  sweeps. The current winner is a comparison control, not a design constraint.
- Classify the scope before implementation: parameter tuning, local dataflow
  replacement, or whole-projection/FFN architecture redesign. C09 row staging
  and C11 partial VRF are local dataflow replacements; do not count them as
  evidence that broad architecture alternatives have been explored. C10 changes
  operand ownership/mapping, but its lowering failure is not a performance test.
- For a major candidate, write the proposed full execution graph and identify
  which inherited assumptions it removes: weight materialization, activation
  replication, cluster/row/column ownership, reduction/scale decomposition, or
  whole-stage completion dependencies. Estimate both removed and newly added
  work along that graph. Internal helper boundaries, E104 term count and E104
  internal layouts are design choices, not competition contracts. Preserve the
  actual public interface/input contract and validate changed arithmetic against
  the unchanged correctness criteria.
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
- Screen correctness and improvement magnitude early. Close modest improvements
  promptly after bounded validation, then move to a materially different design.
  Reserve extensive repeats and diagnostics for promising gains or a concrete
  correctness failure; avoid prolonged polishing of small improvements.
- When asked to find candidates, audit prior attempts and produce a shortlist
  before implementing. Distinguish an untried execution path from a new parameter
  choice or an already-rejected architecture. Ground paper-inspired proposals
  in the installed SDK and the competition's actual input/submission contract.
- Use bounded feasibility gates: one minimal prototype and at most one obvious
  API/shape correction by default. Compile-only closure, or a documented API/
  lowering failure, is a valid outcome. Do not keep repairing an unpromising
  candidate just to make it build; record why it ended and move to the next one.
  Only promising candidates proceed to Arena and extended validation. A coupled
  design may proceed on a concrete cost argument even if one isolated part loses.
- The default screen closes regressions and gains below 5% of the target kernel
  without further parameter tuning. This is a provisional experiment-budget
  cutoff, not a significance or promotion rule. Record concrete reasons for an
  exception. Maintain correctness and contemporary-control requirements before
  adoption, and distinguish Arena experiments from official submissions.
- Apply measured percentage screens to NPU results. Static schedules can justify
  a budget-based early stop for a specific prototype, but are not a calibrated
  NPU performance oracle; small static differences do not establish a hardware
  regression or an architecture-wide upper bound. Report that limitation.
