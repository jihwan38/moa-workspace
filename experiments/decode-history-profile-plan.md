# Current decoder with actual independently computed histories

Previous C64 timing evidence predates several host/kernel improvements. The current mixedC7 decoded step is~56ms, so prefill wins alone cannot establish high-concurrency capacity. Measure current automatic B16/R64 serving with64 independent512-token histories, then8 actual autoregressive steps including the production greedy selection/CPU rows. Profile48layer stage costs on the laststep.

All histories are independently computed with the full model; no KV copies/reuse across requests. Allphysicalpages/slots are disjoint, whole canonicalpool and originalattention. This is a diagnostic, not an HTTP throughput score or accuracy comparison. Exact output correctness is covered by existing model/greedy tests. Bounds:width1..64,contextmultiple512≤1536,steps1..64; no serving behavior changes.
