# Clean physical32 source export

The local 07718f07 diagnostic added a direct ring dependency for cross-program SHA digests. That dependency cannot be reproduced by source-only submission with the organizer Cargo.toml. The earlier c080 lock repair did not solve this source-export constraint; c080 remains preserved local evidence, not a submission-ready artifact.

This new branch starts from current root c0794e024028. It carries f3164337 physical32/admission/guards, the workspace.requests cache-comparison validation fix, and the slot31 cancelled-checkpoint CPU test. It keeps the upstream Cargo.toml/toolchain and root resolved ignored Cargo.lock byte-identical, restores the root no-digest KV comparison, and omits the three SHA diagnostic binaries.

The source audit compares every source file to frozen07718f07: only the no-digest diagnostic and three omitted binaries differ. All model kernels, physical views, admission, scheduling, ownership, and131 CPU tests are identical. Original32/64 frozen diagnostic builds/runs remain untouched.

Run package-only131 CPU tests with incremental disabled and package test debug0, locked plain server check, then a fresh plain release server through record_run. Freeze only after before/after source and dependency hashes match. Root owns c081 candidate freeze and any device/public gates.
