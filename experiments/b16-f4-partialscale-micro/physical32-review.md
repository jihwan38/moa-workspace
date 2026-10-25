# Read-only physical pool32 review

Reviewed peer `physical-pool32` frozen `f3164337`; no source edits. The physical
`Rc=32` axis is exported under the existing `Rc64` compatibility spelling while
the query `R64=64` axis stays unchanged. Changed ring views use Rc1#32 and
Ts#32768 consistently; the zero loop follows Rc::SIZE. FullFc256 and context
axes are untouched. The affine byte stride of one ring is unchanged.

`Workspace::new_storage` bounds canonical requests by Rc::SIZE, allocates the
new physical pool, and prepares ring offsets only for0..requests. Worker now
passes physical capacity32. B16 and R64 readers/scatters derive the enclosing
extent from Rc; no retained hardcoded physical64 extent was found in the
inspected production paths. R64 direct64-live commits remain guarded by
metadata.live==R64::SIZE and are unreachable with32 distinct admitted owners;
<=32 calls retain B16 live-prefix segment commits and untouched inactive rows.

The admission/model tests address slot31 success, slot32 rejection,32-owner
exhaustion/queueing and epoch-safe reuse. This source/ownership review does not
prove actual device lowering or parity. Existing44/64-owner diagnostics cannot
be reused unchanged; peer is preparing separate <=32-owner model gates.
