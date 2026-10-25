# Next separate model-only step: three real owners totaling512

This is a source-backed proposal after the Tail384 gate, not implemented here.
Current legal full512 constructors cover one owner512, two owners with quarter
counts3+1/2+2/1+3, or four independent128 owners. The missing real-input cases
are256+128+128 and its two order-preserving permutations.

Add a separate PreparedThreeOwner512 wrapping the existing PreparedGroup plus
three first/final row maps. Validate exactly3 independent reservations and three
aligned checked ranges of128 or256, totaling512, with final-only logits. Derive
the four chronological quarters privately, then call the existing private
GroupPrefill::prepare_rows and unchanged advance_layer. This helper already
serves the validated one/two-owner constructors; the public four-independent
GroupPrefill::supports guard must remain unchanged. There is no tensor padding,
new kernel, weight allocation, or KV cache copy.

The three final-row maps are[1,2,3], [0,2,3], [0,1,3]. Active allocations expose
exactly3 actual reservations once each; finishing returns3 owner-ordered logits.
Cancellation, shared-scratch exclusion and independent decode between complete
layers follow the same existing active-allocation contract.

Hardware oracle must compare all vocabulary outputs, true full/ring histories,
raw whole-cache bytes and untouched sentinel across all three permutations,
128/page/ring boundaries, reset and48 independent R64 interleaves. Current
Tail384+P128 or Tail256 plus remaining P128 work must be measured as the serving
baseline, not only four old P128 calls. No Worker changes before this gate.

Eventual selection can retain existing full quad/full pair priorities, then
consider the oldest3 aligned owners if one advertises256 and the other two128.
Give the extra quarter to the earliest eligible owner without reordering or
skipping any actual owner. This legal full512 route must precede padded Tail384
and Tail256 choices. Every route remains cost class4; admission, deadline and
minimum-progress policies are unchanged.
