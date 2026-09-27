# E138 native-v2 boundary checks

Prepared only. Rebind the immutable v1 full2/raw9 input and reference suites to
the native-v2 seed-DMA/ring32 BF16 input routing. Never inherit a v1 result as a
v2 pass. Source/helper/image and all actual uploaded bytes are verified anew;
the v1 suite remains unchanged in ../boundary-suite. Standard whole performance
and extra6 are separate from these diagnostic images.

Status at closure: **NotBuilt / NotTested**. Native-v2 whole job111314 passed the original three standard inputs but regressed4.104136%; bounded adoption validation was not entered. These prepared tools/fixtures are preserved as source only. No v2 boundary result or inherited v1 pass is claimed.
