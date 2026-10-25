# Dense I32 nibble pack actual gate

Arena321124/r1185 passed with frozen sourcecc5f9c40 and ELF9fbc1b80dcbc8d640b6e0d30e6702f077d56c6ab368ae75f4df3fbcbafb11ad7. All491,520 signedcodes and245,760 outputbytes matched the independentCPU pack/decode oracle across eightcases. Input/shift immutability, destinationpoison, true row+word permutation, all signednibble codes/topnibbles, and partial-subset AddSat safety assertions passed.

Completepack timing averaged20.933us, range20.258–22.016us over16 measurements of20 completedlaunches each. It includes input/shifts, mask/shift/saturating-add, final two-word8B commit and denseHBMstore; upload/readback are excluded. No floatquantization or rawconsumer is inside this measurement.

The validated producer returns ordinary ownedI32[B16,480] bytes, exactly30,720B. Lowcode nibble comes first; every adjacent two little-endianI32 words represent16 signedI4codes. This is suitable for the separate checked public rawbinding gate, with owner retained through consumerawait. No production route, modelquantizer or serving claim is made.
