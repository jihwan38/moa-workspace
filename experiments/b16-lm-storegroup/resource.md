# Exact-store resource gate

`a005-b16-lm-storegroup-resource` on d7ff22ea compiles. No DramStack. Estimated complete end2428542 versus original3579067 cycles (~32% reduction). Every original input/weight/TU/band-result operation has unchanged estimates. Only the final store changes: new actual DM grouping33203 cycles plus new canonical store7284, replacing original1191012. All costs are in the graph and require actual measurement.

`math-equivalence.json` proves the3711-byte prefix from the device entry through the original final output transpose is byte-identical. Unlike held H960, this adds no numerical operation and changes no reduction shape. The grouped store's actual physical row placement still requires the separate strict provided-weight all-vocabulary gate.
