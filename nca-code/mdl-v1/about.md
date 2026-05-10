Modification of vanilla-v5
Instead of predicting outputs given inputs,
This predicts both inputs and outputs given a seed

NCA^n(seed) = input
NCA^n(input) = output

Loss =  MSE(NCA^n(seed) - actual input) (both test and train input grids) + MSE(NCA^n(input) - actual output) (only train grids)

Note that we do NOT want to leak the actual test output, so the second term only trains on inputs