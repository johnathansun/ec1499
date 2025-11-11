clear; close all;
LoadData_Benchmark;

gdp = exp(cumsum(rpcgdpgr));
taxrev = taxrevgdp .* gdp;
spend = spendgdp .* gdp;

w = [gdp, taxrev, spend];
[h, pValue, stat, cValue] = jcitest(w);

save('MAT/coint_test_w.mat');
