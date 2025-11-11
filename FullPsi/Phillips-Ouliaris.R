rm(list = ls())
library(data.table)
library(plyr)
library(tseries)
library(R.matlab)

m <- readMat("MAT/coint_test_w.mat")
m <- data.table(m$w)

res <- ts(m)

po.test(res, lshort = F)
