rm(list = ls())
library(data.table)
library(plyr)
library(roll)
library(readxl)
library(zoo)
library(R.matlab)

###########################################################################
# This file imports the data and computes the conversion yield spread
###########################################################################

# 3m TB
C <- data.table(read_xls("../RawData/TB3MS.xls", skip = 10))
C[, date := as.Date(observation_date)]
C[, year := year(date)]
C[, max.date := max(date), by = .(year)]
C <- C[date == max.date]
C <- C[, .(year = year, tb3m = TB3MS)]
C <- C[year >= 1946 & year <= 2020]

m <- C

# 3m CD
C <- data.table(read_xls("../RawData/IR3TCD01USM156N.xls", skip = 10))
C[, date := as.Date(observation_date)]
C[, year := year(date)]
C[, max.date := max(date), by = .(year)]
C <- C[date == max.date]
C <- C[, .(year = year, cd3m = IR3TCD01USM156N)]

m <- merge(m, C, by = c("year"), all.x = T)


# Banker's acceptance rate
C <- data.table(read_xls("../RawData/BA3M.xls", skip = 10))
C[, date := as.Date(observation_date)]
C[, year := year(date)]
C[, max.date := max(date), by = .(year)]
C <- C[date == max.date]
C <- C[, .(year = year, ba3m = BA3M)]

m <- merge(m, C, by = c("year"), all.x = T)

# summ stat
mean(m[, .(cd3m - ba3m)]$V1, na.rm = T)

m[, rf := cd3m]
m[is.na(rf), rf := ba3m]

m[, spread := (rf - tb3m) / 100]
rates <- m


m <- data.table(read_xlsx("../RawData/treasuries_crsp_46_20.xlsx"))
m[, caldt := as.Date(MCALDT)]
m[, maturity := as.Date(TMATDT)]
m[, yearmon := as.yearmon(caldt)]
m[, price := TMNOMPRC / 100]
m[, quantity := TMTOTOUT]
m[, mktcap := price * quantity]
m[, duration := TMDURATN] # Macaulay's duration
m[, semiann.yield := TMPCYLD]

m[, day.to.maturity := as.numeric(maturity - caldt)]

m[, mult := 0]
m[day.to.maturity <= 90, mult := 1]
m[day.to.maturity > 90 & day.to.maturity <= 365, mult := .9]
m[day.to.maturity > 365 & day.to.maturity <= 365 * 2, mult := .8]
m[day.to.maturity > 365 * 2 & day.to.maturity <= 365 * 3, mult := .7]
m[day.to.maturity > 365 * 3 & day.to.maturity <= 365 * 5, mult := .5]
m[day.to.maturity > 365 * 5 & day.to.maturity <= 365 * 7, mult := .3]
m[day.to.maturity > 365 * 7 & day.to.maturity <= 365 * 10, mult := .1]


m[, duration := duration / 365]
m <- m[year(yearmon) >= 1946]
m <- m[!is.na(mktcap), .(mult = sum(mult * mktcap) / sum(mktcap)), by = .(yearmon)]

m[, year := year(yearmon)]
m[, max.date := max(yearmon), by = .(year)]
m <- m[yearmon == max.date]
m <- m[, .(year = year, mult = mult)]

m <- merge(m, rates, by = c("year"))
setorder(m, year)

m <- m[, .(year, spread, mult)]
# write.csv(m, file = "convyield.csv", row.names = F)
writeMat("MAT/convyield.mat", m = m)
