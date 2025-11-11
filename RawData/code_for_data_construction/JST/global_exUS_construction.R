rm(list = ls())
library(data.table)
library(plyr)
library(roll)
library(readxl)
library(zoo)

JST_Data <- data.table(read_excel("JSTdatasetR6.xlsx"))

# keep non-US countries
m <- JST_Data[iso != "USA"]
unique(m$country)
m[country == "Canada", gdp := gdp * 1e3]
m[country == "Denmark", gdp := gdp * 1e3]
m[country == "France", gdp := gdp * 1e3]
m[country == "Germany", gdp := gdp * 1e3]
m[country == "Italy", gdp := gdp * 1e3]
m[country == "Japan", gdp := gdp * 1e6]
m[country == "UK", gdp := gdp * 1e3]

m[, inflation := cpi / shift(cpi) - 1, by =.(iso)]
m[, real_gdp_growth := rgdpbarro / shift(rgdpbarro) - 1, by =.(iso)]
m[, gdp_growth := gdp / shift(gdp) - 1, by =.(iso)]
m[, growth_xrusd := xrusd / shift(xrusd) - 1, by =.(iso)]
m[, eq_tr := eq_tr - growth_xrusd]
m[, dollar_gdp := gdp / xrusd]
m[, gdp_weight := dollar_gdp / sum(dollar_gdp), by =.(year)]

m <- m[year > 1945]
m <- m[!is.na(inflation) & !is.na(real_gdp_growth) & !is.na(eq_tr)]

m1 <- m[,
    .(inflation = sum(inflation * gdp_weight) / sum(gdp_weight),
    real_gdp_growth = sum(real_gdp_growth * gdp_weight) / sum(gdp_weight),
    eq_tr = sum(eq_tr * gdp_weight) / sum(gdp_weight)),
    by = .(year)
]

write.csv(m1, "global_exUS.csv", row.names = FALSE)