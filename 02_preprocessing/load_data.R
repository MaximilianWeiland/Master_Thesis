setwd("/Users/maxweiland/Desktop/SEDS/Master_Thesis")
data <- readRDS("data/Corp_HouseOfCommons_V2.rds")
write.csv(data, "data/data.csv", row.names = FALSE)
