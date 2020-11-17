#Merge Mental Health Results to labels (~157000 -> 88250)

diagnosis <- read.csv(file="Mental_Health_Results.csv", header=TRUE, sep=",")

library(reticulate)
np <- import("numpy")
labels <- np$load("labels_full.npy")
labels <- as.data.frame(labels)

names(labels) <- c("ID")
labels$ID <- as.numeric(labels$ID)

#Join mental health results data to labels
library(dplyr)
full <- dplyr::left_join(labels,diagnosis,by="ID")
full <- full[-c(2)]

colSums(is.na(full))

#replace NA with 0
full[is.na(full)] <- 0
colSums(is.na(full))

sum(full$Depression == 1)
#7752

write.csv(full, file="Diagnosis.csv")
