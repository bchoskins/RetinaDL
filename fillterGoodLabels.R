#filter diagnosis based in 'good' labels

diagnosis <- read.csv(file="Diagnosis.csv", header=TRUE, sep=",")
diagnosis <- diagnosis[-c(1)]

library(reticulate)
np <- import("numpy")
labels <- np$load("labels_good.npy")
labels <- as.data.frame(labels)

names(labels) <- c("ID")
labels$ID <- as.numeric(labels$ID)

bar = diagnosis[!(duplicated(diagnosis)),]
dim(bar)
foo = left_join(labels, bar)

dim(foo)


write.csv(foo, file="newDiagnosis.csv")

