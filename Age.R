#Age Data
data <-  read.csv(file="age.out.csv", header=TRUE, sep=',')
diagnosis <- read.csv(file="newDiagnosis.csv", header=TRUE, sep=",")
diagnosis <- diagnosis[-c(1)]

names(data) <- c("ID", "birth_year")

#library(eeptools)
#data$age <- floor(age_calc(data$birth_year, units='years'))

#or
data$age <- abs(c(data$birth_year) - 2019)

full <- dplyr::left_join(diagnosis, data, by = "ID")

#remove birth year
full <- full[-c(18)]

write.csv(full, file = "DiagnosisWithAge.csv")

