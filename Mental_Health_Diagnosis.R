#Data pre-processesing for mental health diagnosis codes
data <- read.csv(file='mental_health_probs.out.csv', header=TRUE, sep=',')
depressionDF <- read.csv(file='depression_feelings.out.csv', header=TRUE, sep=",")

#Change column headers
names(data) <- c("ID", "Diagnosis_1","Diagnosis_2","Diagnosis_3","Diagnosis_4","Diagnosis_5","Diagnosis_6","Diagnosis_7","Diagnosis_8",
                 "Diagnosis_9","Diagnosis_10","Diagnosis_11","Diagnosis_12","Diagnosis_13","Diagnosis_14","Diagnosis_15","Diagnosis_16")

names(depressionDF) <- c("ID", "Depressed")

#Remove NAs from Depression Survey
depressionAnswered <- na.omit(depressionDF)

#Filter only those who took depression survey from full data
df <- subset(data, ID %in% depressionAnswered$ID)

#check for duplicate IDs
n_occur <- data.frame(table(df$ID))
df[df$ID %in% n_occur$Var1[n_occur$Freq > 1],]

library(tidyverse)
full <- df %>% 
  mutate(Diagnosis_0 = if_else(is.na(df[,-1]) %>% rowSums()!=16,as.double(NA),0)) %>%
  gather(key,diagnosis,-ID) %>%
  mutate(tmp=1) %>%
  select(-key) %>%
  filter(!is.na(diagnosis))%>%
  spread(diagnosis,tmp, fill=0)

#remove prefer not to answer and No diagnosis colums 
#NOTE: There is no code for 8 or 9 
full <- full[,-c(2:4)]

#duplicate check 
n_occur <- data.frame(table(full$ID))
full[full$ID %in% n_occur$Var1[n_occur$Freq > 1],]

#Renaming Columns
names(full) <- c("ID",  "Social anxiety or social phobia", "Schizophrenia", "Any other type of psychosis or psychotic illness", 
                 "A personality disorder", "Any other phobia(eg disabling fear of heights or spiders", "Panic Attacks",
                 "Obsessive compulsive disorder (OCD)", "Mania, hypomania, bipolar or manic-depression",
                 "Depression", "Bulimia nervosa", "Psychological over-eating or binge-eating",
                 "Autism, Asperger's or autistic spectrum disorder", "Anxiety, nerves or generalized anxiety disorder",
                 "Anorexia nervosa","Agoraphobia", "Attention deficit or attention deficit and hyperactivity disorder (ADD/ADHD)")

write.csv(full, file="Mental_Health_Results.csv")



