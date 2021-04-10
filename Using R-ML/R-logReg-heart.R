library(tidyverse)
library(pROC)
library(ggplot2)
library(dplyr)
library(caret)
library(caTools)

# Loading the dataset
df <- read.csv("C:/Users/sanja/OneDrive/Documents/Datasets/heart_failure.csv")
head(df,10)

# seeing the structure of data
str(df)

# checking for NA's in the dataset
table(is.na(df))

# replacing numerical data as factors to perform Logistic Regression
df$anaemia <- factor(df$anaemia)
df$diabetes <- factor(df$diabetes)
df$high_blood_pressure <- factor(df$high_blood_pressure)
df$sex <- factor(df$sex)
df$smoking <- factor(df$smoking)
df$DEATH_EVENT <- factor(df$DEATH_EVENT)

# Visualization
ggplot(df, aes(x=age,fill=as.factor(DEATH_EVENT))) + 
    geom_histogram(binwidth = 5,color = "white") + 
    xlab("Age") + ylab("Number of subjects") + theme_classic() + 
    labs(caption = "Age Distribution with Death Event")

ggplot(df, aes(x=sex, fill= DEATH_EVENT)) + 
    geom_bar() +
    xlab("SEX") +
    ylab("Count") +
    ggtitle("Deaths of Male and Female") +
    scale_fill_discrete(name = "Deaths", labels = c("no", "yes"))


# Creating test and train set to perform regression
set.seed(123)
split <- sample.split(Y = df$DEATH_EVENT, SplitRatio = 0.75)
#split <- sample(c(T, F), nrow(df), replace = T, prob = c(0.75,0.25))
train <- subset(x = df, split == T)
test <- subset(x = df, split == F)
head(test)

# Building a model
model <- glm(DEATH_EVENT~.,data = train,family = "binomial")
summary(model)

test$prob <- predict(object = model, newdata = test, type ='response')
test$death_pred <- ifelse(test$prob >= 0.5, 1, 0)

colAUC(test$death_pred,test$DEATH_EVENT,plotROC=T)
auc(roc(test$DEATH_EVENT,test$death_pred))
plot(roc(test$DEATH_EVENT,test$death_pred),col='red')

CM = confusionMatrix(table(test$DEATH_EVENT,test$death_pred))
Accuracy <- round(CM$overall[1], 2)
Accuracy

test %>% 
    arrange(prob) %>% 
    mutate(rank = rank(prob), Event = ifelse(prob >= 0.5, 'Dead', 'Alive')) %>% 
    ggplot(aes(rank, prob)) +
    geom_point(aes(color = Event)) +
    geom_smooth(method = "glm", method.args = list(family = "binomial")) +
    ggtitle('Logistic Regression',
            subtitle='Predicting Mortality by Heart Failure') 
    
