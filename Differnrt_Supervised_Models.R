# Heart Disease Prediction Using different Supervised Learning Models

# Importing Dataset: Heart_Disease_Data.xlsx
library(readxl)
Heart_Disease_Data <- read_excel("D:/Documents/Heart_Disease_Data.xlsx")
View(Heart_Disease_Data)

# Preliminary Analysis

# Finding Summary of dataset to check for missing values
summary(Heart_Disease_Data)   

# No missing values in the dataset. hence we can proceed further
# In the dataset we are having some discrete explanatory variables. So to find their summary we use

attach(Heart_Disease_Data)      # attaching the dataset
table(Sex)
table(CP)
table(FBP)
table(RestECG)
table(ExAngina)
table(Slope)
table(CA)
table(Thal)
table(Result)

# Splitting the data randomly into training data(80%) and test data(20%)
set.seed(1)
sampleid=sample(2,303,replace=TRUE,prob=c(0.8,0.2))
result=cbind(Heart_Disease_Data,sampleid)
training=Heart_Disease_Data[sampleid==1,]
test=Heart_Disease_Data[sampleid==2,]

x_train = training[,1:13]
y_train = training$Result

x_test = test[,1:13]
y_test = test$Result


# Logistic Regression

LR_model = glm(factor(Result)~., data=training, family = binomial(logit))
summary(LR_model)
anova(LR_model,test='Chisq')

# From the ANOVA test we observe that only for Cholesteral,FBP,RestECG,Slope the p value is >0.05.
# Hence thses must be removed from the model to make the model significant. Therefore the new model becomes:
attach(training)
LR_model2 = glm(factor(Result)~Age+Sex+CP+RestBP+Max_HR+ExAngina+Oldpeak+CA+Thal,family = binomial(logit))
summary(LR_model2)
anova(LR_model2,test='Chisq')
# From the ANOVA test we observe that none of the explanatory variables are having p values >0.05. 
# Hence the model is significant to proceed further.

# Model Significance
# To check whether our model is significant or not, we need to first create a null model and compare it with our model.
null_model = glm(factor(Result)~1,family=binomial(logit))
anova(null_model,LR_model2,test='Chisq')

# Model Accuracy
pred = predict(LR_model2,type='response')
pred_class = ifelse(pred>0.5,'1','0') 

# Confusion Matrix
LR_table = table(y_train, pred_class)
LR_table
round(prop.table(LR_table)*100,2)

# Model performance on test data
pred_test = predict(LR_model2, newdata = x_test, type = 'response')
pred_class_test = ifelse(pred_test > 0.5, '1', '0')

# Confusion Matrix
table_test = table(y_test, pred_class_test)
table_test
round(prop.table(table_test)*100,2)


# Classification Tree

library(rpart)
library(rpart.plot)
attach(training)
CT_model = rpart(Result~.,data=training,method='class',control = rpart.control(minsplit = 2))

# Cp plot - to get optimum Cp value.
plotcp(model,pch=19,col='red')

# Pruning the tree with the optimum Cp value
CT_model2 = prune(CT_model,cp=0.011)
rpart.plot(CT_model2)

# Model Accuracy
pred = predict(CT_model2,type='class')
# confusion matrix
CT_table = table(training$Result,pred)
CT_table
round(prop.table(CT_table)*100,2)

# Model performance on test data
pred_test = predict(CT_model2,newdata = test,type = 'class')
# confusion matrix
CT_table2=table(test$Result,pred_test)
CT_table2
round(prop.table(CT_table2)*100,2)


# Bagging

library(randomForest)
attach(Heart_Disease_Data)
Bag_model = randomForest(factor(training$Result)~., data = training,mtry=13,importance=TRUE)
Bag_model

# important x variables  
importance(Bag_model)
varImpPlot(Bag_model)
# which RM has the highest value => it is more important

# Model performance on test data
pred_test = predict(Bag_model,newdata = test,type = 'class')
Bag_table = table(test$Result,pred_test)
Bag_table
round(prop.table(Bag_table)*100,2)


# Random Forest

RF_model = randomForest(factor(training$Result)~.,data=training,importance=TRUE)
RF_model

# Model performance on test data
pred_test = predict(RF_model,newdata = test,type = 'class')
RF_table = table(test$Result,pred_test)
RF_table
round(prop.table(RF_table)*100,2)


# Naive Bayes

library(e1071)
NB_model = naiveBayes(x_train, y_train)

pred_train = predict(NB_model, x_train)
train_table = table(y_train, pred_train)
train_table
round(prop.table(train_table)*100,2)

# Model performance on test data
pred_test = predict(NB_model, x_test)
table_test = table(y_test, pred_test)
table_test
round(prop.table(table_test)*100,2)
round(prop.table(table_test)*100,2)


# K Nearest Neighbors

library(FNN)
KNN_model = ownn(x_train,x_train,y_train)
KNN_model$k

pred_train = KNN_model$knnpred
table_train = table(y_train,pred_train)
table_train                  
round(prop.table(table_train)*100,2)

# Model performance on test data
KNN_model2 = ownn(x_train,x_test,y_train,prob = T)

pred_test = KNN_model2$knnpred
table_test = table(y_test,pred_test)
table_test
round(prop.table(table_test)*100,2)


# Support Vector Machine

library(e1071)
tune_model=tune(svm,factor(Result)~.,data=training,kernel='linear',ranges=list(cost=c(0.001,0.01,0.1,1,5,10,100)))
summary(tune_model)
best_model = tune_model$best.model
summary(best_model)

tune_model2=tune(svm,factor(Result)~.,data=training,kernel='polynomial',ranges=list(cost=c(0.001,0.01,0.1,1,5,10,100),
                                                                                    degree=c(1,2,3)))
summary(tune_model2)
best_model2 = tune_model2$best.model
summary(best_model2)

tune_model3=tune(svm,factor(Result)~.,data=training,kernel='radial',ranges=list(cost=c(0.001,0.01,0.1,1,5,10,100),
                                                                                gamma=c(1,2,3,4,5)))
summary(tune_model3)
best_model3 = tune_model3$best.model
summary(best_model3)

pred_train = predict(best_model,training)
table_train=table(training$Result,pred_train)
table_train
round(prop.table(table_train)*100,2)

pred_train2 = predict(best_model2,training)
table_train2=table(training$Result,pred_train2)
table_train2
round(prop.table(table_train2)*100,2)

pred_train3 = predict(best_model3,training)
table_train3 =table(training$Result,pred_train3)
table_train3
round(prop.table(table_train3)*100,2)

# Model performance on test data
pred_test = predict(best_model,test)
table_test = table(test$Result,pred_test)
table_test
round(prop.table(table_test)*100,2)

pred_test2 = predict(best_model2,test)
table_test2 = table(test$Result,pred_test2)
table_test2
round(prop.table(table_test2)*100,2)

pred_test3 = predict(best_model3,test)
table_test3 = table(test$Result,pred_test3)
table_test3
round(prop.table(table_test3)*100,2)