# Heart Disease Prediction Using Logistic Regression

# Importing Dataset: Heart_Disease_Data.xlsx
library(readxl)
Heart_Disease_Data <- read_excel("D:/Documents/Heart_Disease_Data.xlsx")
View(Heart_Disease_Data)

summary(Heart_Disease_Data)               # Finding Summary of dataset to check for missing values

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


# Preliminary Analysis

# Finding relation between Response Variable & Explanatory Variables with suitable plots

# Density plots for Continuous (Explanatory Variables) Vs Categorical (Response Variable)
library(tidyverse)
data = as_tibble(data)
x = Heart_Disease_Data %>% select(Age, RestBP, Cholesteral, Max_HR, Oldpeak)
y = factor(Result)         # since y is categorical
featurePlot(x, y, plot = 'density', auto.key = list(columns = 2),
            scales = list(x = list(relation = 'free'),
                          y = list(relation = 'free')),
            adjust = 1.5, pch = "", layout=c(3,2))

# Box plots for Continuous (Explanatory Variables) Vs Categorical (Response Variable)
featurePlot(x, y, plot = 'boxplot', auto.key = list(columns = 2),
            scales = list(x = list(relation = 'free'),
                          y = list(relation = 'free')))

# Mosaic plots for Catrgorical (Explanatory Variables) Vs Categorical (Response Variable)

plot(factor(Sex),factor(Result), main = 'Mosaic Plot of Sex vs Result', xlab = 'Sex', ylab = 'Result') 
plot(factor(CP),factor(Result), main = 'Mosaic Plot of CP vs Result', xlab = 'CP', ylab = 'Result')
plot(factor(FBP),factor(Result), main = 'Mosaic Plot of FBP vs Result', xlab = 'FBP', ylab = 'Result')
plot(factor(RestECG),factor(Result), main = 'Mosaic Plot of RestECG vs Result', xlab = 'RestECG', ylab = 'Result')
plot(factor(ExAngina),factor(Result), main = 'Mosaic Plot of ExAngina vs Result',xlab = 'ExAngina',ylab = 'Result')
plot(factor(Slope),factor(Result), main = 'Mosaic Plot of Slope vs Result', xlab = 'Slope', ylab = 'Result')
plot(factor(CA),factor(Result), main = 'Mosaic Plot of CA vs Result', xlab = 'CA', ylab = 'Result')
plot(factor(Thal),factor(Result), main = 'Mosaic Plot of Thal vs Result', xlab = 'Thal', ylab = 'Result')


# Developing the model - Binary Logistic Regression Model

model=glm(factor(Result)~Age+Sex+CP+RestBP+Cholesteral+FBP+RestECG+Max_HR+ExAngina+Oldpeak+Slope+CA+Thal,
          family = binomial(logit))
summary(model)
# From the Coefficient table we observe that for Age,RestBP,Cholesteral,FBP,RestECG,Slope the p value is >0.05.
# so we can check through ANOVA test, which tells which of the variables are significant.

anova(model,test='Chisq')
# From the ANOVA test we observe that only for Cholesteral,FBP,RestECG,Slope the p value is >0.05.
# Hence thses must be removed from the model to make the model significant. Therefore the new model becomes:

new_model=glm(factor(Result)~Age+Sex+CP+RestBP+Max_HR+ExAngina+Oldpeak+CA+Thal,family = binomial(logit))
summary(new_model)
# From the Coefficient table we observe that for Age, RestBP the p values are >0.05. Here the p value for RestBP 
# is close to 0.05, so we can check through ANOVA which tells which of the variables are significant.

anova(new_model,test='Chisq')
# From the ANOVA test we observe that none of the explanatory variables are having p values >0.05. 
# Hence the model is significant to proceed further.


# Model Significance

# To check whether our model is significant or not, we need to first create a null model and compare it 
# with our new model.

null_model=glm(factor(Result)~1,family=binomial(logit))
anova(null_model,new_model,test='Chisq')
# From the above ANOVA result of null model we observe that the p value is <0.05 which implies that our model is 
# superior to the null model. Hence, we can conclude that the model is significant.


# Model Accuracy

# Predicting probability
pred=predict(new_model,type='response')

# Predicting Class
pred_class=ifelse(pred>0.5,'1','0') 
result=cbind(Heart_Disease_Data,pred,pred_class)

# Computing actual Vs Predicted Matrix
mytable=table(factor(Result),pred_class)
mytable

# calculation of accuracy -> actual Vs Predicted
round(prop.table(mytable)*100,2)

# accuracy % = (35.31 + 48.84) = 84.15 %
# misclassification % = (10.23 + 5.61) = 15.84 %
# From the above obtained accuracy % we can conclude that our model is accurate to 84.15% 
# which is a very good response. Hence, we can conclude that the model is accurate.


# Model Generalizability

# Using LOOCV (Leave one out cross validation)
library(boot)
LOOCV=cv.glm(Heart_Disease_Data,new_model)
LOOCV_MSE=LOOCV$delta[1]      # misclassification error of LOOCV
LOOCV_MSE

# misclassification % = 0.1248 * 100 = 12.48 %
# accuracy % = (100-12.48) = 87.52 %

# k fold cross validation (k=10)
set.seed(1)
k_fold=cv.glm(Heart_Disease_Data,new_model,K=10)
k_fold_MCE=k_fold$delta[1]
k_fold_MCE 

# misclassification % = 0.1235 * 100 = 12.35 %
# accuracy % = (100-12.35) = 87.65 %

# From the obtained values of accuracy% & mis-classification% for original data and LOOCV data, k fold
# cross validation data we notice that there is a slight deterioration in model performance as the accuracy%
# is slightly increased and mis-classification% is slightly decreased compared to original data values.

# Therefore, we can conclude that if we use this model around 87% it will be correctly predicted and around 12% 
# it will be wrongly predicted. If 87% accuracy is acceptable with us then we can use this model otherwise we 
# check for other models. Hence it shows that the model is good and the model can be generalizable.


# Predicting Using Confusion Matrix

mytable

# Sensitivity / Recall = TP / (TP + FN) =
148/(148+17)                     # ~ 0.90

# Specificity = TN / (TN + FP) = 
107/(107+31)                     # ~ 0.78

# Precision = TP / (TP + FP) = 
148/(148+31)                     # ~ 0.83

# F - measure = 2 *Precision * Recall / (Precision + Recall) =
2*0.83*0.90/(0.83+0.90)          # ~ 0.86

# From the obtained values of Sensitivity, Specificity, Precision and F-measure we observe that all the values 
# close to each other and fall in the range of 80 - 90% which implies that our model is capable of predicting 
# atmost 90% of the cases. Hence, we can conclude that our model is equally good in predicting heart disease 
# (positive) and not having heart disease (negative) cases.