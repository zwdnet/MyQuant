# Polynomial Regression

# Importing the dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[, 2:3]

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
# library(caTools)
# set.seed(123)
# split = sample.split(dataset$Purchased, SplitRatio = 0.8)
# training_set = subset(dataset, split == TRUE)
# test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set[, 2:3] = scale(training_set[, 2:3])
# test_set[, 2:3] = scale(test_set[, 2:3])

# Fitting Linear Regression to the dataset
lin_reg=lm(formula = Salary~.,
           data=dataset)

# Fitting Polynomial Regression to the dataset
dataset$level2 = dataset$Level^2
dataset$level3 = dataset$Level^3
dataset$level4 = dataset$Level^4
poly_reg = lm(formula = Salary~.,
              data=dataset)

#Visualising the Linear Regression results
install.packages('ggplot2')
library(ggplot2)
ggplot() +
  geom_point(aes(x=dataset$Level,y=dataset$Salary),
             colour = 'red') +
  geom_line (aes (x=dataset$Level, y = predict (lin_reg, newdata=dataset)),
             colour = 'blue') +
  xlab('Level') +
  ylab('Salary')

#Visualising the Polynomial Regression results
ggplot() +
  geom_point(aes(x=dataset$Level,y=dataset$Salary),
             colour = 'red') +
  geom_line (aes (x=dataset$Level, y = predict (poly_reg, newdata=dataset)),
             colour = 'blue') +
  xlab('Level') +
  ylab('Salary')

#Predicting a new result with Linear Regression
y_pred=predict(lin_reg, data.frame(Level=6.5))

#Predicting a new result with Polynomial Regression
y_pred=predict(poly_reg, data.frame(Level=6.5,
                                    level2=6.5^2,
                                    level3=6.5^3,
                                    level4=6.5^4))
