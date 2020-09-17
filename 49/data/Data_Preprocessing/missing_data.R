# Data Preprocessing

# Importing the dataset
dataset = read.csv('Data.csv')

# Taking care of missing data
dataset$Age[is.na(dataset$Age)] = mean(dataset$Age, na.rm = T)
dataset$Salary[is.na(dataset$Salary)] = mean(dataset$Salary, na.rm = T)