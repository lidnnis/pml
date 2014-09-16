library(caret)
library(corrplot)
library(randomForest)

training_data = read.csv("./pml-training.csv")
testing_data = read.csv("./pml-testing.csv")


#STEP 1: Preprocess/Clean the data

#Remove columns that have no learning significance
training_cleaned = training_data[,-(1:7)]

#Remove columns that have low variance
nsv = nearZeroVar(training_cleaned,saveMetrics=TRUE)
nsv_list = nsv$nzv
training_cleaned = training_cleaned[,!nsv_list]

#colSums(is.na(new_training)) shows that columns with NA, mostly have NA, and therefore can be removed
training_cleaned = training_cleaned[,colSums(is.na(training_cleaned)) == 0]

#In order to predict on the testing set, we must also remove the same columns
#from the testing set that we removed from the training set
testing = testing_data[,which(names(testing_data)%in%names(training_cleaned))]

#STEP 2: Partition the data (70% training, 30% CV)

inTrain = createDataPartition(y = training_cleaned$classe, p = 0.7, list=FALSE)
training = training_cleaned[inTrain, ]
crossvalidation = training_cleaned[-inTrain, ]

#Plot a correlation matrix
cm = cor(training[, -length(training)])
corrplot(cm, order = "FPC", method = "circle", type = "lower", tl.cex = 0.8,  tl.col = rgb(0, 0, 0))

#STEP 3: Fit a model with the random forest classifier
modFit = randomForest(y=training$classe,x=training[-54],ntree=300,do.trace=TRUE)


#STEP 4: Predict on the CV set
pred_cv = predict(modFit,crossvalidation)


```{confusionMatrix(crossval$classe,pred_cv)}

```


#STEP 5: Predict on the testing set
pred_test = predict(modFit,testing)

#Write predictions to file
pml_write_files = function(x)
{
	n = length(x)
	for (i in 1:n)
	{
		filename = paste0("problem_id_",i,".txt")
		write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
	}
}

pml_write_files(pred_test)