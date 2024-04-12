# Mapping waste piles from drone imagery

# Libraries
required_packages <- c("sf", "dplyr", "caret", "tictoc", "rsample")
for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, dependencies = TRUE)
  }
  library(pkg, character.only = TRUE)
}

# Load segments derived from drone imagery

reflectance <- read_sf("inputs/md_data.shp")
raw <- read_sf("inputs/segments.shp")

# Creating the binary variable
format_variables <- function(segments) {
  segments$class <- as.factor(segments$class)
  segments$bin <- segments$class
  levels(segments$bin)[levels(segments$bin) %in% c('1', '2', '5', '6', '7')] <- '0'
  levels(segments$bin)[levels(segments$bin) == '4'] <- '1'
  return(segments)
}
reflectance <- format_variables(reflectance)

################################################################################
# Model development
################################################################################
set.seed(42)

# Data partitioning
reflectance$class = as.factor(reflectance$class)
data_split_multi <- initial_split(reflectance, prop = 0.8, strata = class)
train_data_multi <- training(data_split_multi)
test_data_multi <- testing(data_split_multi)

# assigning the same dataset to be used for training binary classifier

train_data_bin <-train_data_multi 
test_data_bin <- test_data_multi 

# Function to train and evaluate models multi-class models
train_and_evaluate <- function(train_data, test_data, model_type) {
  tic()
  model <- train(class ~ redmean + redmajorit + greenmean + greenmajor + bluemean + 
                   bluemajori + energymean + energymajo + entropymea + entropymaj + 
                   cormean + cormajorit + idmmean + idmmajorit + inertiamea + 
                   inertiamaj + CSmean + CSmajority + CPmean + CPmajority + 
                   HarCormean + HarCormajo, 
                 data = train_data, 
                 method = model_type,
                 trControl = trainControl(method = "repeatedcv", 
                                          number = 5, 
                                          repeats = 5,
                                          search = "grid"),
                 verbose = FALSE)
  toc()
  
  # Predictions
  test_data$pred <- predict(model, test_data)
  
  # Confusion matrix
  confusion <- confusionMatrix(reference = test_data$class,
                               data = test_data$pred,
                               mode = "everything")
  
  return(list(model = model, confusion = confusion))
}

# Same but for binary models
train_and_evaluate_bin <- function(train_data, test_data, model_type) {
  tic()
  model <- train(bin ~ redmean + redmajorit + greenmean + greenmajor + bluemean + 
                   bluemajori + energymean + energymajo + entropymea + entropymaj + 
                   cormean + cormajorit + idmmean + idmmajorit + inertiamea + 
                   inertiamaj + CSmean + CSmajority + CPmean + CPmajority + 
                   HarCormean + HarCormajo, 
                 data = train_data, 
                 method = model_type,
                 trControl = trainControl(method = "repeatedcv", 
                                          number = 5, 
                                          repeats = 5,
                                          search = "grid"),
                 verbose = FALSE)
  toc()
  
  # Predictions
  test_data$pred <- predict(model, test_data)
  
  # Confusion matrix
  confusion <- confusionMatrix(reference = test_data$bin,
                               data = test_data$pred,
                               mode = "everything")
  
  return(list(model = model, confusion = confusion))
}

# Multi-class models
rf_multi <- train_and_evaluate(train_data_multi, test_data_multi, "rf")
ann_multi <- train_and_evaluate(train_data_multi, test_data_multi, "nnet")
nb_multi <- train_and_evaluate(train_data_multi, test_data_multi, "nb")
svm_multi <- train_and_evaluate(train_data_multi, test_data_multi, "svmLinear")

# binary models
rf_bin <- train_and_evaluate_bin(train_data_bin, test_data_bin, "rf")
ann_bin <- train_and_evaluate_bin(train_data_bin, test_data_bin, "nnet")
nb_bin <- train_and_evaluate_bin(train_data_bin, test_data_bin, "nb")
svm_bin <- train_and_evaluate_bin(train_data_bin, test_data_bin, "svmLinear")

# Function to display confusion matrix
display_confusion_matrix <- function(conf_matrix) {
  print(conf_matrix)
}

# confusion matrices for multi-class models
display_confusion_matrix(rf_multi$confusion)
display_confusion_matrix(ann_multi$confusion)
display_confusion_matrix(nb_multi$confusion)
display_confusion_matrix(svm_multi$confusion)

# confusion matrices for binary models
display_confusion_matrix(rf_bin$confusion)
display_confusion_matrix(ann_bin$confusion)
display_confusion_matrix(nb_bin$confusion)
display_confusion_matrix(svm_bin$confusion)

