##-----------------------------------------------------------------##
#                   Objective 3                                    ##    
##-----------------------------------------------------------------##

#-----------   Detection of Waste Piles  ---------------#

#-------------      Loading of data packages   ---------------------#

pacman::p_load(sf,       
               tibble,   
               rpart,    
               randomForest,    
               ggplot2,       
               caret,     
               tictoc,       
               varImp,     
               mlbench,      
               tidymodels,    
               rsample,       
               utils,
               cvms,
               dplyr,
               klaR
               
)

#-------------        Loading Segments        ---------------------#
reflectance <- read_sf("inputs/md_data.shp")  #-------Model Development Data  
raw <- read_sf("inputs/segments.shp")  #--- All data
as.vector(str(raw))

#-------------        Format the variables      ---------------------#
#-Helper function to format the variables

formatting <- function(segments){
  segments$class = as.factor(segments$class)
  #To support development of a binary classifier
  segments$bin = segments$class
  levels(segments$bin)[levels(segments$bin)=='1'] <- '0'
  levels(segments$bin)[levels(segments$bin)=='2'] <- '0'
  levels(segments$bin)[levels(segments$bin)=='5'] <- '0'
  levels(segments$bin)[levels(segments$bin)=='6'] <- '0'
  levels(segments$bin)[levels(segments$bin)=='7'] <- '0'
  levels(segments$bin)[levels(segments$bin)=='4'] <- '1'
  
  return(segments)
  #Dropping geometery has some setbacks
}
reflectance <- formatting(reflectance)


#------------------- Summarizing the extracted parameters ----------------------#

str(reflectance)

# Assuming your data frame is called 'reflectance'
reflectance2 <- st_drop_geometry(reflectance)


library(sf)
library(dplyr)
library(tidyr)


#---------------------------#

summary <- reflectance %>%
  group_by(class) %>%
  summarize(mean_energy_mode = mean(energymajo),
            mean_entropy_mean = mean(entropymea),
            mean_entropy_maj = mean(entropymaj),
            mean_cor_mean = mean(cormean),
            mean_cor_maj = mean(cormajorit),
            mean_idm_mean = mean(idmmean),
            mean_idm_maj = mean(idmmajorit),
            mean_inertia_mean = mean(inertiamea),
            mean_inertia_maj = mean(inertiamaj),
            mean_CS_mean = mean(CSmean),
            mean_CS_maj = mean(CSmajority),
            mean_CP_mean = mean(CPmean),
            mean_CP_maj = mean(CPmajority),
            mean_HarCol_mean = mean(HarCormean),
            mean_HarCol_maj = mean(HarCormajo))%>%
  pivot_longer(cols = starts_with("mean_"), 
               names_to = "test",
               values_to = "values")%>%
  pivot_wider(names_from = "class",
               values_from = "values")



#---------------- Generation of summary ----------------------------------------#

# Joining rows using rbind
joined_df <- rbind(variable1_summary, variable2_summary)

# Output
print(joined_df)


reflectance2 = st_set_geometry(reflectance, NULL)
#Remove ID and bin column 


summary.test2 <- joined_df %>%
  pivot_wider(names_from = "class",
              values_from =c("MeanValue", "se"))

#In progress

# Prevent NA's

## Remove unnecessary columns
# First coarse the data to a dataframe
reflectance2 = st_set_geometry(reflectance, NULL)
#Remove ID and bin column 
library(tidyverse)
#reflectance2 <- reflectance2 %>% select(-DN, -bin)
library(dplyr)

#reflectance3 <- select(reflectance2, -c(DN,bin))

#reflectance2 <- reflectance2 %>% 
 # select(-c(DN,bin))

drop <- c("DN","bin", "id")
reflectance2 = reflectance2[,!(names(reflectance2) %in% drop)]





#-----------------------------------------------------------------##
#-----------            MODEL DEVELOPMENT            ---------------#
#-----------------------------------------------------------------##


#We will develop four models namely: (1) Random forest; (2) Artificial Neural Network; (3) Naive Bayes classifier; and (4) Logistical regression

#-------           Data Partitioning and subseting          -------#

set.seed(42)

#---- Data for a multiclass model
data_split <- initial_split(reflectance, 
                            prop= 0.8, 
                            strata = class)

train_data <- training(data_split)   
test_data <- testing(data_split)

#---- Data for a binary
data_split_bin <- initial_split(reflectance, 
                            prop= 0.8, 
                            strata = bin)

train_data_bin <- training(data_split_bin)   
test_data_bin <- testing(data_split_bin)

#######################################################
# ------ Multiclass classification models    ---------#
#######################################################

#---- Random forest classifier 
#Define the parameters
#first we define a Cartesian grid
man_grid <- data.frame(mtry = seq(from = 1, to = 50, by = 2))

#---- Generating parameters that control how the models are created i.e Cross-Validation  
fitControl <- trainControl(method = "repeatedcv", 
                           number = 5, 
                           repeats = 5,
                           search = "grid") #Repeating 5 fold cross validation 5 times

#---- Model Training          

tic()
set.seed(42)

rf_model <- train(class ~ redmean+ redmajorit+ greenmean+ greenmajor+ bluemean+ 
                    bluemajori+ energymean+ energymajo+ entropymea+ entropymaj+ 
                    cormean+ cormajorit+ idmmean+ idmmajorit+ inertiamea+ 
                    inertiamaj+ CSmean+ CSmajority+ CPmean+ CPmajority+ 
                    HarCormean+ HarCormajo, 
                  data = train_data, 
                  method = "rf",
                  trControl= fitControl, 
                  verbose = FALSE,
                  tuneGrid = man_grid)
toc()

rf_model

#---- Plot tuning process

plot(rf_model)
print(rf_model)

#---- Using the model on test data  

test_data$pred <- predict(rf_model, test_data)
test_data$predrf <- predict(rf_model, test_data)
#test_data$RFCertainity <- test_data$predrf == test_data$class
#-----   Confusion Matrix  

confusion <- table(factor(test_data$class), factor(test_data$pred))
conf.matr.rf.multi <- confusionMatrix(reference = test_data$class,
                             data = test_data$pred,
                             mode ="everything")
conf.matr.rf.multi

####
raw$nsima <- predict(rf_model, raw)
write_sf(raw, "outputs/visualizations/test3.shp")


raw$area2 = st_area(raw)

summary_data <- raw %>%
  group_by(nsima) %>%
  summarize(total_area = sum(area2))










#------------predictions -----------------------------#

#randomforest
rf.data = raw
rf.data$pred <- predict(rf_model, rf.data)
st_write(rf.data, "outputs/errors/MC/RF/MCRF.shp")

#---- Trying to understand where the model is having problems

#Filter for class5

newdata <- test_data[ which(test_data$class==5), ]
newdata$FN <- newdata$pred == newdata$class #Here a 0 will a false negative
st_write(newdata, "outputs/errors/MC/RF/FNerror.shp")

newdata <- test_data[ which(test_data$pred==5), ] #These are predicted as waste but they are not
newdata$FP <- newdata$pred != newdata$class  #Here a 0 will a false negative
st_write(newdata, "outputs/errors/MC/RF/FPerror.shp")


###################################

#---- Artificial Neural Network

#---- Random forest classifier 
#Define the parameters
#first we define a Cartesian grid
man_grid <- data.frame(mtry = seq(from = 1, to = 50, by = 2))

#---- Generating parameters that control how the models are created i.e Cross-Validation  
fitControl <- trainControl(method = "repeatedcv", 
                           number = 5, 
                           repeats = 5,
                           search = "grid") #Repeating 5 fold cross validation 5 times

#---- Model Training          

tic()
set.seed(42)

ann.model.multi <- train(class ~ redmean+ redmajorit+ greenmean+ greenmajor+ bluemean+ 
                    bluemajori+ energymean+ energymajo+ entropymea+ entropymaj+ 
                    cormean+ cormajorit+ idmmean+ idmmajorit+ inertiamea+ 
                    inertiamaj+ CSmean+ CSmajority+ CPmean+ CPmajority+ 
                    HarCormean+ HarCormajo, 
                  data = train_data, 
                  method = "nnet",
                  trControl= fitControl, 
                  verbose = FALSE)
toc()

ann.model.multi

#---- Plot tuning process

plot(ann.model.multi)
print(ann.model.multi)

#---- Using the model on test data  

test_data$pred.ann <- predict(ann.model.multi, test_data)
test_data$Error <- test_data$pred.ann == test_data$class
#-----   Confusion Matrix  

confusion <- table(factor(test_data$class), factor(test_data$pred.ann))
conf.matr.ann.multi <- confusionMatrix(reference = test_data$class,
                             data = test_data$pred.ann,
                             mode ="everything")
conf.matr.ann.multi

#Filter for class5

newdata <- test_data[ which(test_data$class==5), ]
newdata$FP <- newdata$pred.ann == newdata$class #Here a 0 will a false negative
st_write(newdata, "outputs/errors/ANN_FPMCtest.shp")

newdata <- test_data[ which(test_data$pred.ann==5), ]
newdata$FN <- newdata$pred.ann == newdata$class  #Here a 0 will a false negative
st_write(newdata, "outputs/errors/ANN_FNMCtest.shp")


st_write(test_data, "outputs/errors/ANN_MCtest.shp")

#---- Naive Bayes classifier 
#Define the parameters
#first we define a Cartesian grid
man_grid <- data.frame(FL = seq(from = 1, to = 50, by = 2), usekernel = TRUE, adjust = seq(from = 1, to = 50, by = 2))

#---- Generating parameters that control how the models are created i.e Cross-Validation  
fitControl <- trainControl(method = "repeatedcv", 
                           number = 5, 
                           repeats = 5,
                           search = "grid") #Repeating 5 fold cross validation 5 times

#---- Model Training          

tic()
set.seed(42)

nb_model <- train(class ~ redmean+ redmajorit+ greenmean+ greenmajor+ bluemean+ 
                    bluemajori+ energymean+ energymajo+ entropymea+ entropymaj+ 
                    cormean+ cormajorit+ idmmean+ idmmajorit+ inertiamea+ 
                    inertiamaj+ CSmean+ CSmajority+ CPmean+ CPmajority+ 
                    HarCormean+ HarCormajo, 
                  data = train_data, 
                  method = "nb",
                  trControl= fitControl, 
                  verbose = FALSE)
#Find a way of incoroprating hyperparameter tuning i.e. (tunegrid)
toc()

nb_model

#---- Plot tuning process

plot(nb_model)
print(nb_model)

#---- Using the model on test data  

test_data$nb.pred <- predict(nb_model, test_data)
test_data$Error <- test_data$nb.pred == test_data$class

#-----   Confusion Matrix  

confusion.nb <- table(factor(test_data$class), factor(test_data$nb.pred))
conf.matr.nb <- confusionMatrix(reference = test_data$class,
                             data = test_data$nb.pred,
                             mode ="everything")
conf.matr.nb

st_write(test_data, "outputs/errors/NB_MCtest.shp")

#---- Logistical regression classifier
#Define the parameters
#first we define a Cartesian grid
man_grid <- data.frame(mtry = seq(from = 1, to = 50, by = 2))

#---- Generating parameters that control how the models are created i.e Cross-Validation  
fitControl <- trainControl(method = "repeatedcv", 
                           number = 5, 
                           repeats = 5,
                           search = "grid") #Repeating 5 fold cross validation 5 times

#---- Model Training          

tic()
set.seed(42)

svm_model <- train(class ~ redmean+ redmajorit+ greenmean+ greenmajor+ bluemean+ 
                    bluemajori+ energymean+ energymajo+ entropymea+ entropymaj+ 
                    cormean+ cormajorit+ idmmean+ idmmajorit+ inertiamea+ 
                    inertiamaj+ CSmean+ CSmajority+ CPmean+ CPmajority+ 
                    HarCormean+ HarCormajo, 
                  data = train_data, 
                  method = "svmLinear",
                  trControl= fitControl, 
                  verbose = FALSE)
toc()

svm_model


#---- Using the model on test data  

test_data$svm.pred <- predict(svm_model, test_data)
test_data$Error <- test_data$svm.pred == test_data$class

#-----   Confusion Matrix  

confusion.svm <- table(factor(test_data$class), factor(test_data$svm.pred))
conf.matr.svm <- confusionMatrix(reference = test_data$class,
                             data = test_data$svm.pred,
                             mode ="everything")
conf.matr.svm
st_write(test_data, "outputs/errors/SVM_MCtest.shp")



#RF

rf_model
conf.matr.rf.multi

#ANN

ann.model.multi
conf.matr.ann.multi

#Naive Bayes

nb_model
conf.matr.nb

#SVM 
svm_model
conf.matr.svm


###############
# Colineality - mean and mode of the same variable seems redundant
# Class imbalance
#-----------------Class inbalance

count_train <- table(train_data$class)
prop_1_train <- count_train[1]/
  sum(count_train)
prop_1_train

#####



#---- Data for a multiclass model
data_split <- initial_split(reflectance, 
                            prop= 0.8, 
                            strata = class)

train_data <- training(data_split)   
test_data <- testing(data_split)

#---- Data for a binary
data_split_bin <- initial_split(reflectance, 
                                prop= 0.8, 
                                strata = bin)

train_data_bin <- training(data_split_bin)   
test_data_bin <- testing(data_split_bin)

#######################################################
# ------ Binary classification models    ---------#
#######################################################

#---- Random forest classifier 
#Define the parameters
#first we define a Cartesian grid
man_grid <- data.frame(mtry = seq(from = 1, to = 50, by = 2))

#---- Generating parameters that control how the models are created i.e Cross-Validation  
fitControl <- trainControl(method = "repeatedcv", 
                           number = 5, 
                           repeats = 5,
                           search = "grid") #Repeating 5 fold cross validation 5 times

result <- table(test_data_bin$bin)

print(result)

#---- Model Training          

tic()
set.seed(42)

rf.model.bin <- train(bin ~ redmean+ redmajorit+ greenmean+ greenmajor+ bluemean+ 
                    bluemajori+ energymean+ energymajo+ entropymea+ entropymaj+ 
                    cormean+ cormajorit+ idmmean+ idmmajorit+ inertiamea+ 
                    inertiamaj+ CSmean+ CSmajority+ CPmean+ CPmajority+ 
                    HarCormean+ HarCormajo, 
                  data = train_data_bin, 
                  method = "rf",
                  trControl= fitControl, 
                  verbose = FALSE,
                  tuneGrid = man_grid)
toc()

rf.model.bin

#---- Plot tuning process

plot(rf.model.bin)
print(rf.model.bin)

#---- Using the model on test data  

test_data_bin$pred <- predict(rf.model.bin, test_data_bin)

raw$pred_fin <- predict(rf.model.bin, raw)

raw$area = st_area(raw)

write_sf(raw, "outputs/visualizations/test2.shp")

summary_data <- raw %>%
  group_by(pred_fin) %>%
  summarize(total_area = sum(area))



#-----   Confusion Matrix  

confusion <- table(factor(test_data_bin$bin), factor(test_data_bin$pred))
conf.matr.rf.bin <- confusionMatrix(reference = test_data_bin$bin,
                                      data = test_data_bin$pred,
                                      mode ="everything")
conf.matr.rf.bin

#------------predictions -----------------------------#

#randomforest
rf.data = raw
rf.data$pred <- predict(rf_model, rf.data)
st_write(rf.data, "outputs/errors/binary/RF/BRF.shp")



###################################

#---- Artificial Neural Network

#---- Random forest classifier 
#Define the parameters
#first we define a Cartesian grid
man_grid <- data.frame(mtry = seq(from = 1, to = 50, by = 2))

#---- Generating parameters that control how the models are created i.e Cross-Validation  
fitControl <- trainControl(method = "repeatedcv", 
                           number = 5, 
                           repeats = 5,
                           search = "grid") #Repeating 5 fold cross validation 5 times

#---- Model Training          

tic()
set.seed(42)

ann.model.bin <- train(bin ~ redmean+ redmajorit+ greenmean+ greenmajor+ bluemean+ 
                           bluemajori+ energymean+ energymajo+ entropymea+ entropymaj+ 
                           cormean+ cormajorit+ idmmean+ idmmajorit+ inertiamea+ 
                           inertiamaj+ CSmean+ CSmajority+ CPmean+ CPmajority+ 
                           HarCormean+ HarCormajo, 
                         data = train_data_bin, 
                         method = "nnet",
                         trControl= fitControl, 
                         verbose = FALSE)
toc()

ann.model.bin

#---- Plot tuning process

plot(ann.model.bin)
print(ann.model.bin)

#---- Using the model on test data  

test_data_bin$pred.ann <- predict(ann.model.bin, test_data_bin)

#-----   Confusion Matrix  

confusion <- table(factor(test_data_bin$bin), factor(test_data_bin$pred.ann))
conf.matr.ann.bin <- confusionMatrix(reference = test_data_bin$bin,
                                       data = test_data_bin$pred.ann,
                                       mode ="everything")
conf.matr.ann.bin


#---- Naive Bayes classifier 
#Define the parameters
#first we define a Cartesian grid
man_grid <- data.frame(FL = seq(from = 1, to = 50, by = 2), usekernel = TRUE, adjust = seq(from = 1, to = 50, by = 2))

#---- Generating parameters that control how the models are created i.e Cross-Validation  
fitControl <- trainControl(method = "repeatedcv", 
                           number = 5, 
                           repeats = 5,
                           search = "grid") #Repeating 5 fold cross validation 5 times

#---- Model Training          

tic()
set.seed(42)

nb_model.bin <- train(bin ~ redmean+ redmajorit+ greenmean+ greenmajor+ bluemean+ 
                    bluemajori+ energymean+ energymajo+ entropymea+ entropymaj+ 
                    cormean+ cormajorit+ idmmean+ idmmajorit+ inertiamea+ 
                    inertiamaj+ CSmean+ CSmajority+ CPmean+ CPmajority+ 
                    HarCormean+ HarCormajo, 
                  data = train_data_bin, 
                  method = "nb",
                  trControl= fitControl, 
                  verbose = FALSE)
#Find a way of incoroprating hyperparameter tuning i.e. (tunegrid)
toc()

nb_model.bin

#---- Plot tuning process

plot(nb_model.bin)
print(nb_model.bin)

#---- Using the model on test data  

test_data_bin$nb.pred <- predict(nb_model.bin, test_data_bin)

#-----   Confusion Matrix  

confusion.nb.bin <- table(factor(test_data_bin$bin), factor(test_data_bin$nb.pred))
conf.matr.nb.bin <- confusionMatrix(reference = test_data_bin$bin,
                                data = test_data_bin$nb.pred,
                                mode ="everything")
conf.matr.nb.bin

st_write(test_data, "outputs/errors/NB_MCtest.shp")

#---- Logistical regression classifier
#Define the parameters
#first we define a Cartesian grid
man_grid <- data.frame(mtry = seq(from = 1, to = 50, by = 2))

#---- Generating parameters that control how the models are created i.e Cross-Validation  
fitControl <- trainControl(method = "repeatedcv", 
                           number = 5, 
                           repeats = 5,
                           search = "grid") #Repeating 5 fold cross validation 5 times

#---- Model Training          

tic()
set.seed(42)

svm_model.bin <- train(bin ~ redmean+ redmajorit+ greenmean+ greenmajor+ bluemean+ 
                     bluemajori+ energymean+ energymajo+ entropymea+ entropymaj+ 
                     cormean+ cormajorit+ idmmean+ idmmajorit+ inertiamea+ 
                     inertiamaj+ CSmean+ CSmajority+ CPmean+ CPmajority+ 
                     HarCormean+ HarCormajo, 
                   data = train_data_bin, 
                   method = "svmLinear",
                   trControl= fitControl, 
                   verbose = FALSE)
toc()

svm_model


#---- Using the model on test data  

test_data_bin$svm.pred <- predict(svm_model.bin, test_data_bin)


#-----   Confusion Matrix  

confusion.svm.bin <- table(factor(test_data_bin$bin), factor(test_data_bin$svm.pred))
conf.matr.svm.bin <- confusionMatrix(reference = test_data_bin$bin,
                                 data = test_data_bin$svm.pred,
                                 mode ="everything")
conf.matr.svm.bin
st_write(test_data, "outputs/errors/SVM_MCtest.shp")



#RF

conf.matr.rf.bin

#ANN
conf.matr.ann.bin

#Naive Bayes

nb_model
conf.matr.nb.bin

#SVM 
svm_model
conf.matr.svm.bin


# SUmmarizing the perfomance

# Join the confusion matrices into a table automatically










#---- Trying to understand where the model is having problems

#Filter for class5

newdata <- test_data[ which(test_data$class==5), ]
newdata$FN <- newdata$pred == newdata$class #Here a 0 will a false negative
st_write(newdata, "outputs/errors/MC/RF/FNerror.shp")

newdata <- test_data[ which(test_data$pred==5), ] #These are predicted as waste but they are not
newdata$FP <- newdata$pred != newdata$class  #Here a 0 will a false negative
#st_write(newdata, "outputs/errors/MC/RF/FPerror.shp")





