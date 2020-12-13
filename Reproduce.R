    library(reticulate)
    use_condaenv('r-reticulate', conda = "/Users/james/programs/anaconda3/bin/anaconda")
    library(tensorflow)
    library(keras)
    library(plyr)
    library(neuralnet)
    library(DMwR)
    library(UBL)
    tf$constant("Hellow Tensorflow")
  
    
    all <- read.table("./drug_consumption.data", header = FALSE, sep = ",")
    colnames(all) <- c("ID","Age","Gender","Education","Country",
                       "Ethnicity","Neuroticism","Extraversion","Openness","Agreeableness",
                       "Conscientiousness","Impulsiveness","Sensation","Alcohol",
                       "Amphet","Amyl","Benzos","Caff","Cannabis","Choc","Coke",
                       "Crack","Ecstasy","Heroin","Ketamine","Legalh","LSD","Meth",
                       "Mushrooms","Nicotine","Semer","VSA")
    
    
    # (20,10,3,2,1,1)
    
    my_model = function(k.folds, df, drug) {
      
      data = df[, c(2:13, grep(drug, colnames(df)))]
      data[, drug] = factor(df[, drug])
      
      #shuffle the data for the k-fold cross validation
      shuffled.data <- data[sample(nrow(data)),]
      #transform the data
      # shuffled.data = shuffled.data[shuffled.data[ ,13]!="CL0", ]
      shuffled.data[,13]=revalue(shuffled.data[,13], c(
        "CL0"=0,  # Never used
        "CL1"=1,  # Used over a Decade Ago
        "CL2"=1,  # Used in Last Decade
        "CL3"=1,  # Used in Last Year
        "CL4"=1,  # Used in Last Month
        "CL5"=1,  # Used in Last Week
        "CL6"=1)) # Used in Last Day
      
      print(nrow(shuffled.data[shuffled.data[,13]==1,])/
              (nrow(shuffled.data[shuffled.data[,13]==0,])+nrow(shuffled.data[shuffled.data[,13]==1,])))
      
      shuffled.data[, drug] = factor(shuffled.data[, drug])
      folds <- cut(seq(1,nrow(shuffled.data)), breaks = k.folds, labels = FALSE)
      error.cv <- 0
      for(i in 1:k.folds){
        testIndexes <- which(folds==i,arr.ind=TRUE)
        
        #oversample the data for training 
        train <- shuffled.data[-testIndexes, ]
        # validationIndexes <- valid
        # print(table(train$Alcohol))
        # print(head(train))
        # train <- SmoteClassif(as.formula(paste(drug, " ~ .")), train, C.perc = "balance")
        # print(table(train$Alcohol))
        # print(head(train))
        train <- SmoteClassif(as.formula(paste(drug, " ~ .")), train, C.perc = "balance")
        # train <- SMOTE(as.formula(paste(drug, " ~ .")), train, perc.over = 200, perc.under = 200)
        # print(table(train$Alcohol))
        # df1 = train[train[,13]==0, ]
        # df2 = train[train[,13]==1, ]
        # df3 = train[train[,13]==2, ]
        # df4 = train[train[,13]==3, ]
        # df5 = train[train[,13]==4, ]
        # df6 = train[train[,13]==5, ]
        # rows_num = c(nrow(df1),nrow(df2),nrow(df3),nrow(df4),nrow(df5),nrow(df6))
        # max_rows = max(rows_num)
        # print(rows_num)
        # print(max_rows)
        # classes = list(df1, df2, df3, df4, df5, df6)
        # for (j in 1:6) {
        #   times = floor(max_rows/rows_num[j]) - 1
        #   if (times/3 >= 1) {
        #     for (k in 1:times/3){
        #       train <- rbind(train, classes[[j]])
        #     }
        #   }
        # }

        # print(nrow(train[train[,13]==0, ]))
        # print(max_rows)
        # 
        #Keep the randomness
        train <- train[sample(nrow(train)),]
        # print(head(train))
        x_train = data.matrix(train[, 1:12])
        y_train = data.matrix(train[, 13])
        x_test = data.matrix(shuffled.data[testIndexes, 1:12])
        y_test = data.matrix(shuffled.data[testIndexes, 13])  
        
        # x_train = data.matrix(train[, 6:10])
        # x_test = data.matrix(shuffled.data[testIndexes, 6:10])
        # print("y_test before")
        # print(head(y_test))
        #change the classes to the one-hot vector
        y_train = to_categorical(y_train, 2)
        y_test = to_categorical(y_test, 2)
        # print("y_test after")
        # print(head(y_test))
        model <- keras_model_sequential() 
        model %>% 
          layer_dense(units = 40, activation = 'relu', input_shape = c(12)) %>% 
          layer_dense(units = 60, activation = 'relu') %>%
          layer_dense(units = 120, activation = 'relu') %>%
          layer_dropout(rate = 0.2) %>%
          layer_dense(units = 80, activation = 'relu') %>%
          # layer_dropout(rate = 0.2) %>%
          layer_dense(units = 40, activation = 'relu') %>%
          layer_dense(units = 20, activation = 'relu') %>%
          layer_dense(units = 2, activation = 'sigmoid')
        model %>% compile(
          loss = 'categorical_crossentropy',
          optimizer = optimizer_adagrad(1e-02),
          # optimizer = optimizer_sgd(lr = 1e-02),
          # optimizer = optimizer_adam(lr = 1e-03),
          # optimizer = optimizer_rmsprop(lr = 1e-03),
          metrics = c('accuracy')
        )
        
        history <- model %>% fit(
          x_train, y_train, 
          epochs = 30, batch_size = 16, 
          # validation_data = list(),
          # validation_split = 0.1,
          # class_weight = list("0" = 10, "1" = 1)
          # class_weight = list("0" = 1, "1" = 1, "2" = 1, "3" = 1, "4" = 2, "5" = 1)
        )
        model %>% evaluate(x_test, y_test)
        model %>% predict_classes(x_test)
        pred <- model %>% predict_classes(x_test)
        print(pred)
        real = shuffled.data[testIndexes, 13]
        lvls = union(levels(real), levels(pred))
        real = factor(real, levels=lvls)
        pred = factor(pred, levels=lvls)
        t  <- table(real, pred)
        if(i == 1) table.cv = t
        else table.cv <- table.cv + t
        error.cv <- error.cv + (1 - sum(diag(t))/length(shuffled.data[testIndexes, 13]))
        
        print("*******")
        print(table.cv)
      }
      print(paste('Cross-validation', 'Neural Network', 'Error:', (error.cv/k.folds)*100, '%', sep = ' '))
      print(paste('Cross-validation', 'Neural Network', 'Accurary:', (1- (error.cv/k.folds))*100, '%', sep = ' '))
      ret = table.cv / k.folds
      return(ret)
    }
    
    
    #Adam
    drug = "Alcohol"
    # drug = "Caff"
    # parameters = c()
    # for i in 
    my_model(10, all, drug)
    # table <- funmeth.mlp(all, drug)
    #table

      