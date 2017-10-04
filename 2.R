install.packages("drat", repos="https://cran.rstudio.com")
drat:::addRepo("dmlc")
install.packages("xgboost", repos="http://dmlc.ml/drat/", type = "source")
data(agaricus.train, package='xgboost')
data(agaricus.test, package='xgboost')
train <- agaricus.train
test <- agaricus.test
bstSparse <- xgboost(data = train$data, label = train$label, max.depth = 2, eta = 1, nthread = 2, nround = 2, objective = "binary:logistic")
bstDense <- xgboost(data = as.matrix(train$data), label = train$label, max.depth = 2, eta = 1, nthread = 2, nround = 2, objective = "binary:logistic")
dtrain <- xgb.DMatrix(data = train$data, label = train$label)
bstDMatrix <- xgboost(data = dtrain, max.depth = 2, eta = 1, nthread = 2, nround = 2, objective = "binary:logistic")
# verbose = 0, no message
bst <- xgboost(data = dtrain, max.depth = 2, eta = 1, nthread = 2, nround = 2, objective = "binary:logistic", verbose = 0)
# verbose = 1, print evaluation metric
bst <- xgboost(data = dtrain, max.depth = 2, eta = 1, nthread = 2, nround = 2, objective = "binary:logistic", verbose = 1)
# verbose = 2, also print information about tree
bst <- xgboost(data = dtrain, max.depth = 2, eta = 1, nthread = 2, nround = 2, objective = "binary:logistic", verbose = 2)
pred <- predict(bst, test$data)

# size of the prediction vector
print(length(pred))
# limit display of predictions to the first 10
print(head(pred))

prediction <- as.numeric(pred > 0.5)
print(head(prediction))

err <- mean(as.numeric(pred > 0.5) != test$label)
print(paste("test-error=", err))

dtrain <- xgb.DMatrix(data = train$data, label=train$label)
dtest <- xgb.DMatrix(data = test$data, label=test$label)

watchlist <- list(train=dtrain, test=dtest)

bst <- xgb.train(data=dtrain, max.depth=2, eta=1, nthread = 2, nround=2, watchlist=watchlist, objective = "binary:logistic")

bst <- xgb.train(data=dtrain, max.depth=2, eta=1, nthread = 2, nround=2, watchlist=watchlist, eval.metric = "error", eval.metric = "logloss", objective = "binary:logistic")

bst <- xgb.train(data=dtrain, booster = "gblinear", max.depth=2, nthread = 2, nround=2, watchlist=watchlist, eval.metric = "error", eval.metric = "logloss", objective = "binary:logistic")

xgb.DMatrix.save(dtrain, "dtrain.buffer")

# to load it in, simply call xgb.DMatrix
dtrain2 <- xgb.DMatrix("dtrain.buffer")

bst <- xgb.train(data=dtrain2, max.depth=2, eta=1, nthread = 2, nround=2, watchlist=watchlist, objective = "binary:logistic")

label = getinfo(dtest, "label")
pred <- predict(bst, dtest)
err <- as.numeric(sum(as.integer(pred > 0.5) != label))/length(label)
print(paste("test-error=", err))

importance_matrix <- xgb.importance(model = bst)
print(importance_matrix)
xgb.plot.importance(importance_matrix = importance_matrix)

xgb.dump(bst, with.stats = T)

xgb.plot.tree(model = bst)
