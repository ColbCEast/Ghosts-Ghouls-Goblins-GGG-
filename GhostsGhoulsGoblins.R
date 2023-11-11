setwd("C:/Users/colby/OneDrive/Desktop/STAT 348/Ghosts-Ghouls-Goblins-GGG-")

library(tidymodels)
library(embed)
library(vroom)
library(themis)
library(rstanarm)

train_data_missing <- vroom("trainWithMissingValues.csv")

train_data <- vroom("train.csv")

test_data <- vroom("test.csv")

train_data <- train_data %>%
  mutate(type = as.factor(type))

ggplot(data = train_data,
       mapping = aes(x = hair_length,
                     fill = type)) +
  geom_boxplot()

ggplot(data = train_data,
       mapping = aes(x = bone_length,
                     fill = type)) +
  geom_boxplot()

ggplot(data = train_data,
       mapping = aes(x = has_soul,
                     fill = type)) +
  geom_boxplot()

the_recipe <- recipe(type~., data = train_data_missing) %>%
  step_impute_mean(all_numeric_predictors()) %>%
  step_impute_mode(all_nominal_predictors())

prepped <- prep(the_recipe)
baked <- bake(prepped, new_data = train_data_missing)

rmse_vec(train_data[is.na(train_data_missing)], baked[is.na(train_data_missing)])


# KNN Model

knn_recipe <- recipe(type~., data = train_data) %>%
  step_mutate_at(color, fn = factor) %>%
  step_lencode_glm(color, outcome = vars(type)) %>%
  step_center(all_numeric_predictors()) %>%
  step_scale(all_numeric_predictors())

prepped <- prep(knn_recipe)
bake(prepped, new_data = train_data)

knn_model <- nearest_neighbor(neighbors = tune()) %>%
  set_mode("classification") %>%
  set_engine("kknn")

knn_wf <- workflow() %>%
  add_recipe(knn_recipe) %>%
  add_model(knn_model)

tuning_grid_knn <- grid_regular(neighbors(),
                                levels = 5)

folds <- vfold_cv(train_data, v = 5, repeats = 1)

CV_results <- knn_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid_knn,
            metrics = metric_set(accuracy))

best_tune_knn <- CV_results %>%
  select_best("accuracy")

final_knn_wf <- knn_wf %>%
  finalize_workflow(best_tune_knn) %>%
  fit(data = train_data)

knn_preds <- predict(final_knn_wf,
                     new_data = test_data,
                     type = "class") %>%
  mutate(type = .pred_class) %>%
  bind_cols(., test_data) %>%
  select(id, type) %>%
  rename(Id = id)

vroom_write(knn_preds, file = "KNNPreds.csv", delim = ",")


# Neural Networks

train_data <- vroom("train.csv")

test_data <- vroom("test.csv")

train_data <- train_data %>%
  mutate(type = as.factor(type))

nn_recipe <- recipe(type~., data = train_data) %>%
  update_role(id, new_role = "id") %>%
  step_mutate_at(color, fn = factor) %>%
  step_dummy(color) %>%
  step_range(all_numeric_predictors(), min = 0, max = 1)

nn_model <- mlp(hidden_units = tune(),
                epochs = 50) %>%
  set_engine("nnet", verbose = 0) %>%
  set_mode("classification")

nn_wf <- workflow() %>%
  add_recipe(nn_recipe) %>%
  add_model(nn_model)

nn_tune_grid <- grid_regular(hidden_units(range = c(1, 50)),
                             levels = 5)

folds <- vfold_cv(train_data, v = 5, repeats = 1)

tuned_nn <- nn_wf %>%
  tune_grid(resamples = folds,
            grid = nn_tune_grid,
            metrics = metric_set(accuracy))

best_tune_nn <- tuned_nn %>%
  select_best("accuracy")

final_nn_wf <- nn_wf %>%
  finalize_workflow(best_tune_nn) %>%
  fit(data = train_data)

nn_preds <- predict(final_nn_wf,
                     new_data = test_data,
                     type = "class") %>%
  mutate(type = .pred_class) %>%
  bind_cols(., test_data) %>%
  select(id, type) %>%
  rename(Id = id)

vroom_write(nn_preds, file = "NNPreds.csv", delim = ",")


# Boosted model

library(bonsai)
library(lightgbm)

train_data <- vroom("train.csv")

test_data <- vroom("test.csv")

train_data <- train_data %>%
  mutate(type = as.factor(type))

boost_recipe <- recipe(type~., data = train_data) %>%
  update_role(id, new_role = "id") %>%
  step_mutate_at(color, fn = factor) %>%
  step_dummy(color) %>%
  step_range(all_numeric_predictors(), min = 0, max = 1)

boost_model <- boost_tree(tree_depth = tune(),
                          trees = tune(),
                          learn_rate = tune()) %>%
  set_engine("lightgbm") %>%
  set_mode("classification")

boost_wf <- workflow() %>%
  add_recipe(boost_recipe) %>%
  add_model(boost_model)

boost_tune_grid <- grid_regular(tree_depth(),
                          trees(),
                          learn_rate(),
                          levels = 5)

folds <- vfold_cv(data = train_data, v = 5, repeats = 1)

tuned_boost <- boost_wf %>%
  tune_grid(resamples = folds,
            grid = boost_tune_grid,
            metrics = metric_set(accuracy))

best_tune_boost <- tuned_boost %>%
  select_best("accuracy")

final_boost_wf <- boost_wf %>%
  finalize_workflow(best_tune_boost) %>%
  fit(data = train_data)

boost_preds <- predict(final_boost_wf,
                    new_data = test_data,
                    type = "class") %>%
  mutate(type = .pred_class) %>%
  bind_cols(., test_data) %>%
  select(id, type) %>%
  rename(Id = id)

vroom_write(boost_preds, file = "BoostPreds.csv", delim = ",")

# BART Model

library(dbarts)

train_data <-  vroom("train.csv")

test_data <- vroom("test.csv")

train_data <- train_data %>%
  mutate(type = as.factor(type))

bart_recipe <- recipe(type~., data = train_data) %>%
  update_role(id, new_role = "id") %>%
  step_mutate_at(color, fn = factor) %>%
  step_dummy(color) %>%
  step_range(all_numeric_predictors(), min = 0, max = 1)

bart_model <- bart(trees = tune()) %>%
  set_engine("dbarts") %>%
  set_mode("classification")

bart_wf <- workflow() %>%
  add_recipe(bart_recipe) %>%
  add_model(bart_model)

bart_tune_grid <- grid_regular(trees(),
                               levels = 5)

folds <- vfold_cv(data = train_data, v = 5, repeats = 1)

tuned_bart <- bart_wf %>%
  tune_grid(resamples = folds,
            grid = bart_tune_grid,
            metrics = metric_set(accuracy))

best_tune_bart <- tuned_bart %>%
  select_best("accuracy")

final_bart_wf <- bart_wf %>%
  finalize_workflow(best_tune_bart) %>%
  fit(data = train_data)

bart_preds <- predict(final_bart_wf,
                       new_data = test_data,
                       type = "class") %>%
  mutate(type = .pred_class) %>%
  bind_cols(., test_data) %>%
  select(id, type) %>%
  rename(Id = id)

vroom_write(bart_preds, file = "BartPreds.csv", delim = ",")

## Random Forests, PCA threshold 0.8

library(ranger)

train_data <-  vroom("train.csv")

test_data <- vroom("test.csv")

train_data <- train_data %>%
  mutate(type = as.factor(type))

rf_recipe <- recipe(type~., data = train_data) %>%
  update_role(id, new_role = "id") %>%
  step_mutate_at(color, fn = factor) %>%
  step_dummy(color) %>%
  step_pca(all_numeric_predictors(), threshold = 0.8)

rf_model <- rand_forest(mtry = tune(),
                        min_n = tune(),
                        trees = 1000) %>%
  set_engine("ranger") %>%
  set_mode("classification")

rf_wf <- workflow() %>%
  add_recipe(rf_recipe) %>%
  add_model(rf_model)

rf_tune_grid <- grid_regular(mtry(range = c(1,10)),
                             min_n(),
                             levels = 5)

folds <- vfold_cv(data = train_data, v = 5, repeats = 1)

cv_results <- rf_wf %>%
  tune_grid(resamples = folds,
            grid = rf_tune_grid,
            metrics = metric_set(accuracy))

best_tune_rf <- cv_results %>%
  select_best("accuracy")

final_rf_wf <- rf_wf %>%
  finalize_workflow(best_tune_rf) %>%
  fit(data = train_data)

rf_preds <- predict(final_rf_wf,
                    new_data = test_data,
                    type = "class") %>%
  mutate(type = .pred_class) %>%
  bind_cols(., test_data) %>%
  select(id, type) %>%
  rename(Id = id)

vroom_write(rf_preds, file = "RF_preds.csv", delim = ",")


## Naive Bayes, Target Encoding

library(naivebayes)
library(discrim)

train_data <-  vroom("train.csv")

test_data <- vroom("test.csv")

train_data <- train_data %>%
  mutate(type = as.factor(type))

nb_recipe <- recipe(type~., data = train_data) %>%
  update_role(id, new_role = "id") %>%
  step_mutate_at(color, fn = factor) %>%
  step_lencode_glm(color, outcome = vars(type)) %>%
  step_range(all_numeric_predictors(), min = 0, max = 1)

nb_model <- naive_Bayes(Laplace = tune(),
                        smoothness = tune()) %>%
  set_engine("naivebayes") %>%
  set_mode("classification")

nb_wf <- workflow() %>%
  add_recipe(nb_recipe) %>%
  add_model(nb_model)

nb_tuning_grid <- grid_regular(Laplace(),
                               smoothness(),
                               levels = 5)

folds <- vfold_cv(data = train_data, v = 10, repeats = 1)

cv_results <- nb_wf %>%
  tune_grid(resamples = folds,
            grid = nb_tuning_grid,
            metrics = metric_set(accuracy))

best_tune_nb <- cv_results %>%
  select_best("accuracy")

final_nb_wf <- nb_wf %>%
  finalize_workflow(best_tune_nb) %>%
  fit(data = train_data)

nb_preds <- predict(final_nb_wf,
                    new_data = test_data,
                    type = "class") %>%
  mutate(type = .pred_class) %>%
  bind_cols(., test_data) %>%
  select(id, type) %>%
  rename(Id = id)

vroom_write(nb_preds, file = "NB_Preds.csv", delim = ",")


## Support Vector Machines

library(kernlab)

train_data <-  vroom("train.csv")

test_data <- vroom("test.csv")

train_data <- train_data %>%
  mutate(type = as.factor(type))

svm_recipe <- recipe(type~., data = train_data) %>%
  update_role(id, new_role = "id") %>%
  step_mutate_at(color, fn = factor) %>%
  step_lencode_glm(color, outcome = vars(type)) %>%
  step_range(all_numeric_predictors(), min = 0, max = 1)

svm_model <- svm_linear(cost = tune()) %>%
  set_engine("kernlab") %>%
  set_mode("classification")

svm_wf <- workflow() %>%
  add_recipe(svm_recipe) %>%
  add_model(svm_model)

svm_tuning_grid <- grid_regular(cost(),
                                levels = 5)

folds <- vfold_cv(data = train_data, v = 5, repeats = 1)

cv_results <- svm_wf %>%
  tune_grid(resamples = folds,
            grid = svm_tuning_grid,
            metrics = metric_set(accuracy))

best_tune_svm <- cv_results %>%
  select_best("accuracy")

final_svm_wf <- svm_wf %>%
  finalize_workflow(best_tune_svm) %>%
  fit(data = train_data)

svm_preds <- predict(final_svm_wf,
                     new_data = test_data,
                     type = "class") %>%
  mutate(type = .pred_class) %>%
  bind_cols(., test_data) %>%
  select(id, type) %>%
  rename(Id = id)

vroom_write(svm_preds, file = "SVM_Preds.csv", delim = ",")

# SVM Degree 2

svm2_model <- svm_poly(cost = tune()) %>%
  set_engine("kernlab") %>%
  set_mode("classification")

svm_wf <- workflow() %>%
  add_recipe(svm_recipe) %>%
  add_model(svm2_model)

svm_tuning_grid <- grid_regular(cost(),
                                levels = 5)

folds <- vfold_cv(data = train_data, v = 5, repeats = 1)

cv_results <- svm_wf %>%
  tune_grid(resamples = folds,
            grid = svm_tuning_grid,
            metrics = metric_set(accuracy))

best_tune_svm <- cv_results %>%
  select_best("accuracy")

final_svm_wf <- svm_wf %>%
  finalize_workflow(best_tune_svm) %>%
  fit(data = train_data)

svm_preds <- predict(final_svm_wf,
                     new_data = test_data,
                     type = "class") %>%
  mutate(type = .pred_class) %>%
  bind_cols(., test_data) %>%
  select(id, type) %>%
  rename(Id = id)

vroom_write(svm_preds, file = "SVM2_Preds.csv", delim = ",")

# SVM Degree 3

svm3_model <- svm_rbf(cost = tune()) %>%
  set_engine("kernlab") %>%
  set_mode("classification")

svm_wf <- workflow() %>%
  add_recipe(svm_recipe) %>%
  add_model(svm3_model)

svm_tuning_grid <- grid_regular(cost(),
                                levels = 5)

folds <- vfold_cv(data = train_data, v = 5, repeats = 1)

cv_results <- svm_wf %>%
  tune_grid(resamples = folds,
            grid = svm_tuning_grid,
            metrics = metric_set(accuracy))

best_tune_svm <- cv_results %>%
  select_best("accuracy")

final_svm_wf <- svm_wf %>%
  finalize_workflow(best_tune_svm) %>%
  fit(data = train_data)

svm_preds <- predict(final_svm_wf,
                     new_data = test_data,
                     type = "class") %>%
  mutate(type = .pred_class) %>%
  bind_cols(., test_data) %>%
  select(id, type) %>%
  rename(Id = id)

vroom_write(svm_preds, file = "SVM3_Preds.csv", delim = ",")