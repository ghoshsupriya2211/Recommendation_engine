#Amazon Recommendation System (Video Games)
# Data Wrangling
library(tidyverse)
library(lubridate)
# Data Visualization
library(scales)
#install.packages("skimr")
library(skimr)
# Recommender System
library(recommenderlab)
options(scipen = 999)
library(data.table)
library(magrittr)
library(ggplot2)
library(dplyr)

#Reading ratings and metadata file
# Dataset contains information about :
# rating : The rating given by each user to each time with a timestamp
# metadata : The metadata/information about each video game item/accessories
# There are over 2 millions rating given by users and more than 80,000 different video game items.
rating <- data.table::fread("Video_Games.csv")
metadata <- data.table::fread("metadata.csv")

metadata <- metadata %>% 
  select(asin, title, category1, category2, category3, price, brand, feature, tech1, tech2, 
         image, description)

cat("----------Rating Dataset----------\n")
glimpse(rating)

rating <- rating %>% 
  set_names(c("item", "user", "rating", "timestamp")) %>% 
  mutate(
    timestamp = as_datetime(timestamp)
  )
head(rating, 10)

cat("\n\n----------Metadata Dataset----------\n")
glimpse(metadata)

# Exploratory Data Analysis(EDA)
# counting how many times an item is rated by a single user.
rating %>% 
  count(item, user) %>% 
  arrange(desc(n)) %>% 
  head(10)
# An item is seen to be rated more than once by a single user.
# Perhaps they buy them multiple times or there are duplicate data. Depending on purpose, 
# the mean of the rating can be calculated or the recent rating only can be chosen. Thus, 
# for the next analysis only the latest rating given by the user is considered & rest is ignored.
rating <- rating %>% 
  group_by(item, user) %>% 
  arrange(desc(timestamp)) %>% # Arrange rating by timestamp 
  slice(1) %>% # Take only the latest rating
  ungroup()

cat(
  paste("Number of Unique Rating :", nrow(rating) %>% prettyNum(big.mark = ","))
)

# Frequency of rating given to each game is checked. Based on the summary, some games are 
# rated only once or twice. On average, a item is rated 5 times based on the median value.
game_count <- rating %>% 
  count(item)

game_count %>% 
  skim()
# For the next analysis, considering items that's been rated at least 50 times. The choice of 
# number of rating is arbitrary, the limit is set as per random choice. After filtering out 
# the item, the number of rating is significantly reduced.
select_item <- game_count %>%  
  filter(n > 50) %>% 
  pull(item)
# Update the rating
rating <- rating %>% 
  filter(item %in% select_item)
cat(
  paste("Number of Rating :", nrow(rating) %>% prettyNum(big.mark = ","))
)

# User Rating Distribution
# updated rating data is considered, and the frequency of each rating score (1-5) given by user 
# is considered. Based on the bar chart, most user gave rating score of 5 for the game 
# they bought.
rating %>% 
  ggplot(aes(rating)) +
  geom_bar(fill = "firebrick") +
  scale_y_continuous(labels = number_format(big.mark = ",")) +
  labs(x = "Rating", y = "Frequency",
       title = "Number of Rating Given by User") +
  theme_minimal()

# how many video games a user have rated is checked by looking at the distribution. Based on the 
# statistics, most user only rate a single game item. This data may not informative to us 
# since we don't know what other item that those user also buy.
user_count <- rating %>% 
  count(user)

user_count %>% 
  skim()

# For the next analysis, users who've rated at least 10 different games are considered. Again, 
# this choice is arbitrary.
select_user <- user_count %>% 
  filter(n > 10) %>% 
  pull(user)
# update rating
rating <- rating %>% 
  filter(user %in% select_user)
cat(
  paste("Number of Rating :", nrow(rating) %>% prettyNum(big.mark = ","))
)
# This also decreased the dimension for our dataset with only 7113 users remaining. We omit 
# most of the user since they only give one rating and most game are only rated once so they 
# are not very informative.
# Let's once again check the rating distribution.
rating %>% 
  ggplot(aes(rating)) +
  geom_bar(fill = "firebrick") +
  scale_y_continuous(labels = number_format(big.mark = ",")) +
  labs(x = "Rating", y = "Frequency",
       title = "Number of Rating Given by User") +
  theme_minimal()
# We may also check the number of rating frequency over time. The graph below shows that 
# rating activity reach its peak around 2015 and start to decrease afterwards.
rating %>% 
  mutate(
    timestamp = floor_date(timestamp, unit = "week")
  ) %>% 
  count(timestamp, rating) %>% 
  ggplot(aes(timestamp, n, color = as.factor(rating), group = rating)) +
  geom_line() +
  scale_color_brewer(palette = "Dark2") +
  scale_x_datetime(date_breaks = "2 year", labels = date_format(format = "%Y")) +
  labs(x = NULL, y = "Frequency",
       title = "Weekly Frequency of Rating Activity",
       color = "Rating") +
  theme_minimal() +
  theme(legend.position = "top")
# We can check the most rated game at time interval between 2014 and 2016 by joining the 
# rating data with the item metadata.
rating %>% 
  filter(year(timestamp) > 2014,
         year(timestamp) < 2016) %>% 
  count(item) %>% 
  arrange(desc(n)) %>% 
  head(10) %>% 
  left_join(metadata, by = c("item" = "asin"))

# Data Preprocessing
# After we finished exploring the data, we will convert the data into a matrix, with the row is 
# the user and each column is the item/item. The value in each cell is the rating given by the 
# user. If the user haven't rated any item, the cell value will be a missing value (NA).
# We can make a recommendation system using 2 type of matrix:
# Real Rating Matrix
# Each cell represent a normalized rating given by the user.
# Binary rating matrix
# Each cell represents a response given by the user and can only have binary values 
# (recommended/not recommended, good/bad).
#  real rating matrix of our data. From 133,197 ratings, we have build a 7,113 x 8,550 
# rating matrix.
rating_matrix <- rating %>% 
  select(item, user, rating) %>% 
  reshape2::dcast(user ~ item) %>% # Convert long data.frame to wide data.frame
  column_to_rownames("user") %>% 
  as.matrix() %>% 
  as("realRatingMatrix")

rating_matrix
# We may peek inside the sample of the matrix. Since a user rarely give rating to all available 
# items, the matrix is mostly empty. This kind of matrix is called the sparse matrix because 
# mostly it's sparse and has missing value. On the example below, we only have 2 ratings from 9 
# user and 9 items.
rating_matrix@data[1:9, 2001:2009]
# As you can see, the value is not normalized yet and still in range of [1,5]. 
# Since we use the real rating matrix, we need to normalize the rating. However, you can also 
# skip this step since the model fitting will normalize the data by default. You can normalize 
# the data via two method:
# Normalization by mean (center method)
# We normalize the data by subtracting the data with it's own mean for each user.
# Normalization by Z-score
# We use the Z-score of the standard normal distribution to scale the data.
# We don't have to manually normalize the rating matrix, since the model fitting process in 
# recommenderlab will normalize our data by default. But if you want to do it outside the model, 
# you use the normalize() function and determine the method, either center method or Z-score 
# method
normalize(rating_matrix, method = "center")
# Building Recommender System
# There are several algorithm that you can use to build a recommendation system using the 
# recommenderlab package. You can check it by looking at the registry and specify the data type. 
# Below are some recommendation algorithm for a rating matrix with real value.
recommenderRegistry$get_entries(dataType = "realRatingMatrix") %>% 
  names()
# POPULAR : Popular Recommendation
# UBCF : User-Based Collaborative Filtering
# IBCF : Item-Based Collaborative Filtering
# RANDOM : Random Recommendation
# SVD : Singular Value Decomposition
# SVDF : Funk Singular Value Decomposition
# For now, let's start make a recommendation system with the Funk SVD method. You can check 
# the initial/default parameter of the model.
# recommenderRegistry$get_entry("SVDF", dataType = "realRatingMatrix")
# k : number of features (i.e, rank of the approximation).
# gamma : regularization term.
# lambda : learning rate.
# min_improvement : required minimum improvement per iteration.
# min_epochs : minimum number of iterations per feature.
# max_epochs : maximum number of iterations per feature.
# verbose : show progress.
# We will modify the parameter by using Z-score normalization instead.
recom_svdf <- Recommender(data = rating_matrix,
                          method = "SVDF",
                          parameter = list(normalize = "Z-score"))

recom_svdf <- read_rds("svdf.Rds")
# Give Recommendation
# Now we will try to generate a random new user to simulate the recommendation process.
# Let's say we have the following new users who only gave a single or two rating.
select_item <- unique(rating$item)

set.seed(251)
new_user <- data.frame(user = sample(10, 10, replace = T),
                       item = sample(select_item, 10),
                       rating = sample(1:5, 10, replace = T)
) %>% 
  arrange(user) 

new_user %>% 
  left_join(metadata %>% select(asin, title), by = c("item" = "asin"))
# We also need to convert them into the same real rating matrix.
dummy_df <- data.frame(user = -1,
                       item = select_item,
                       rating = NA) %>% 
  reshape2::dcast(user ~ item) %>% 
  select(-user)

new_matrix <- new_user %>% 
  reshape2::dcast(user ~ item) %>% 
  column_to_rownames("user")

new_matrix
# Let's convert them into the proper real rating matrix
select_empty <- select_item[!(select_item %in% names(new_matrix))]

new_matrix <- new_matrix %>% 
  bind_cols(
    dummy_df %>% select(all_of(select_empty)) 
  ) %>% 
  as.matrix() %>% 
  as("realRatingMatrix")

new_matrix
# You can check the content of the rating matrix.
new_matrix@data[ , 1:9]
# To get the recommendation for the new data, we simply use predict(). Here, we want to get 
# top 5 recommendation for each user based on what items they have already rated. To get the 
# recommended item, use type = "topNList" and specify the number of top n recommendation. 
# The top-n method will automatically give you the top n item that has the highest score/rating
# for each new user.
predict_new <- predict(recom_svdf, 
                       new_matrix,
                       type = "topNList",
                       n = 5
)

predict_new
# We further build the proper data.frame to show the recommendation. Below are the top 5 
# recommended item for each user.
as(predict_new, 'list') %>% 
  map_df(as.data.frame) %>% 
  rename("asin" = 1) %>% 
  mutate(
    user = map(unique(new_user$user), rep, 5) %>% unlist()
  ) %>% 
  select(user, everything()) %>% 
  left_join(metadata %>% select(asin, title)) %>% 
  distinct()
# You can also get the predicted rating from all missing item of each user. The missing value 
# (the dots .) is the item that has been rated previously by the user and so they don't have 
# new predicted rating.
pred_rating <- predict(recom_svdf, 
                       new_matrix,
                       type = "ratings"
)

pred_rating@data[ , 1:9]
# Evaluating Model
# Now that we've successfully build our model, how do we know that the recommendation system 
# is good enough and not just throwing some random suggestions?
# Similar with the classical regression and classification problem, we can use cross-validate by 
# splitting data into data train and test with 90% of the rating data will be the training dataset.
# Selecting given = -1 means that for the test users 'all but 1' randomly selected item is 
# withheld for evaluation.
# The goodRating determine the threshold to classify whether an item should be recommended or not,
# similar with how we determine threshold for classification problem. The goodRating is set on 
# 0 since our normalized data is zero-centered and any rating that has value above 0 will be 
# considered as positive and will be recommended.
# Using the top-N recommendation, we will get the following confusion matrix from the model.
# Recommended      Actually Buy - TP           Actually Not Buy - FN
# Not Recommend   Actually Buy - FP            Actually Not Buy - FN
# We then evaluate the model using the same metrics as the usual classification method, such as 
# model accuracy, recall, and precision.
# Recall(Sensitivity)=TP/(TP+FN)
# Precision=TP/(TP+FP)
set.seed(123)
scheme <- rating_matrix %>% 
  evaluationScheme(method = "split",
                   train  = 0.9,  # 90% data train
                   given  = -1,
                   goodRating = 0
  )
scheme
# Now we will run the training process for the Funk SVD method with Z-score normalization. 
# We will look at the model performance performance when it give use 1, 4, 8, 12, 16, and 20 
# recommended items.
# Rating Error Measurement
# You can get the rating score of the recommended item and calculate the error instead. 
# The evaluation method using top-N method rely on the good rating as the threshold for 
# classifying positive and negative recommendation. For a real rating matrix, we can also 
# directly measure how good the model predict the rating and measures their error, including MAE,
# MSE, and RMSE.
result_rating <- evaluate(scheme, 
                          method = "svdf",
                          parameter = list(normalize = "Z-score", k = 20),
                          type  = "ratings"
)
library(beepr)
beepr::beep(8)
# result_rating <- read_rds("output/svdf_rating.Rds")
# From the evaluation process, we can summarize the mean of each performance measures 
# from each fold.
result_rating@results %>% 
  map(function(x) x@cm) %>% 
  unlist() %>% 
  matrix(ncol = 3, byrow = T) %>% 
  as.data.frame() %>% 
  summarise_all(mean) %>% 
  setNames(c("RMSE", "MSE", "MAE"))
# Top-N Recommendation
set.seed(123)
result <- evaluate(scheme, 
                   method = "svdf",
                   parameter = list(normalize = "Z-score", k = 20),
                   type  = "topNList", 
                   n     = c(1, seq(4, 20, 4))
)
# The evaluation scheme took some time to run, so I have provided the saved object as well.
# Here is the recap of the model performance using the top-N recommendation.
result <- read_rds("output/svdf_val.Rds")

result@results %>% 
  map_df(function(x) x@cm %>% 
           as.data.frame %>% 
           rownames_to_column("n")) %>% 
  mutate(n = as.numeric(n)) %>% 
  arrange(n) %>% 
  rename("Top-N" = n)
# ROC Curve
# From the result of the evaluation method, we can get the performance metrics. Here, 
# we will visualize the ROC Curve of the model.
result %>% 
  getConfusionMatrix() %>% 
  map_df(~as.data.frame(.) %>% rownames_to_column("n")) %>%
  group_by(n) %>% 
  summarise_all(mean) %>% 
  ggplot(aes(x = FPR, y = TPR)) +
  geom_line() +
  geom_point(shape = 21, fill = "skyblue", size = 2.5) +
  scale_x_continuous(limits = c(0, 0.0025)) +
  labs(title = "ROC Curve",
       x = "False Positive Rate", 
       y = "True Positive Rate",
       subtitle = "method : SVD") +
  theme_minimal()
# Precision-Recall Curve
result %>% 
  getConfusionMatrix() %>% 
  map_df(~as.data.frame(.) %>% rownames_to_column("n")) %>%
  group_by(n) %>% 
  summarise_all(mean) %>% 
  ggplot(aes(x = recall, y = precision)) +
  geom_line() +
  geom_point(shape = 21, fill = "skyblue", size = 2.5) +
  labs(title = "Precision-Recall Curve",
       x = "Recall", y = "Precision",
       subtitle = "method : SVD") +
  theme_minimal()
# Model Comparison
# Now that we've learn how to evaluate a recommendation model, we can start to compare 
# multiple model to get the best model for our dataset. Since we've evaluated Funk SVD on 
# the previous step, for this part we will evaluate the following method:
# Random
# Popular item
# SVD
# Alternating Least Square (ALS)
# Item-Based Collaborative Filtering (IBCF)
algorithms <- list(
  "Random items" = list(name = "RANDOM"),
  "Popular items" = list(name = "POPULAR"),
  "SVD" = list(name = "SVD"),
  "ALS" = list(name = "ALS")
  #"item-based CF" = list(name = "IBCF")
)
# Rating Error Measurement
# We will evaluate the model by measuring the ratings and get the RMSE, MSE, and MAE value.
set.seed(123)
result_error <- evaluate(scheme, 
                         algorithms, 
                         type  = "ratings"
)
# result_error <- read_rds("output/eval_error.Rds")
# Then, we visualize the result.
get_error <- function(x){
  x %>% 
    map(function(x) x@cm) %>% 
    unlist() %>% 
    matrix(ncol = 3, byrow = T) %>% 
    as.data.frame() %>% 
    summarise_all(mean) %>% 
    setNames(c("RMSE", "MSE", "MAE"))
}


result_error_svdf <- result_rating@results %>% 
  get_error() %>% 
  mutate(method = "Funk SVD")

map2_df(.x = result_error@.Data, 
        .y = c("Random", "Popular", "SVD", "ALS"), 
        .f = function(x,y) x@results %>% get_error() %>% mutate(method = y)) %>% 
  bind_rows(result_error_svdf) %>%
  pivot_longer(-method) %>% 
  mutate(method = tidytext::reorder_within(method, -value, name)) %>% 
  ggplot(aes(y =  method, 
             x =  value)) +
  geom_segment(aes(x = 0, xend = value, yend = method)) +
  geom_point(size = 2.5, color = "firebrick" ) +
  tidytext::scale_y_reordered() +
  labs(y = NULL, x = NULL, title = "Model Comparison") +
  facet_wrap(~name, scales = "free_y") +
  theme_minimal()

# The Funk SVD method acquire the lowest error compared to other algorithms. However, the 
# difference is not that significant with the SVD method.
# Top-N Recommendation
# If you are interested, you may also evaluate all algorithm using the top-N recommendation 
# instead.
result_multi <- evaluate(scheme, 
                         algorithms, 
                         type  = "topNList", 
                         n     = c(1, seq(4, 20, 4))
)

beepr::beep(16)
# result_multi <- read_rds("output/eval_scheme.Rds")
# Popular and SVD method is competing as the best method for this problem with the Funk SVD 
# following behind. With bigger N, popular method is expected to be better since during the 
# preprocess step we only consider game items that has been rated more than 50 times, so less 
# popular item is out of the data.
get_recap <- function(x){
  x %>% 
    getConfusionMatrix() %>% 
    map_df(~as.data.frame(.) %>% rownames_to_column("n")) %>%
    group_by(n) %>% 
    summarise_all(mean)
}

result_svdf <- result %>% 
  get_recap() %>% 
  mutate(method = "Funk SVD")

result_eval <- map2_df(.x = result_multi, 
                       .y = c("Random", "Popular", "SVD","ALS"), 
                       .f = function(x, y) x %>% get_recap() %>% mutate(method = y)
) %>% 
  bind_rows(result_svdf)


result_eval %>% 
  ggplot(aes(x = FPR, y = TPR, color = method)) +
  geom_line() +
  geom_point() +
  labs(title = "ROC Curve", color = "Method",
       y = "True Positive Rate", x = "False Positive Rate") +
  theme_minimal() +
  theme(legend.position = "top")











