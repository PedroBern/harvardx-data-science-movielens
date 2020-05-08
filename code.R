################################
# Create edx set, validation set
################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #


################################
# MovieLens Project
################################


# Install libraries, if not already installed
if(!require(tidyverse)) install.packages("tidyverse")
if(!require(ggplot2)) install.packages("ggplot2")
if(!require(caret)) install.packages("caret")

# Load libraries
library(tidyverse)
library(caret)
library(ggplot2)

# Set global options
options(pillar.sigfig = 5)

# --------------------------------------------------------------------------- #

# RMSE function from the material course book
# reference: section 33.7.3 Loss function, page 641
RMSE <- function(true_ratings, predicted_ratings) {
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# Utility function to get RMSE grade
rmse_grade <- function(rmse = NULL){
  grade <- 5
  if (rmse >= 0.9) {
    grade <- 5
  } else if (0.86550 <= rmse & rmse <= 0.89999) {
    grade <- 10
  } else if (0.86500 <= rmse & rmse <= 0.86549) {
    grade <- 15
  } else if (0.86490 <= rmse & rmse <= 0.86499) {
    grade <- 20
  } else if (rmse < 0.86490) {
    grade <- 25
  }
  grade
}

# --------------------------------------------------------------------------- #

# The simplest model
# reference: 33.7.4 A first model, page 641

# The simplest model is the mean of the training set
mu <- mean(edx$rating)
mu # 3.512465

naive_rmse <- RMSE(validation$rating, mu)
naive_rmse # 1.061202

rmse_results <- tibble(
  method = "Just the average",
  RMSE = naive_rmse,
  grade = rmse_grade(naive_rmse))

rmse_results
# 1 Just the average                  1.0612      5

# Any value different from the simplest model
# should return a worse RMSE:
# Values lesser than mu
RMSE(validation$rating, runif(1, 0.0, mu - 0.1)) > naive_rmse
# Values greater then mu
RMSE(validation$rating, runif(1, mu + 0.1, 5.0)) > naive_rmse


# --------------------------------------------------------------------------- #

# Adding the movie effect to the model
# reference: 33.7.5 Modeling movie effects, page 642

# Some movies are generally rated higher than others
# We can see this by computing the difference from
# the average for each movie rating and the average for
# all movies ratings
movie_avgs <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))

# plot the movie effect
qplot(b_i, data = movie_avgs, bins = 100, color = I("black"))

# Compute the new prediction rate, adding the movie effect
predicted_ratings <- mu +
  validation %>%
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)

movie_effect_rmse <- RMSE(validation$rating, predicted_ratings)
movie_effect_rmse # 0.9439087

rmse_results <- add_row(
  rmse_results,
  method = "Movie effect",
  RMSE = movie_effect_rmse,
  grade = rmse_grade(movie_effect_rmse))

rmse_results
# 1 Just the average                  1.0612      5
# 2 Movie effect                      0.94391     5

# --------------------------------------------------------------------------- #

# Adding the user effect to the model
# reference: 33.7.6 User effects, page 643

# Similar to the movie effect, there are users that
# rate always higher or always lower than the average
# We can see this by ploting the average ratings by users
# who have rated more than 100 movies
edx %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating)) %>%
  filter(n()>=100) %>%
  ggplot(aes(b_u)) +
  geom_histogram(bins = 30, color = "black")

# The user effect is the difference from the
# user rating, the average rating and the movie effect
user_avgs <- edx %>%
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

predicted_ratings <- validation %>%
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

user_and_movie_effect_rmse <- RMSE(predicted_ratings, validation$rating)
user_and_movie_effect_rmse # 0.86535

rmse_results <- add_row(
  rmse_results,
  method = "User and movie effect",
  RMSE = user_and_movie_effect_rmse,
  grade = rmse_grade(user_and_movie_effect_rmse))

rmse_results
# 1 Just the average                  1.0612      5
# 2 Movie effect                      0.94391     5
# 3 User and movie effect             0.86535    15

# --------------------------------------------------------------------------- #

# Regulatization on the movie effect
# reference: 33.9.2 Penalized least squares, page 647

# If a movie was rated many times, the movie effect is more precices,
# while movies rated less times, e.g. 1 time, will not be precie, since
# the movie effect will be the single movie rating deviation from the
# overall rating mean

edx %>%
  group_by(movieId) %>%
  summarise(cnt = n()) %>%
  as.data.frame() %>%
  left_join(movie_avgs, by='movieId') %>%
  ggplot(aes(b_i, cnt)) +
  xlab("Movie effect") +
  ylab("Count - log scale") +
  scale_y_log10() +
  geom_point() +
  geom_hline(yintercept=10, linetype="dashed", color = "red")

# Tunning parameter
lambdas <- seq(0, 10, 0.25)

# Summarize the sum of the deviation from the mean,
# and the numbers of appearence for each movie,
# to make the further movie effect calculation with the
# tunning parameter
just_the_sum <- edx %>%
  group_by(movieId) %>%
  summarize(s = sum(rating - mu), n_i = n())

# Calculate the rmse for each tunning parameter
rmses <- sapply(lambdas, function(l){
  predicted_ratings <- validation %>%
    left_join(just_the_sum, by='movieId') %>%
    mutate(b_i = s/(n_i+l)) %>% # regularized movie effect
    mutate(pred = mu + b_i) %>%
    pull(pred)
  return(RMSE(predicted_ratings, validation$rating))
})

# Plot the tunning parameter and the rmse
qplot(lambdas, rmses)

# Get the best tunning parameter
best_lambda <- lambdas[which.min(rmses)] # 2.5

regularized_movie_effect_rmse <- rmses[match(best_lambda, lambdas)]
regularized_movie_effect_rmse # 0.9438521

# Plot the difference between original and regularized effect
movie_reg_avgs <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n() + best_lambda), n_i = n())
tibble(original = movie_avgs$b_i, regularlized = movie_reg_avgs$b_i, n = movie_reg_avgs$n_i) %>%
  ggplot(aes(original, regularlized, size=sqrt(n))) +
  geom_point(shape=1, alpha=0.5) +
  ggtitle("Difference between original and regularized effect")

# Print the results
rmse_results <- add_row(
  rmse_results,
  method = "Regularized movie effect",
  RMSE = regularized_movie_effect_rmse,
  grade = rmse_grade(regularized_movie_effect_rmse))

rmse_results
# 1 Just the average                  1.0612      5
# 2 Movie effect                      0.94391     5
# 3 User and movie effect             0.86535    15
# 4 Regularized movie effect          0.94385     5

# --------------------------------------------------------------------------- #

# Regularized movie and user effect - same tuning parameter
# reference: 33.9.3 Choosing the penalty terms, page 650

# Tunning parameter for both user and movie effect
lambdas <- seq(0, 1, 0.25)

rmses <- sapply(lambdas, function(l){
  
  b_i <- edx %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- edx %>%
    left_join(b_i, by = "movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  predicted_ratings <- edx %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, edx$rating))
})

# Plot the tunning parameter and the rmse
qplot(lambdas, rmses)

# Get the best tunning parameter
best_lambda <- lambdas[which.min(rmses)] # 0.5

regularized_movie_and_user_effect_same_lambda_rmse <- rmses[match(best_lambda, lambdas)]

rmse_results <- add_row(
  rmse_results,
  method = "Regularized movie and user effect same tuning parameter",
  RMSE = regularized_movie_and_user_effect_same_lambda_rmse,
  grade = rmse_grade(regularized_movie_and_user_effect_same_lambda_rmse))

rmse_results
# 1 Just the average                                        1.0612      5
# 2 Movie effect                                            0.94391     5
# 3 User and movie effect                                   0.86535    15
# 4 Regularized movie effect                                0.94385     5
# 5 Regularized movie and user effect same tuning parameter 0.85670    25


# --------------------------------------------------------------------------- #

# Regularized movie and user effect
# reference: 33.9.3 Choosing the penalty terms, page 650

# Tunning parameter just for the user effect
lambdas <- seq(0, 2, 0.25)

rmses <- sapply(lambdas, function(l){
  
  b_u <- edx %>%
    left_join(movie_reg_avgs, by = "movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  predicted_ratings <- edx %>%
    left_join(movie_reg_avgs, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, edx$rating))
})

# Plot the tunning parameter and the rmse
qplot(lambdas, rmses)

# Get the best tunning parameter
best_lambda <- lambdas[which.min(rmses)] # 0.0

# Plot the difference between original and regularized effect
user_reg_avgs <- edx %>%
  left_join(movie_reg_avgs, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n() + best_lambda), n_i = n())
tibble(original = user_avgs$b_u, regularlized = user_reg_avgs$b_u, n = user_reg_avgs$n_i) %>%
  ggplot(aes(original, regularlized, size=sqrt(n))) +
  geom_point(shape=1, alpha=0.5) +
  ggtitle("Difference between original and regularized effect")


# Print the result
regularized_movie_and_user_effect_rmse <- rmses[match(best_lambda, lambdas)]

rmse_results <- add_row(
  rmse_results,
  method = "Regularized movie and user effect independent tuning parameter",
  RMSE = regularized_movie_and_user_effect_rmse,
  grade = rmse_grade(regularized_movie_and_user_effect_rmse))

rmse_results
# 1 Just the average                                               1.0612      5
# 2 Movie effect                                                   0.94391     5
# 3 User and movie effect                                          0.86535    15
# 4 Regularized movie effect                                       0.94385     5
# 5 Regularized movie and user effect same tuning parameter        0.85670    25
# 6 Regularized movie and user effect independent tuning parameter 0.85668    25






