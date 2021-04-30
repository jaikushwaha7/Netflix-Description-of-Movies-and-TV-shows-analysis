# TidyTuesday Netflix titles are movies and which are TV shows
setwd("D:/Study/R/TidyTuesday Netflix titles are movies and which are TV shows")

# how to use the tidymodels packages, from just starting out to tuning more complex models with many 
# hyperparameters.

# How to build features for modeling from text, with this week's #TidyTuesday dataset on Netflix titles. ????

# Our modeling goal is to predict whether a title on Netflix is a TV show or a movie based on its description 
# in this week's  #TidyTuesday dataset.

library(tidyverse)

netflix_titles <- read_csv("https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2021/2021-04-20/netflix_titles.csv")

netflix_titles %>% count(type)
summary(netflix_titles)
head(netflix_titles)
anyNA(netflix_titles)

# What do the description look like? It is alaways a good idea to actually look at your data before modelling
netflix_titles %>%
  slice_sample(n=10) %>%
  pull(description)


# What are the top words in each category
install.packages('tidytext')
library(tidytext)



netflix_titles %>%
  unnest_tokens(word, description) %>%
  anti_join(get_stopwords()) %>%
  count(type, word, sort=T) %>%
  group_by(type) %>%
  slice_max(n, n=15) %>%
  ungroup %>%
  mutate(word= reorder_within(word, n,type)) %>%
  ggplot(aes(n, word, fill=type)) +
  geom_col(show.legend=FALSE, alpha=.8) +
  scale_y_reordered()+
  facet_wrap(~type, scales='free') +
  labs(
    x="Word frequency", y=NULL,
    title='Top words in netflix description by frequecy',
    subtitle = 'After removing the stop words'
  )

# Building a model
library(tidymodels)

set.seed(123)
netflix_split<- netflix_titles %>%
  select(type, description) %>%
  initial_split(strata = type)

netflix_train <- training(netflix_split)
netflix_test <- testing(netflix_split)

set.seed(234)
netflix_folds<- vfold_cv(netflix_train, strata=type)
netflix_folds

# Next, let's create our feature engineering recipe and our model, and then put them together in a modeling workflow. 

library(textrecipes)
library(themis)

netflix_rec <- recipe(type~description, data=netflix_train) %>%
  step_tokenize(description) %>%
  step_tokenfilter(description, max_tokens=1e3) %>%
  step_tfidf(description) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_smote(type)

netflix_rec

install.packages('scotus')
linstall.packages("remotes")
remotes::install_github("EmilHvitfeldt/scotus")
ibrary(scotus)
svm_spec <- svm_() %>%
  set_mode('classification') %>%
  set_engine('LiblineaR')

netflix_wf <- workflow() %>%
  add_recipe(netflix_rec) %>%
  add_model(svm_spec)

netflix_wf

doParallel::registerDoParallel()
set.seed(123)
svm_rs <- fit_resamples(
  netflix_wf,
  netflix_folds,
  metrics = metric_set(accuracy, recall, precision),
  control = control_resamples(save_pred = TRUE)
)

collect_metrics(svm_rs)

# We can use conf_mat_resampled() to compute a separate confusion matrix for each resample,
# and then average the cell counts.
svm_rs %>%
  conf_mat_resampled(tidy = FALSE) %>%
  autoplot()

# Fit and evaluate final model
final_fitted <- last_fit(
  netflix_wf,
  netflix_split,
  metrics = metric_set(accuracy, recall, precision)
)
collect_metrics(final_fitted)


collect_predictions(final_fitted) %>%
  conf_mat(type, .pred_class)

netflix_fit <- pull_workflow_fit(final_fitted$.workflow[[1]])

tidy(netflix_fit) %>%
  arrange(estimate)

tidy(netflix_fit) %>%
  filter(term != "Bias") %>%
  group_by(sign = estimate > 0) %>%
  slice_max(abs(estimate), n = 15) %>%
  ungroup() %>%
  mutate(
    term = str_remove(term, "tfidf_description_"),
    sign = if_else(sign, "More from TV shows", "More from movies")
  ) %>%
  ggplot(aes(abs(estimate), fct_reorder(term, abs(estimate)), fill = sign)) +
  geom_col(alpha = 0.8, show.legend = FALSE) +
  facet_wrap(~sign, scales = "free") +
  labs(
    x = "Coefficient from linear SVM", y = NULL,
    title = "Which words are most predictive of movies vs. TV shows?",
    subtitle = "For description text of movies and TV shows on Netflix"
  )
