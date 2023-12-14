library(R6)
library(tidymodels)
library(tidyverse)
library(here)
library(poissonreg)
library(glmnet)


freMTPLfreq <- read_csv(here("data", "freMTPLfreq.csv"))

set.seed(1994)
data_split <- initial_split(freMTPLfreq, prop = 0.7)
data_train <- training(data_split)
data_test <- testing(data_split)


# Toy example of an R6 class ----------------------------------------------------------------------------------------------------

Person <-
  R6Class(
    classname = "Person", # Nom de la classe
    public = list(
      name = NULL, # Attributs
      hair = NULL,
      
      initialize = function(name = NA, hair = NA) { # Fonction spéciale « initialize »
        self$name <- name
        self$hair <- hair
        self$greet()
      },
      
      set_hair = function(val) { # Méthode
        self$hair <- val
      },
      
      greet = function() { # Méthode
        cat(paste0("Hello, my name is ", self$name, ".\n"))
      }
    )
  )

personne <- Person$new(name = "Francis", hair = "brown") # Création d'un objet de classe « Person »

# Person: classe R6
# personne: objet de classe R6


personne <- Person$new(name = "Catherine", hair = "plat")
personne

personne$set_hair("frisé")
personne

# R6 class for Poisson GLM ------------------------------------------------------------------------------------------------------

PoissonGLM <-
  R6Class(
    classname = "PoissonGLM",
    
    public = list(
      
      features = NULL,
      response_name = NULL,
      expo_name = NULL,
      formula = NULL,
      fitted_wf = NULL,
      
      initialize = function(features, response_name = "ClaimNb", expo_name = "Exposure") {
        self$features <- features
        self$response_name <- response_name
        self$expo_name <- expo_name
        self$formula <- as.formula(paste(response_name, "~", paste(features, collapse = "+"), "-", expo_name, " + offset(log(", expo_name, "))"))
        
        self$make_description()
      },
      
      make_description = function() {
        cat("PoissonGLM object initialized with the following formula:\n\n")
        print(self$formula)
      },
      
      fit_model = function(train_df) {
        
        # Keep only provided features, response variable and exposure
        response_name_sym <- rlang::sym(self$response_name)
        expo_name_sym <- rlang::sym(self$expo_name)
        
        dat <- 
          train_df |>
          select(
            !!response_name_sym, 
            !!expo_name_sym,
            all_of(self$features)
          )
        
        # Workflow specification
        recette <- 
          recipe( ~ ., data = dat) |>
          update_role(!!response_name_sym, new_role = "outcome")
        
        spec <- poisson_reg(engine = "glm")
        
        wf <-
          workflow() |>
          add_recipe(recette) |>
          add_model(spec, formula = self$formula)
        
        # Fit model and save as an attribute
        fit <- parsnip::fit(wf, data = dat)
        self$fitted_wf <- fit
        
        return(fit)
      },
      
      predict = function(new_df) {
        predict(self$fitted_wf, new_data = new_df)
      },
      
      print_summary = function() {
        if (is.null(self$fitted_wf)) {stop("You must fit the model before having a summary of it")}
        broom::tidy(self$fitted_wf)
      }
    )
  )


# Use case ----------------------------------------------------------------------------------------------------------------------

model <- PoissonGLM$new(features = c("CarAge", "DriverAge"), response_name = "ClaimNb", expo_name = "Exposure")
model$fit_model(data_train)
model$predict(new_df = data_test)
model$print_summary()

# Idea: adding a method "$add_your_recipe" that takes a recipe object as an argument and saves it as an attribute
# Then, when calling the $fit method, we can use this supplied recipe in add_recipe  


# R6 class for Poisson GLM lasso that inherits from PoissonGLM ------------------------------------------------------------------

PoissonGLMLasso <-
  R6Class(
    classname = "PoissonGLMLasso",
    inherit = PoissonGLM,
    
    public = list()
  )

model_lasso <- PoissonGLMLasso$new(features = c("CarAge", "DriverAge"))
model_lasso$fit_model(data_train)


# Let's overwrite make_description and fit methods ------------------------------------------------------------------------------ 

PoissonGLMLasso <-
  R6Class(
    classname = "PoissonGLMLasso",
    inherit = PoissonGLM,
    
    public = list(
      make_description = function() {
        cat("PoissonGLMLasso object initialized with the following formula:\n\n")
        print(self$formula)
      },
      
      
      fit_model = function(train_df, lambda) {
        
        # Keep only provided features, response variable and exposure
        response_name_sym <- rlang::sym(self$response_name)
        expo_name_sym <- rlang::sym(self$expo_name)
        
        dat <- 
          train_df |>
          select(
            !!response_name_sym, 
            !!expo_name_sym,
            all_of(self$features)
          )
        
        # Workflow specification
        recette <- 
          recipe( ~ ., data = dat) |>
          update_role(!!response_name_sym, new_role = "outcome") |>
          step_normalize(all_numeric_predictors(), -!!expo_name_sym)
        
        spec <- poisson_reg(engine = "glmnet", penalty = lambda, mixture = 1)
        
        wf <-
          workflow() |>
          add_recipe(recette) |>
          add_model(spec, formula = self$formula)
        
        # Fit model and save as an attribute
        fit <- parsnip::fit(wf, data = dat)
        self$fitted_wf <- fit
        
        return(fit)
      }
    )
  )

model_lasso <- PoissonGLMLasso$new(features = c("CarAge", "DriverAge"))
model_lasso$fit_model(data_train, lambda = 0.0005)
model_lasso$print_summary()
model_lasso$predict(data_test)


# Idée: ajout d'une méthode "tune_lambda" qui fait un grid search et nous renvoie la valeur optimale de lambda

# R6 class for metrics ----------------------------------------------------------------------------------------------------------
Metrics <- 
  R6Class(
    classname = "Metrics",
    
    public = list(
      
      PoissonGLM_model = NULL,
      valid_df = NULL,
      
      initialize = function(PoissonGLM_model, valid_df) {
        self$PoissonGLM_model = PoissonGLM_model
        self$valid_df = valid_df
        if (is.null(PoissonGLM_model$fitted_wf)) {stop("You must fit the model before using the Metrics class")}
      },
      
      compute_rmse = function() {
        pred_vec <- pull(self$PoissonGLM_model$predict(self$valid_df))
        target_vec <- self$valid_df[[self$PoissonGLM_model$response_name]]
        
        sqrt(mean((pred_vec - target_vec) ^ 2))
      },
      
      plot_lorenz_curve = function() {
        df_lorenz <- 
          self$valid_df |>
          add_column(self$PoissonGLM_model$predict(self$valid_df)) |>
          mutate(.pred = .pred / Exposure) |>
          select(ClaimNb, Exposure, .pred) |>
          arrange(.pred) |>
          mutate(
            cum_expo = cumsum(Exposure) / sum(Exposure),
            cum_ClaimNb = cumsum(ClaimNb) / sum(ClaimNb)
          )
        
        df_lorenz |>
          ggplot(aes(x= cum_expo, y = cum_ClaimNb)) + 
          geom_abline(slope = 1, linetype = "dashed", color = "black") +
          geom_line(alpha = 0.7)   + 
          scale_y_continuous(labels = scales::percent_format()) +
          scale_x_continuous(labels = scales::percent_format()) +
          labs(
            title = "Lorenz Curve",
            x = "Cumulative earned exposure (% of total)",
            y = "Cumulative number of claims (% of total)",
          ) +
          theme_bw()
      }
    )
  )

model$fit_model(data_train)
metrics <- Metrics$new(model, data_test)
metrics$compute_rmse()
metrics$plot_lorenz_curve()





# Lorenz curve

x <- 
  data_test |> 
  add_column(model$predict(data_test)) |>
  mutate(.pred = .pred / Exposure)



df.Lorenz <- 
  x |>
  select(PolicyID, ClaimNb, Exposure, .pred) |>
  arrange(.pred) |>
  mutate(
    cum_expo = cumsum(Exposure) / sum(Exposure),
    cum_ClaimNb = cumsum(ClaimNb) / sum(ClaimNb)
  )

df.Lorenz %>% 
  ggplot(aes(x= cum_expo, y = cum_ClaimNb)) + 
  geom_abline(slope = 1, linetype = "dashed", color = "black") +
  geom_line(alpha = 0.7)   + 
  guides(colour = guide_legend(override.aes = list(alpha=1))) +# full opacity dans la legende
  scale_y_continuous(labels = scales::percent_format()) +
  scale_x_continuous(labels = scales::percent_format()) +
  labs(
    title = 'Lorenz Curves',
    x = "cumulative earned exposure (% of total)",
    y = "cumulative nb of claims (% of total)",
    color = "Model"
  ) +
  theme_bw()
