setwd("~/Codebase/iTS/src")
require("reticulate")
use_condaenv("its", required = TRUE)
py_config()

os <- import("os")
np <- import("numpy")
library(MASS)


py_run_string("
import pickle
import numpy as np
with open('cache_global/toR/dataset_name.pkl', 'rb') as f:
    dataset_name = pickle.load(f)
with open('cache_global/toR/method.pkl', 'rb') as f:
    method = pickle.load(f)
with open('cache_global/toR/lags.pkl', 'rb') as f:
    lags = pickle.load(f)
with open('cache_global/toR/valid_data.pkl', 'rb') as f:
    valid_data = pickle.load(f)
with open('cache_global/toR/test_data.pkl', 'rb') as f:
    test_data = pickle.load(f)
with open('cache_global/toR/scale_factor.pkl', 'rb') as f:
    scale_factor = pickle.load(f)
with open('cache_global/toR/h.pkl', 'rb') as f:
    h = pickle.load(f)
    
def get_quantiles(x, quantiles=[0.5, 0.80, 0.90, 0.95, 0.99]):
  return np.transpose(np.quantile(x, q=quantiles, axis=1, method='higher'), (2,0,1))
")

dataset_name = py$dataset_name
method = py$method
lags = py$lags
valid_data = py$valid_data
test_data = py$test_data
scale_factor = py$scale_factor
h = py$h
n = nrow(valid_data)
L = ncol(valid_data)
B = 200

set.seed(42)

for (lag in lags) {
  print(lag)
  model_folder_name <- paste0(method, "l", lag, "__", dataset_name)
  model_folder_path <- file.path("discrete-regression", model_folder_name)
  
  XY = vector("list", n)
  Xtest = vector("list", n)
  Ytest = vector("list", n)
  for (i in 1:n) {
    ts = valid_data[i,]
    ts = tail(ts, min(200, L))
    sf = scale_factor[i]
    ts = as.integer(round(ts / sf))
    ts_stack = embed(ts, lag+1)[, (lag+1):1]
    XY[[i]] = ts_stack
    Xtest[[i]] = matrix(tail(ts, lag), ncol=lag)
    Ytest[[i]] = matrix(tail(test_data[i,], h), ncol=h)
  }
  XY <- do.call(rbind, XY)
  max_value = round(quantile(XY, 0.9999))
  Xtest <- do.call(rbind, Xtest)
  actuals = do.call(rbind, Ytest)
  
  colnames(XY) <- c(paste("X", 1:lag, sep = ""), 'Y')
  colnames(Xtest) <- colnames(XY)[1:lag]
  
  formula <- paste("Y ~", paste(colnames(XY)[1:lag], collapse = " + "))
  print('  training glm.nb')
  
  sampled_indices <- sample(1:nrow(XY), size = min(1e6, nrow(XY)), replace = FALSE)
  model_ok = TRUE
  model <- tryCatch({
    glm.nb(formula = formula, data = data.frame(XY[sampled_indices, ]), control = glm.control(maxit = 1000))
  }, error = function(e) {
    message("An error occurred while fitting the model: ", e)
    NULL
  })
  if (is.null(model) | length(model$th.warn) > 0) {  # fallback to a poisson model
    model_ok = FALSE
    model <- glm(formula = formula, data = data.frame(XY[sampled_indices, ]), family = poisson(link = "log"), control = glm.control(maxit = 100))
  }
  
  print('  forecasting...')
  samples = array(NA, dim = c(h, B, n))
  pb <- txtProgressBar(min = 0, max = h, style = 3)
  for (h_ in 1:h) {
    if (h_==1) {
      predicted_means <- predict(model, newdata = data.frame(Xtest), type = "response")
      if (!model_ok) {
        s <- sapply(predicted_means, function(mu) rpois({B}, lambda = mu))
      } else {
        s <- sapply(predicted_means, function(mu) rnbinom({B}, size = model$theta, mu = mu))
      }
      s[s > max_value] = max_value
      samples[h_,,] = s
    } else {
      for (b in 1:B) {
        samples_sub = samples[tail(1:(h_-1),lag), b,]
        if (h_ > 2) samples_sub = t(samples_sub)
        if (h_ > lag) {
          Xtest_ = samples_sub
          if (lag == 1) {
            Xtest_ = matrix(samples_sub)
          }
        } else {
          Xtest_ = cbind(Xtest[,h_:lag], samples_sub)
        }
        colnames(Xtest_) = colnames(Xtest)
        predicted_means <- predict(model, newdata = data.frame(Xtest_), type = "response")
        if (!model_ok) {
          s <- rpois(length(predicted_means), lambda=predicted_means)
        } else {
          s <- rnbinom(length(predicted_means), size=model$theta, mu=predicted_means)
        }
        s[s > max_value] = max_value
        samples[h_,b,] = s
      }
    }
    setTxtProgressBar(pb, h_)
  }
  close(pb)
  quantile_forecasts = py$get_quantiles(samples)
  print(max(max(quantile_forecasts)))
  print(sum(is.na(quantile_forecasts)))
  
  if (!os$path$exists(model_folder_path)) {
    os$makedirs(model_folder_path)
  }
  if (lag == 1) np$save(os$path$join(model_folder_path, "actuals.npy"), actuals)
  np$save(os$path$join(model_folder_path, "qforecasts.npy"), quantile_forecasts)
}

###############################################################################