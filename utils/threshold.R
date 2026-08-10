library(pROC)
library(caret)
library(PRROC)


set.seed(123)
n <- 1000
y_true <- sample(c(0, 1), n, replace = TRUE, prob = c(0.9, 0.1))  # Imbalanced data
y_pred_prob <- runif(n)  # Example probabilities from a logistic regression model

# ROC curve
roc_obj <- roc(y_true, y_pred_prob)
plot(roc_obj, main = "ROC Curve")
auc(roc_obj)

optimal_idx <- which.max(roc_obj$sensitivities + roc_obj$specificities - 1)
optimal_threshold_roc <- roc_obj$thresholds[optimal_idx]
optimal_threshold_roc


# Precision-Recall Curve
pr_curve <- pr.curve(scores.class0 = y_pred_prob[y_true == 1], 
                     scores.class1 = y_pred_prob[y_true == 0], 
                     curve = TRUE)
plot(pr_curve, main = "Precision-Recall Curve")
pr_curve$auc.integral


# Maximizing F1 score

f1_score <- function(precision, recall) {
  if (precision + recall == 0) {
    return(0)
  } else {
    return(2 * precision * recall / (precision + recall))
  }
}

thresholds <- unique(y_pred_prob)
f1_scores <- sapply(thresholds, function(threshold) {
  predicted <- ifelse(y_pred_prob >= threshold, 1, 0)
  cm <- confusionMatrix(factor(predicted, levels = c(0,1)), as.factor(y_true), positive = "1")
  precision <- cm$byClass["Precision"]
  recall <- cm$byClass["Recall"]
  f1_score(precision, recall)
})

optimal_threshold_f1 <- thresholds[which.max(f1_scores)]
optimal_threshold_f1



opt_threshold = function(y_pred_prob, y_true){
  
  y_true = factor(y_true, levels = c(0,1))
  # ROC threshold
  roc_obj <- suppressMessages(roc(y_true, y_pred_prob))
  #plot(roc_obj, main = "ROC Curve")
  #auc(roc_obj)
  
  optimal_idx <- which.max(roc_obj$sensitivities + roc_obj$specificities - 1)
  optimal_threshold_roc <- roc_obj$thresholds[optimal_idx]
  
  
  distances <- sqrt((1 - roc_obj$sensitivities)^2 + (roc_obj$specificities - 1)^2)
  
  # Find the threshold that minimizes the distance
  optimal_idx <- which.min(distances)
  optimal_threshold_roc_dist <- roc_obj$thresholds[optimal_idx]
  
  # F1 score maximization
  thresholds <- unique(y_pred_prob)
  f1_scores <- sapply(thresholds, function(threshold) {
    predicted <- ifelse(y_pred_prob >= threshold, 1, 0)
    predicted = factor(predicted, levels = c(0,1))
    cm <- confusionMatrix(predicted, y_true, positive = "1")
    precision <- cm$byClass["Precision"]
    recall <- cm$byClass["Recall"]
    f1_score(precision, recall)
  })
  
  optimal_threshold_f1 <- thresholds[which.max(f1_scores)]
  
  return(list(optimal_threshold_roc = optimal_threshold_roc, 
              optimal_threshold_roc_dist = optimal_threshold_roc_dist,
              optimal_threshold_f1 = optimal_threshold_f1))
}

# Apply the opt_threshold function row-wise on y_pred_prob matrix
apply_opt_threshold <- function(y_pred_prob_matrix, y_true) {
  results <- apply(y_pred_prob_matrix, 1, function(row) {
    opt_threshold(row, y_true)
  })
  
  optimal_thresholds_roc <- sapply(results, function(res) res$optimal_threshold_roc)
  optimal_thresholds_roc_dist <- sapply(results, function(res) res$optimal_threshold_roc_dist)
  optimal_thresholds_f1 <- sapply(results, function(res) res$optimal_threshold_f1)
  
  return(list(optimal_thresholds_roc = optimal_thresholds_roc, 
              optimal_thresholds_roc_dist = optimal_thresholds_roc_dist,
              optimal_thresholds_f1 = optimal_thresholds_f1))
}


### Compute AUC ### 

library(pROC)
library(PRROC)

compute_auc <- function(pred_mat, y_true, compute_pr = TRUE, method_names = NULL) {
  # Coerce to matrix
  pred_mat <- as.matrix(pred_mat)
  stopifnot(length(y_true) == ncol(pred_mat))
  
  
  n_methods <- nrow(pred_mat)
  auc_roc <- numeric(n_methods)
  auc_pr  <- rep(NA_real_, n_methods)
  
  for (i in seq_len(n_methods)) {
    preds <- pred_mat[i, ]
    
    # ROC–AUC (set levels/direction explicitly)
    roc_obj <- roc(response = y_true, predictor = preds, quiet = TRUE)
    auc_roc[i] <- as.numeric(auc(roc_obj))
    
    # PR–AUC (positive class = 1 goes to scores.class0)
    if (compute_pr) {
      pr_obj <- pr.curve(scores.class0 = preds[y_true == 1],
                         scores.class1 = preds[y_true == 0],
                         curve = FALSE)
      auc_pr[i] <- pr_obj$auc.integral
    }
  }
  
  # Method names logic
  if (!is.null(method_names)) {
    stopifnot(length(method_names) == n_methods)
    methods <- method_names
  } else if (!is.null(rownames(pred_mat))) {
    methods <- rownames(pred_mat)
  } else if (n_methods == 2) {
    methods <- c("LASSO", "BART")
  } else if (n_methods == 3) {
    methods <- c("LASSO", "BART", "SSLASSO")
  } else {
    methods <- paste0("Method_", seq_len(n_methods))
  }
  
  data.frame(method = methods, AUC_ROC = auc_roc, AUC_PR = auc_pr, row.names = NULL)
}

plot_roc <- function(y_pred, y_true, save = NULL){
  # plot(res[[1]]~as.factor(y_train))
  if (!is.null(save)) {
    # Construct the file name with .png extension
    file_name <- paste0(save, ".pdf")
    pdf(file_name,pointsize=15)
  }
  roc_obj <- roc(y_true, y_pred)
  auc_roc = auc(roc_obj)
  plot(roc_obj, 
       legacy.axes = TRUE, # False positive rate (not by specificity)
       col = "steelblue", lwd = 2,
       main = bquote("ROC Curve" ~ (AUC == .(round(auc_roc, 3)))),
       xlab = "False Positive Rate", 
       ylab = "True Positive Rate")
  abline(a = 0, b = 1, lty = 2, col = "gray60")
  
  pr_obj <- pr.curve(scores.class0 = y_pred[y_true == 1],
                     scores.class1 = y_pred[y_true == 0],
                     curve = TRUE)
  auc_pr <- pr_obj$auc.integral
  if (!is.null(save)) {dev.off()}
  return(c(auc_roc, auc_pr))
}
