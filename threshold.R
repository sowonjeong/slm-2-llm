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
              optimal_threshold_f1 = optimal_threshold_f1))
}

# Apply the opt_threshold function row-wise on y_pred_prob matrix
apply_opt_threshold <- function(y_pred_prob_matrix, y_true) {
  results <- apply(y_pred_prob_matrix, 1, function(row) {
    opt_threshold(row, y_true)
  })
  
  optimal_thresholds_roc <- sapply(results, function(res) res$optimal_threshold_roc)
  optimal_thresholds_f1 <- sapply(results, function(res) res$optimal_threshold_f1)
  
  return(list(optimal_thresholds_roc = optimal_thresholds_roc, 
              optimal_thresholds_f1 = optimal_thresholds_f1))
}

