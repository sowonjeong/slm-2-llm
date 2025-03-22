library(wordcloud)
library(ggplot2)
library(reshape2)
library(dplyr)

# Create word cloud

wordcloud(words = names(lasso1[[3]][-1]), 
          freq = abs(lasso1[[3]][-1]), 
          max.words = 200, random.order = FALSE, rot.per = 0.35, 
          scale = c(4, 0.5), # Adjust scale to make the word cloud more readable
          colors = brewer.pal(8, "Dark2"))

wordcloud(words = names(lasso2[[3]][-1]), 
          freq = abs(lasso2[[3]][-1]), 
          max.words = 200, random.order = FALSE, rot.per = 0.35, 
          scale = c(4, 0.5), # Adjust scale to make the word cloud more readable
          colors = brewer.pal(8, "Dark2"))

wordcloud(words = names(lasso3[[3]][-1]), 
          freq = abs(lasso3[[3]][-1]), 
          max.words = 200, random.order = FALSE, rot.per = 0.35, 
          scale = c(4, 0.5), # Adjust scale to make the word cloud more readable
          colors = brewer.pal(8, "Dark2"))

wordcloud(words = names(sslasso2[[3]][-1]), 
          freq = abs(sslasso2[[3]][-1]), 
          max.words = 200, random.order = FALSE, rot.per = 0.35, 
          scale = c(4, 0.5), # Adjust scale to make the word cloud more readable
          colors = brewer.pal(8, "Dark2"))


# oos_prediction = read.table("/Users/sowonjeong/txt-analysis/data/CV_result2.txt")
# lasso bow
# bart bow
# lasso lda
# bart lda
# bart w2v
# bart sentence
# bart chunk

l2error(oos_prediction, y_train)

setwd("/Users/sowonjeong/txt-analysis")

name = c("bow1","bow2","bow3",
         #"bow_bh","bow_ebh","bow_bf","bow_HC",
         "lda1","lda2","lda3",
         "svd1","svd2","svd3",
         "nmf1","nmf2","nmf3",
         "w2v1","w2v2","w2v3",
         "bert","roberta","bart",
         "bert_ft","roberta_ft","bart_ft",
         "gpt","llama2","llama3")
sum_oos_l2 = matrix(0, nrow = 3, ncol = length(name))
for (i in 1:3){
  for (j in (1:length(name))){
    tab_name = paste0("oos_",name[j])
    sum_oos_l2[i,j] = l2error(get(tab_name), y_train)[i]
  }
}

colnames(sum_oos_l2)=name
rownames(sum_oos_l2)=c("lasso","bart","sslasso")
write.table(sum_oos_l2, "sum_oos_l2.txt")

sum_oos_clf = matrix(0, nrow = 3, ncol = length(name))
sum_oos_clf_roc = matrix(0, nrow = 3, ncol = length(name))
sum_oos_clf_f1 = matrix(0, nrow = 3, ncol = length(name))
thr_roc = matrix(0, nrow = 3, ncol = length(name))
thr_f1 = matrix(0, nrow = 3, ncol = length(name))
for (i in 1:3){
  for (j in (1:length(name))){
    tab_name = paste0("oos_",name[j])
    err = clferror(get(tab_name), y_train)
    thresholds = apply_opt_threshold(get(tab_name), y_train)
    sum_oos_clf[,j] = clferror(get(tab_name),y_train,c = 0.3)
    sum_oos_clf_roc[i,j] = err$error_roc[i]
    sum_oos_clf_f1[i,j] = err$error_f1[i]
    thr_roc[i,j] = thresholds$optimal_thresholds_roc[i]
    thr_f1[i,j] = thresholds$optimal_thresholds_f1[i]
  }
}

colnames(sum_oos_clf) = name
colnames(sum_oos_clf_roc) = name
colnames(sum_oos_clf_f1) = name
colnames(thr_roc) = name
colnames(thr_f1) = name
rownames(sum_oos_clf_roc)=c("lasso","bart","sslasso")
rownames(sum_oos_clf_f1)=c("lasso","bart","sslasso")
rownames(thr_roc)=c("lasso","bart","sslasso")
rownames(thr_f1)=c("lasso","bart","sslasso")


write.table(sum_oos_clf, "sum_oos_clf.txt")
write.table(sum_oos_clf_roc, "sum_oos_clf_roc.txt")
write.table(sum_oos_clf_f1, "sum_oos_clf_f1.txt")
write.table(thr_roc, "thr_roc.txt")
write.table(thr_f1, "thr_f1.txt")

# test-evaluation

sum_l2 = matrix(0, nrow = 2, ncol = length(name))
sum_clf = matrix(0, nrow = 2, ncol = length(name))
sum_clf_roc = matrix(0, nrow = 2, ncol = length(name))
sum_clf_f1 = matrix(0, nrow = 2, ncol = length(name))
for (i in 1:2){
  for (j in (1:length(name))){
    tab_name_oos = paste0("oos_",name[j])
    if(i==1){
    tab_name = paste0("lasso_",name[j])}
    else{tab_name = paste0("bart_",name[j])}
    thresholds = apply_opt_threshold(get(tab_name_oos), y_train)
    sum_l2[i,j] = l2error(t(as.matrix(get(tab_name)[[2]])), 1)
    sum_clf[i,j] = clferror(t(as.matrix(get(tab_name)[[2]])),1,c = 0.3)
    sum_clf_roc[i,j] = clferror(t(as.matrix(get(tab_name)[[2]])),1,c = thresholds$optimal_thresholds_roc[i])
    sum_clf_f1[i,j] = clferror(t(as.matrix(get(tab_name)[[2]])),1,c = thresholds$optimal_thresholds_f1[i])
  }
}

colnames(sum_l2) = name
colnames(sum_clf) = name
colnames(sum_clf_roc) = name
colnames(sum_clf_f1) = name
rownames(sum_l2)=c("lasso","bart")
rownames(sum_clf)=c("lasso","bart")
rownames(sum_clf_roc)=c("lasso","bart")
rownames(sum_clf_f1)=c("lasso","bart")

write.table(sum_l2,"sum_l2.txt")
write.table(sum_clf, "sum_clf.txt")
write.table(sum_clf_roc, "sum_clf_roc.txt")
write.table(sum_clf_f1, "sum_clf_f1.txt")




# predicted prob
predicted_prob = matrix(0, nrow = length(name), ncol = 12)
for (i in (1:length(name))){
  tab_name = paste0("bart_",name[i])
  predicted_prob[i,] = get(tab_name)[[2]]
}

colnames(predicted_prob) = paste0("No.",which(authors=="UNKNOWN"))
rownames(predicted_prob)= name
predicted_prob

predicted_prob_joint = matrix(0, nrow = length(name), ncol = 3)
for (i in (1:length(name))){
  tab_name = paste0("bart_",name[i],"_joint")
  predicted_prob_joint[i,] = get(tab_name)[[2]]
}
colnames(predicted_prob_joint) = paste0("No.",which(authors=="HAMILTON AND MADISON"))
rownames(predicted_prob_joint)= name

write.table(predicted_prob_joint, "predicted_prob_joint.txt")

