## Federalist paper application
library(textir) # to get the data
library(maptpx) # for the topics function
library(tm) 
library(slam) # to preprocess the data into vector(tokenize) 
library(NLP)
library(syllogi)
library(reticulate)
library(reticulate) # call python function
# library(NNLM)
# devtools::install_github("linxihui/NNLM")
# NMF package is very slow for large/sparse matrix

source("/Users/sowonjeong/txt-analysis/data/utils.R") # NEED to specify the path
source("/Users/sowonjeong/txt-analysis/threshold.R")

A = load_federalist_authors()
authors = A$authors
select = A$select
authors_joint = A$authors_joint
authors_train = A$authors_train
y_train = authors_train

paper_nums = c(1:85)

tdm1 = load_federalist(type = 1) # 8601 # 5894
tdm2 = load_federalist(type = 2) # 8723 # 5996
tdm3 = load_federalist(type = 3) # 173 # 145


## Topic modeling

tpcs1 <- topics(as.simple_triplet_matrix(tdm1),K=5*(1:5), verb=10) # it chooses 5 topics 
tpcs2 <- topics(as.simple_triplet_matrix(tdm2),K=5*(1:5), verb=10) # it chooses 5 topics 
tpcs3 <- topics(as.simple_triplet_matrix(tdm3),K=5*(1:5), verb=10) # it chooses 5 topics 

rownames(tpcs1$theta)[order(tpcs1$theta[,1], decreasing=TRUE)[1:10]]
rownames(tpcs2$theta)[order(tpcs2$theta[,1], decreasing=TRUE)[1:10]]
rownames(tpcs3$theta)[order(tpcs3$theta[,1], decreasing=TRUE)[1:10]]


## Numerical Decomposition Approach
## Nonnegative Matrix Factorization
# NMF R package has an issue -- so I load the computed decomposition from python
# W = read.csv("NMF_W.csv", header = FALSE) # topic-word 85 x 10
# H = read.csv("NMF_H.csv", header = FALSE) # document-topic 10 x 8382
# Run sklearn in R
use_condaenv('py3.11', required = TRUE)
sklearn <- import("sklearn")
model1 <- sklearn$decomposition$NMF(
  n_components = 10L,  # number of topics
  random_state  =  1L, # equivalent of seed for reproducibility
  max_iter = 1000L
)$fit(as.matrix(tdm1))
model2 <- sklearn$decomposition$NMF(
  n_components = 10L,  # number of topics
  random_state  =  1L, # equivalent of seed for reproducibility
  max_iter = 5000L
)$fit(as.matrix(tdm2))
model3 <- sklearn$decomposition$NMF(
  n_components = 10L,  # number of topics
  random_state  =  1L, # equivalent of seed for reproducibility
  max_iter = 2000L
)$fit(as.matrix(tdm3))

W1 = model1$fit_transform(as.matrix(tdm1))
#H = model$components_
W2 = model2$fit_transform(as.matrix(tdm2))
W3 = model3$fit_transform(as.matrix(tdm3))

# Latent Semantic Analysis
svd_res1 = svd(as.matrix(tdm1), nu = 10, nv = 10)
svd_res2 = svd(as.matrix(tdm2), nu = 10, nv = 10)
svd_res3 = svd(as.matrix(tdm3), nu = 10, nv = 10)


# BART
library(BART)

bart_bow1 =  run_bart_binary(as.matrix(tdm1[select,]), as.matrix(tdm1[authors == 'UNKNOWN',]), authors_train)
bart_bow2 =  run_bart_binary(as.matrix(tdm2[select,]), as.matrix(tdm2[authors == 'UNKNOWN',]), authors_train)
bart_bow3 =  run_bart_binary(as.matrix(tdm3[select,]), as.matrix(tdm3[authors == 'UNKNOWN',]), authors_train)
#LDA
bart_lda1 = run_bart_binary(as.matrix(tpcs1$omega[select,]), as.matrix(tpcs1$omega[authors == 'UNKNOWN',]), authors_train)
bart_lda_joint1 = run_bart_binary(as.matrix(tpcs1$omega[select,]), as.matrix(tpcs1$omega[authors == 'HAMILTON AND MADISON',]), authors_train)

bart_lda2 = run_bart_binary(as.matrix(tpcs2$omega[select,]), as.matrix(tpcs2$omega[authors == 'UNKNOWN',]), authors_train)
bart_lda_joint2 = run_bart_binary(as.matrix(tpcs2$omega[select,]), as.matrix(tpcs2$omega[authors == 'HAMILTON AND MADISON',]), authors_train)

bart_lda3 = run_bart_binary(as.matrix(tpcs3$omega[select,]), as.matrix(tpcs3$omega[authors == 'UNKNOWN',]), authors_train)
bart_lda_joint3 = run_bart_binary(as.matrix(tpcs3$omega[select,]), as.matrix(tpcs3$omega[authors == 'HAMILTON AND MADISON',]), authors_train)


#SVD
bart_svd1 = run_bart_binary(as.matrix(svd_res1$u[select, ]), as.matrix(svd_res1$u[authors == 'UNKNOWN',]), authors_train)
bart_svd2 = run_bart_binary(as.matrix(svd_res2$u[select, ]), as.matrix(svd_res2$u[authors == 'UNKNOWN',]), authors_train)
bart_svd3 = run_bart_binary(as.matrix(svd_res3$u[select, ]), as.matrix(svd_res3$u[authors == 'UNKNOWN',]), authors_train)

bart_svd_joint3 = run_bart_binary(as.matrix(svd_res3$u[select, ]), as.matrix(svd_res3$u[authors == 'HAMILTON AND MADISON',]), authors_train)

#NMF
bart_nmf1 = run_bart_binary(as.matrix(W1[select,]), as.matrix(W1[authors == 'UNKNOWN',]), authors_train)
bart_nmf2 = run_bart_binary(as.matrix(W2[select,]), as.matrix(W2[authors == 'UNKNOWN',]), authors_train)
bart_nmf3 = run_bart_binary(as.matrix(W3[select,]), as.matrix(W3[authors == 'UNKNOWN',]), authors_train)

bart_nmf_joint = run_bart_binary(as.matrix(W[select,]), as.matrix(W[authors == 'HAMILTON AND MADISON',]), authors_train)

# LASSO
lasso1 = run_lasso_binary(as.matrix(tdm1[select,]), as.matrix(tdm1[authors == 'UNKNOWN',]), authors_train)
lasso2 = run_lasso_binary(as.matrix(tdm2[select,]), as.matrix(tdm2[authors == 'UNKNOWN',]), authors_train)
lasso3 = run_lasso_binary(as.matrix(tdm3[select,]), as.matrix(tdm3[authors == 'UNKNOWN',]), authors_train)

#LDA
lasso_lda1 = run_lasso_binary(as.matrix(tpcs1$omega[select,]), as.matrix(tpcs1$omega[authors == 'UNKNOWN',]), authors_train)
#SVD
lasso_svd = run_lasso_binary(as.matrix(svd_res$u[select, ]), as.matrix(svd_res$u[authors == 'UNKNOWN',]), authors_train)
#NMF
lasso_nmf = run_lasso_binary(as.matrix(W[select,]), as.matrix(W[authors == 'UNKNOWN',]), authors_train)


## Quick test of other embeddings
bart_w2v_joint = run_bart_binary(word_df[select,], word_df[authors=="HAMILTON AND MADISON",],authors_train)
bart_chunk_doc_joint = run_bart_binary(avg_chunk_df[select,], avg_chunk_df[authors=="HAMILTON AND MADISON",],authors_train)
bart_sentence_doc_joint = run_bart_binary(avg_sentence_df[select,], avg_sentence_df[authors=="HAMILTON AND MADISON",],authors_train)

plot_dist_joint(bart_lda1, bart_lda_joint1, authors_train)
plot_dist_joint(bart_lda2, bart_lda_joint2, authors_train)
plot_dist_joint(bart_lda3, bart_lda_joint3, authors_train)

plot_dist(bart_bow1, authors_train)
plot_dist(bart_bow2, authors_train)
plot_dist(bart_bow3, authors_train)

plot_dist(bart_svd1, authors_train)
plot_dist(bart_svd2, authors_train)
plot_dist(bart_svd3, authors_train)

plot_dist(bart_nmf1, authors_train)
plot_dist(bart_nmf2, authors_train)
plot_dist(bart_nmf3, authors_train)


plot_dist(bart_lda_joint1, authors_train)
plot_dist(bart_lda_joint2, authors_train)
plot_dist(bart_lda_joint3, authors_train)


plot_dist(bart_svd_joint, authors_train)
plot_dist(bart_nmf_joint, authors_train)

plot_dist(bart_w2v_joint, authors_train)
plot_dist(bart_chunk_doc_joint, authors_train)
plot_dist(bart_sentence_doc_joint, authors_train)

oos_bow1 = run_oos_cv(as.matrix(tdm1[select,]),y_train)
oos_bow2 = run_oos_cv(as.matrix(tdm2[select,]),y_train)
oos_bow3 = run_oos_cv(as.matrix(tdm3[select,]),y_train)

oos_lda1 = run_oos_cv(as.matrix(tpcs1$omega[select,]), y_train)
oos_lda2 = run_oos_cv(as.matrix(tpcs2$omega[select,]), y_train)
oos_lda3 = run_oos_cv(as.matrix(tpcs3$omega[select,]), y_train)

oos_svd1 = run_oos_cv(as.matrix(svd_res1$u[select, ]), y_train)
oos_svd2 = run_oos_cv(as.matrix(svd_res2$u[select, ]), y_train)
oos_svd3 = run_oos_cv(as.matrix(svd_res3$u[select, ]), y_train)

oos_nmf1 = run_oos_cv(W1[select,], y_train)
oos_nmf2 = run_oos_cv(W2[select,], y_train)
oos_nmf3 = run_oos_cv(W3[select,], y_train)


write.table(oos_prediction,"CV_result2.txt")
oos_prediction = read.table("CV_result2.txt")
result_prev = read.table('/Users/sowonjeong/txt-analysis/CV_result.txt')


threshold = seq(from= 0, to= 1, by = 0.05)

l2error(oos_lda1, y_train)
l2error(oos_lda2, y_train)
l2error(oos_lda3, y_train)

l2error(oos_svd1, y_train)
l2error(oos_svd2, y_train)
l2error(oos_svd3, y_train)

l2error(oos_nmf1, y_train)
l2error(oos_nmf2, y_train)
l2error(oos_nmf3, y_train)


l2error(oos_nmf, y_train)

apply(oos_lda3, 1, function(row) opt_threshold(row, authors_train))

clferror(oos_lda1, y_train)
clferror(oos_lda1, c = 0.4, y_train)

clferror(oos_lda2, c = 0.2, y_train)
clferror(oos_lda3, c = 0.5, y_train)

clferror(oos_svd, c = 0.5, y_train)
clferror(oos_svd3, c = 0.3, y_train)

clferror(oos_nmf, c = 0.5, y_train)



