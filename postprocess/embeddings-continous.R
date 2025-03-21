## Federalist paper application
library(textir) # to get the data
library(maptpx) # for the topics function
library(tm) 
library(slam) # to preprocess the data into vector(tokenize) 
library(NLP)
library(syllogi)
library(NMF)

source("/Users/sowonjeong/txt-analysis/data/utils.R") # NEED to specify the path

# Read the CSV file
setwd("/Users/sowonjeong/Dropbox/LLM_FEDERALIST/data")
setwd("/Users/sowonjeong/txt-analysis/data")

# 1041 x 768
chunk_df <- read.csv("chunk_embedding_200.csv", header = FALSE)
chunk_author <- read.csv("chunking_author_200.csv")
avg_chunk_df <- read.csv("avg_chunk_embedding.csv", header = FALSE)

# 5738 x 768
sentence_df <- read.csv("sent_embedding.csv", header = FALSE)
sentence_author <- read.csv("sentence_author.csv")
avg_sentence_df <- read.csv("avg_sent_embedding.csv", header = FALSE)


# 5738 x 768
bert_sentence_df <- read.csv("bert_sentence_embedding.csv", header = FALSE)
bart_sentence_df <- read.csv("bart_sentence_embedding.csv", header = FALSE)
# 5738 x 4096
llama2_sentence_df <- read.csv("llama2_sentence_embedding.csv", header = FALSE)
# 5738 x 4096
llama3_sentence_df <- read.csv("llama3_sentence_embedding.csv", header = FALSE)

roberta_sentence_df = read.csv("roberta_sentence_embedding.csv", header = FALSE)
gpt_sentence_df = read.csv("gpt_sent_embeddings.csv",header=FALSE)
sentence_author <- read.csv("sentence_author.csv")


# 85 x feat
bert_df = read.csv("bert_avg_doc_embedding.csv", header = FALSE)
bart_df = read.csv("bart_avg_doc_embedding.csv", header = FALSE)
llama2_df = read.csv("llama2_avg_doc_embedding.csv", header = FALSE)
llama3_df = read.csv("llama3_avg_doc_embedding.csv", header = FALSE)
gpt_df = read.csv("gpt_embeddings.csv", header = FALSE)
gpt_df = gpt_df[-1,-1]
roberta_df = read.csv("roberta_avg_doc_embedding.csv",header = FALSE)

authors = rep(0,85)

authors[c(1, 6:9, 11:13, 15:17, 21:36, 59:61, 65:85)] = "HAMILTON"

authors[c(10, 14, 37:48)] = "MADISON"

authors[c(18:20)] = "HAMILTON AND MADISON"

authors[c(2:5, 64)] = "JAY"

authors[c(49:58, 62:63)] = "UNKNOWN"

# Let's focus only on M and H papers as a binary classification for now

select<-authors=="HAMILTON"|authors=="MADISON"
sentence_select <- (sentence_author[,1]=="HAMILTON"|sentence_author[,1]=="MADISON")
sentence_dispute <-sentence_author[,1]=="UNKNOWN"

# set up y variable

authors_train<-authors[select]

authors_train=as.numeric(authors_train=="MADISON")

authors_sentence_train = sentence_author[sentence_select,1]


# BART
library(BART)
# doc_embeddings
bart_bert = run_bart_binary(as.matrix(bert_df[select,]), as.matrix(bert_df[authors == 'UNKNOWN',]), authors_train)
bart_bert_joint = run_bart_binary(as.matrix(bert_df[select,]), as.matrix(bert_df[authors == 'HAMILTON AND MADISON',]), authors_train)

bart_llama2 = run_bart_binary(as.matrix(llama2_df[select,]), as.matrix(llama2_df[authors == 'UNKNOWN',]), authors_train)
bart_llama2_joint = run_bart_binary(as.matrix(llama2_df[select,]), as.matrix(llama2_df[authors == 'HAMILTON AND MADISON',]), authors_train)

bart_llama3 = run_bart_binary(as.matrix(llama3_df[select,]), as.matrix(llama3_df[authors == 'UNKNOWN',]), authors_train)
bart_llama3_joint = run_bart_binary(as.matrix(llama3_df[select,]), as.matrix(llama3_df[authors == 'HAMILTON AND MADISON',]), authors_train)

bart_bart = run_bart_binary(as.matrix(bart_df[select,]), as.matrix(bart_df[authors == 'UNKNOWN',]), authors_train)
bart_bart_joint = run_bart_binary(as.matrix(bart_df[select,]), as.matrix(bart_df[authors == 'HAMILTON AND MADISON',]), authors_train)

bart_gpt = run_bart_binary(as.matrix(gpt_df[select,]), as.matrix(gpt_df[authors == 'UNKNOWN',]), authors_train)
bart_gpt_joint = run_bart_binary(as.matrix(gpt_df[select,]), as.matrix(gpt_df[authors == 'HAMILTON AND MADISON',]), authors_train)

bart_roberta = run_bart_binary(as.matrix(roberta_df[select,]), as.matrix(roberta_df[authors == 'UNKNOWN',]), authors_train)
bart_roberta_joint = run_bart_binary(as.matrix(roberta_df[select,]), as.matrix(roberta_df[authors == 'HAMILTON AND MADISON',]), authors_train)


# LASSO
lasso_bert = run_lasso_binary(as.matrix(bert_df[select,]), as.matrix(bert_df[authors == 'UNKNOWN',]), authors_train)
lasso_bart = run_lasso_binary(as.matrix(bart_df[select,]), as.matrix(bart_df[authors == 'UNKNOWN',]), authors_train)
lasso_llama2 = run_lasso_binary(as.matrix(llama2_df[select,]), as.matrix(llama2_df[authors == 'UNKNOWN',]), authors_train)
lasso_llama3 = run_lasso_binary(as.matrix(llama3_df[select,]), as.matrix(llama3_df[authors == 'UNKNOWN',]), authors_train)
lasso_gpt = run_lasso_binary(as.matrix(gpt_df[select,]), as.matrix(gpt_df[authors == 'UNKNOWN',]), authors_train)



plot_dist(bart_bert, authors_train)
plot_dist(bart_llama2, authors_train)
plot_dist(bart_llama3, authors_train)
plot_dist(bart_bart, authors_train)
plot_dist(bart_gpt, authors_train)

plot_dist(bart_bert_joint, authors_train)
plot_dist(bart_llama2_joint, authors_train)
plot_dist(bart_llama3_joint, authors_train)
plot_dist(bart_bart_joint, authors_train)
plot_dist(bart_gpt_joint, authors_train)

y_train_sentence = authors_sentence_train
y_train_sentence = as.integer(y_train_sentence)

oos_bert = run_oos_cv(as.matrix(bert_df[select,]), y_train)
oos_bart = run_oos_cv(bart_df[select,],y_train)
oos_llama2 = run_oos_cv(as.matrix(llama2_df[select, ]), y_train)
oos_llama3 = run_oos_cv(llama3_df[select,], y_train)
oos_gpt = run_oos_cv(gpt_df[select,],y_train)
oos_roberta = run_oos_cv(roberta_df[select,],y_train)

authors_sentence_train = as.numeric(authors_sentence_train == "MADISON")
oos_bert_sentence = run_oos_cv(as.matrix(bert_sentence_df[sentence_select,]), authors_sentence_train)
oos_bart_sentence = run_oos_cv(as.matrix(bart_sentence_df[sentence_select,]), authors_sentence_train)
oos_llama2_sentence = run_oos_cv(as.matrix(llama2_sentence_df[sentence_select,]), authors_sentence_train)
oos_llama3_sentence = run_oos_cv(as.matrix(llama3_sentence_df[sentence_select,]), authors_sentence_train)
oos_gpt_sentence = run_oos_cv(as.matrix(gpt_sentence_df[sentence_select,]), authors_sentence_train)
oos_roberta_sentence = run_oos_cv(as.matrix(roberta_sentence_df[sentence_select,]), authors_sentence_train)


l2error(oos_bert, y_train)
l2error(oos_llama2, y_train)
l2error(oos_llama3, y_train)
l2error(oos_bart, y_train)

clferror(oos_bert, c = 0.5, y_train)
apply_opt_threshold(oos_bert,y_train)
apply(apply(oos_bert, 1, function(x) (as.numeric(x) > 0.1722236 ) != y_train),2,mean)
clferror(oos_bert, y_train)
clferror(oos_llama2, c = 0.5, y_train)
clferror(oos_llama3, c = 0.5, y_train)
clferror(oos_bart, c = 0.5, y_train)

oos_prediction = read.table("/Users/sowonjeong/txt-analysis/data/CV_result2.txt")
# lasso bow
# bart bow
# lasso lda
# bart lda
# bart w2v
# bart sentence
# bart chunk

l2error(oos_prediction, y_train)

name = c("bert","bart","llama2","llama3")
sum_oos_l2 = matrix(0, nrow = 2, ncol = 4)
for (i in 1:2){
    for (j in (1:4)){
      tab_name = paste0("oos_",name[j])
      sum_oos_l2[i,j] = l2error(get(tab_name), y_train)[i]
    }
}


sum_oos_l2

