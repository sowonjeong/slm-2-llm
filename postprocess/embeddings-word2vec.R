library(reticulate)

setwd("..")
source("utils/utils.R") 
source("utils/threshold.R")

A = load_federalist_authors()
authors = A$authors
select = A$select
authors_joint = A$authors_joint
authors_train = A$authors_train
y_train = authors_train

paper_nums = c(1:85)

tdm1 = load_federalist(type = 1, lemmatize = FALSE) # 8601 # 5894
tdm2 = load_federalist(type = 2, lemmatize = FALSE) # 8723 # 5996
tdm3 = load_federalist(type = 3, lemmatize = FALSE) # 173 # 145


## Word2vec
use_condaenv('py3.11', required = TRUE) # you need to import your own conda environment to use python in R
gensim <- import("gensim")

# Download the pre-trained Word2Vec model if you haven't already
# Load the model path
model_path = "/Users/sowonjeong/txt-analysis/pre-trained-model/GoogleNews-vectors-negative300.bin"
word2vec_model = gensim$models$KeyedVectors$load_word2vec_format(model_path, binary=TRUE)

word2vec_model['upon']

# Accessing a word vector
word <- "king"
vector <- word2vec_model$get_vector(word)

# Function to compute cosine similarity between two words
cosine_similarity <- function(word1, word2, model) {
  vector1 <- model$get_vector(word1)
  vector2 <- model$get_vector(word2)
  similarity <- sum(vector1 * vector2) / (sqrt(sum(vector1 * vector1)) * sqrt(sum(vector2 * vector2)))
  return(similarity)
}

# Compute cosine similarity
cosine_similarity("upon","on", word2vec_model)
cosine_similarity("while","whilst", word2vec_model)
cosine_similarity("while","while", word2vec_model)
cosine_similarity("upon","while", word2vec_model)
cosine_similarity("king","queen", word2vec_model)
cosine_similarity("men","women", word2vec_model)




# Find most similar words
# similar_words <- word2vec_model$most_similar("upon")

is_in_model <- function(word, model) {
  tryCatch({
    model$get_vector(word)
    TRUE
  }, error = function(e) {
    FALSE
  })
}

word2vec_embeddings = function(tdm, model){
  words_not_in_model <- dimnames(tdm)$Terms[!sapply( dimnames(tdm)$Terms, is_in_model, model = model)]
  words_in_model = dimnames(tdm)$Terms[sapply( dimnames(tdm)$Terms, is_in_model, model =model)]
  w2v = sapply(words_in_model,model$get_vector )
  w2v_doc = as.matrix(tdm[,words_in_model]/rowSums(as.matrix(tdm[,words_in_model]))) %*% t(w2v)
  return(list(words_in_model, words_not_in_model, w2v,w2v_doc ))
}

w2v1 = word2vec_embeddings(tdm1, word2vec_model)
w2v2 = word2vec_embeddings(tdm2, word2vec_model)
w2v3 = word2vec_embeddings(tdm3, word2vec_model)

bart_w2v1 =  run_bart_binary(w2v1[[4]][select,], w2v1[[4]][authors == 'UNKNOWN',], authors_train)
bart_w2v2 =  run_bart_binary(w2v2[[4]][select,], w2v2[[4]][authors == 'UNKNOWN',], authors_train)
bart_w2v3 =  run_bart_binary(w2v3[[4]][select,], w2v3[[4]][authors == 'UNKNOWN',], authors_train)

lasso_w2v1 =  run_lasso_binary(w2v1[[4]][select,], w2v1[[4]][authors == 'UNKNOWN',], authors_train)
lasso_w2v2 =  run_lasso_binary(w2v2[[4]][select,], w2v2[[4]][authors == 'UNKNOWN',], authors_train)
lasso_w2v3 =  run_lasso_binary(w2v3[[4]][select,], w2v3[[4]][authors == 'UNKNOWN',], authors_train)


bart_w2v1_joint =  run_bart_binary(w2v1[[4]][select,], w2v1[[4]][authors == 'HAMILTON AND MADISON',], authors_train)
bart_w2v2_joint =  run_bart_binary(w2v2[[4]][select,], w2v2[[4]][authors == 'HAMILTON AND MADISON',], authors_train)
bart_w2v3_joint =  run_bart_binary(w2v3[[4]][select,], w2v3[[4]][authors == 'HAMILTON AND MADISON',], authors_train)

plot_dist_joint(bart_w2v1, bart_w2v1_joint, authors_train)
plot_dist_joint(bart_w2v2, bart_w2v2_joint, authors_train)
plot_dist_joint(bart_w2v3, bart_w2v3_joint, authors_train)

plot_dist(bart_w2v1, authors_train, save = "w2v1")
plot_dist(bart_w2v2, authors_train, save= "w2v2")
plot_dist(bart_w2v3, authors_train, save = "w2v3")

oos_w2v1 = run_oos_cv(w2v1[[4]][select,], y_train)
oos_w2v2 = run_oos_cv(w2v2[[4]][select,], y_train)
oos_w2v3 = run_oos_cv(w2v3[[4]][select,], y_train)

l2error(oos_bow1,y_train)
l2error(oos_bow2,y_train)
l2error(oos_bow3,y_train)


l2error(oos_w2v1, y_train)
l2error(oos_w2v2, y_train)
l2error(oos_w2v3, y_train)


apply(oos_w2v1, 1, function(row) opt_threshold(row, authors_train))
clferror(oos_w2v1, c = 0.2243202, y_train) # roc
clferror(oos_w2v1, c = 0.3379352, y_train) # F1

apply(oos_w2v2, 1, function(row) opt_threshold(row, authors_train))
clferror(oos_w2v2, c = 0.25, y_train)

apply(oos_w2v3, 1, function(row) opt_threshold(row, authors_train))
clferror(oos_w2v3, c =  0.21953, y_train) # roc
clferror(oos_w2v3, c =  0.380073, y_train) # F1
