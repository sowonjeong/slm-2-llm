library(BART)
library(textir) # to get the data
library(tm) 
library(slam) # to preprocess the data into vector(tokenize) 
library(NLP)
library(syllogi)
library(textstem)
# library(SSLASSO)

source("utils/threshold.R")

# Test for the lemmatization
vector <- c("run", "ran", "running")
lemmatize_words(vector)

MW_final = c("upon", "also" ,"a", "by" , "of", "on", "there", "this", "to", "although", "both", "enough", 
              "while", "whilst", "always", "though", "commonly", "consequently", "considerable",
              "according", "apt", "direction", "innovation", "language", "vigor", "kind",
              "matter", "particularly", "probability", "work")

function_words = c( 'a','as','do','has','is','no','or','than','this','when',
                    'all','at','down','have','it','not','our','that','to','which',
                    'also','be','even','her','its','now','shall','the','up','who',
                    'an','been','every','his','may','of','should','their','upon','will',
                    'and','but','for','if','more','on','so','then','was','with',
                    'any','by','from','in','must','one','some','there','were','would',
                    'are','can','had','into','my','only','such','thing','what','your')

MW_words_list = c(
######################## Function Words ########################
 'a','as','do','has','is','no','or','than','this','when',
 'all','at','down','have','it','not','our','that','to','which',
 'also','be','even','her','its','now','shall','the','up','who',
 'an','been','every','his','may','of','should','their','upon','will',
 'and','but','for','if','more','on','so','then','was','with',
 'any','by','from','in','must','one','some','there','were','would',
 'are','can','had','into','my','only','such','thing','what','your',

######################## Additional Set 1 ########################
'affect','city','direction','innovation','perhaps','vigor',
'again','commonly','disgracing','join','rapid','violate','although',
'consequently','either','language','same','violence','among','considerable',
'enough','most','second','voice','another','contribute','nor','still',
'where','because','defensive','fortune','offensive','those','whether',
'between','destruction','function','often','throughout', 'while','both',
'did','himself','pass','under','whilst',

######################## Additional Set 2 ########################
'about','choice','instruct', 'proper','according','common','kind','propriety','adversaries',
'danger','large','provision','after','decide','decides','decided','deciding',
'likely','requisite','aid','degree','matters','matter','substance','always',
'during','moreover','they','apt','expence','necessary','though',
'asserted','expenses','expense','necessity','necessities','truth','truths',
'before','extent','others','us','being','follows','follow','particularly',
'usages','usage','better','i','principle','we','care','imagine',
'probability','work'
)

length(MW_words_list)

# Function to remove specific symbols
removeSpecificSymbols <- function(x, symbols) {
  for (symbol in symbols) {
    x <- gsub(symbol, '', x)
  }
  return(x)
}

# Remove specific symbols, such as quotation marks
specific_symbols <- c('“', '’','”','—')

load_federalist = function(type = 1, lemmatize = TRUE){

  # type 1: regular NLP preprocessing --contextual words onlyso
  # type 2: do not remove stopwords(type 1 + stop words)
  # type 3: MW chosen words
  
  data("federalistPapers", package='syllogi')
  
  docs <- federalistPapers
  
  docs = docs[-70] 
  
  docs<-lapply(docs,function(x){x=x$paper}) # this way we parse just the paper itself
  
  # Create a Corpus
  
  # corpus <- VCorpus(VectorSource(docs))
  corpus <- Corpus(VectorSource(docs))
  
  # Preprocess the text data
  
  # corpus <- tm_map(corpus, content_transformer(tolower)) 
  # corpus <- tm_map(corpus, removePunctuation)
  corpus <- tm_map(corpus, content_transformer(removeSpecificSymbols), symbols = specific_symbols)
  # corpus <- tm_map(corpus, removeNumbers)
  # corpus <- tm_map(corpus, stripWhitespace)
  
  if(lemmatize){
    corpus <- tm_map(corpus, lemmatize_strings)
    #courpus <- tm_map(corpus, stem_strings)
  }
  control_list <- list(
    # content_transformer(function(x) removeSpecificSymbols(x, specific_symbols)),
    # stemming = stem, 
    removePunctuation = TRUE,
    stopwords = (type < 2),
    removeNumbers = TRUE,
    tolower = TRUE,
    wordLengths = c(1, Inf)  # Adjust this if you want to retain very short words
  )
  
  tdm <- DocumentTermMatrix(corpus, control = control_list)
  
  if(type == 3){
    tdm = tdm[,tdm$dimnames$Terms %in% MW_words_list]
  }
  
  return(tdm)
}

# tdm1 = load_federalist(type = 1) # 8700
# tdm2 = load_federalist(type = 2) # 8822
# tdm3 = load_federalist(type = 3) # 173

load_federalist_authors = function(...){
  # setup author vector
  authors = rep(0,85)
  authors[c(1, 6:9, 11:13, 15:17, 21:36, 59:61, 65:85)] = "HAMILTON"
  authors[c(10, 14, 37:48)] = "MADISON"
  authors[c(18:20)] = "HAMILTON AND MADISON"
  authors[c(2:5, 64)] = "JAY"
  authors[c(49:58, 62:63)] = "UNKNOWN"
  select<-authors=="HAMILTON"|authors=="MADISON"
  authors_joint<-authors[authors=="HAMILTON AND MADISON"]
  authors_train<-authors[select]
  authors_train<-as.factor(authors_train)
  authors_train=as.numeric(authors_train=="MADISON")
  authors_train = as.integer(authors_train)
  return(list(authors = authors, select = select, authors_joint = authors_joint, authors_train = authors_train)
  )
}


run_bart_binary = function(x_train, x_test, y_train){
  bart = gbart(x_train, y_train, x_test, sparse=TRUE, type="lbart")
  predicted<-bart$prob.train.mean
  predicted_test<-bart$prob.test.mean
  
  return(list(predicted, predicted_test))
}

run_lasso_binary = function(x_train, x_test, y_train){
  fit3 <- gamlr(x_train,y_train,family="binomial") # based on minimum bic
  beta = as.matrix(coef(fit3))
  beta0 = beta[beta!=0,]
  predicted =  predict(fit3,newdata=x_train,type="response")[,1]
  predicted_test = predict(fit3,newdata=x_test,type="response")[,1]
  return(list(predicted, predicted_test,beta0[order(abs(beta0), decreasing = TRUE)]))
}

# library(BhGLM)
library(glmnet)
#https://github.com/nyiuab/BhGLM/tree/master
# run_sslasso_binary = function(x_train, x_test, y_train,verbose =TRUE, s = c(0.04, 0.5)){
#  fit3 <- bmlasso(x_train,y_train,family="binomial",alpha = 1, ss = s, maxit = 500)
#  fit3$offset <- FALSE
#  beta = as.matrix(coef(fit3))
#  beta0 = beta[beta!=0,]
#  predicted =  predict(fit3,newx=x_train,type="response")[,1]
#  predicted_test = predict(fit3,newx=x_test,type="response")[,1]
#  return(list(predicted, predicted_test,beta0[order(abs(beta0), decreasing = TRUE)]))
#}


plot_dist = function(res, y_train, save = NULL){
  # plot(res[[1]]~as.factor(y_train))
  if (!is.null(save)) {
    # Construct the file name with .png extension
    file_name <- paste0(save, ".pdf")
    pdf(file_name,pointsize=15)
  }
  d0<-density(res[[1]][y_train==0])
  plot(d0,main="Probabilities of Madison",xlim=c(0,1))
  polygon(d0,col="pink")
  d1<-density(res[[1]][y_train==1])
  points(d1,type="l")
  polygon(d1,col="lightblue")
  points(d0,type="l")
  legend("topright",legend=c("Madison","Hamilton","Disputed"),fill=c("lightblue","pink","green"),bty="n")
  abline(v=res[[2]],lty=2, col= "green")
  if (!is.null(save)) {dev.off()}
}




plot_dist_joint = function(res, res_joint, y_train){
  plot(res[[1]]~as.factor(y_train))
  # pdf("raw_probs.pdf",pointsize=15)
  d0<-density(res[[1]][y_train==0])
  plot(d0,main="Probabilities of Madison",xlim=c(0,1))
  polygon(d0,col="pink")
  d1<-density(res[[1]][y_train==1])
  points(d1,type="l")
  polygon(d1,col="lightblue")
  points(d0,type="l")
  legend("topright",legend=c("Madison","Hamilton","Disputed","Joint"),fill=c("lightblue","pink","green","orange"),bty="n")
  abline(v=res[[2]],lty=2, col= "green")
  abline(v=res_joint[[2]], lty = 2, col ="orange")
}


run_oos_cv = function(x_train, y_train){
  nrep = nrow(x_train)
  oos_prediction = matrix(0, 2, nrow(x_train))
  for(i in (1:nrep)){
    print(i)
    # Setup
    x_train_i<-x_train[-i,]
    x_test_i<-t(matrix(x_train[i,]))
    y_train_i<-y_train[-i]
    
    # LASSO words
    
    fit_cv <- cv.gamlr(x_train_i,y_train_i,family="binomial")
    oos_prediction[1,i]<-predict(fit_cv,newdata=x_test_i,type="response")
    
    # BART words
    
    bart = gbart(x_train_i, y_train_i, x.test=data.matrix(x_train), sparse=TRUE, type="lbart")
    oos_prediction[2,i]<-bart$prob.test.mean[i]  
  }
  return(oos_prediction) 
}


run_oos_cv_sslasso = function(x_train, y_train){
  nrep = nrow(x_train)
  oos_prediction = matrix(0, 1, nrow(x_train))
  for(i in (1:nrep)){
    print(i)
    # Setup
    x_train_i<-x_train[-i,]
    x_test_i<-t(matrix(x_train[i,]))
    y_train_i<-y_train[-i]
    
    # sslasso
    
    fit_cv <- bmlasso(x_train_i,y_train_i,family="binomial",alpha = 1, ss= c(0.05,1), maxit = 500)
    fit_cv$offset <- FALSE
    oos_prediction[1,i]<- predict(fit_cv,newx=x_test_i,type="response")
    
  }
  return(oos_prediction) 
}



# l2 error
l2error = function(oos_res, y_train){
  error<-apply(oos_res,1,function(x){(x-y_train)^2})
  return(apply(error,2,mean))
}

clferror = function(oos_res, c = 0.5, y_train){
  error<-apply(oos_res,1,function(x){(as.numeric(x)>c)!=y_train})
  return(apply(error,2,mean))
}

# Define the function to calculate classification error
clferror <- function(oos_res, y_train, c = NULL) {
  # Calculate optimal thresholds if c is not provided
  if (is.null(c)) {
    thresholds <- apply_opt_threshold(oos_res, y_train)
    optimal_threshold_roc <- thresholds$optimal_threshold_roc
    optimal_threshold_f1 <- thresholds$optimal_threshold_f1
    
    # Calculate errors using both thresholds
    error_roc <- apply(oos_res, 1, function(x) (as.numeric(x) > optimal_threshold_roc) != y_train)
    error_f1 <- apply(oos_res, 1, function(x) (as.numeric(x) > optimal_threshold_f1) != y_train)
    
    error_roc <- apply(error_roc, 2, mean)
    error_f1 <- apply(error_f1, 2, mean)
    
    return(list(error_roc = error_roc, error_f1 = error_f1))
  } else {
    # Calculate error using the given threshold c
    error <- apply(oos_res, 1, function(x) (as.numeric(x) > c) != y_train)
    return(apply(error, 2, mean))
  }
}

# Define the function to calculate classification error
clferror <- function(oos_res, y_train, c = NULL) {
  if (is.null(c)) {
    thresholds <- apply_opt_threshold(oos_res, y_train)
    optimal_thresholds_roc <- thresholds$optimal_thresholds_roc
    optimal_thresholds_f1 <- thresholds$optimal_thresholds_f1
    
    # Calculate errors using both thresholds
    error_roc <- apply(oos_res, 1, function(x, idx) (as.numeric(x) > optimal_thresholds_roc[idx]) != y_train, seq_along(optimal_thresholds_roc))
    error_f1 <- apply(oos_res, 1, function(x, idx) (as.numeric(x) > optimal_thresholds_f1[idx]) != y_train, seq_along(optimal_thresholds_f1))
    
    error_roc <- colMeans(error_roc)
    error_f1 <- colMeans(error_f1)
    
    return(list(error_roc = error_roc, error_f1 = error_f1))
  } else {
    # Calculate error using the given threshold c
    error <- apply(oos_res, 1, function(x) (as.numeric(x) > c) != y_train)
    return(colMeans(error))
  }
}

