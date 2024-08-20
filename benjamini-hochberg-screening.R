source("/Users/sowonjeong/txt-analysis/data/utils.R") # NEED to specify the path
HC = read.csv("/Users/sowonjeong/Documents/GitHub/HCAuthorship/Federalists/out_threhold_vocab.csv")

####################### Benjamini-Hochberg #######################
authors = load_federalist_authors()
tdm2 = load_federalist(type = 2) # 8723 # 5996


H_freq = colSums(as.matrix(tdm2)[authors=="HAMILTON",])
M_freq = colSums(as.matrix(tdm2)[authors=="MADISON",])
# check if there is 0 occurence
zero_ind = which((H_freq+M_freq)==0)

# binomial test as in Kipnis (2022)
two_sample_binom_test = function(sample1, sample2, alt = "two.sided"){
  N = length(sample1)
  pvals = rep(0, N)
  for (i in (1:N)){
    n = sample1[i] + sample2[i]
    p = sum(sample1[-i])/(sum(sample1[-i])+sum(sample2[-i]))
    pvals[i] = binom.test(sample1[i], n, p, alternative=alt)$p.value
  }
  names(pvals) = names(sample1)
  return(pvals)
}

pvals_two = two_sample_binom_test(H_freq[-zero_ind], M_freq[-zero_ind], alt = "two.sided")
pvals_geq = two_sample_binom_test(H_freq[-zero_ind], M_freq[-zero_ind], alt = "greater")
pvals_leq = two_sample_binom_test(H_freq[-zero_ind], M_freq[-zero_ind], alt = "less")


## extract p-value cutoff for E[fdf] < q
fdr_cut <- function(pvals, q, plotit=FALSE, ...){
  pvals <- pvals[!is.na(pvals)]
  N <- length(pvals)
  
  k <- rank(pvals, ties.method="min")
  alpha <- max(pvals[ pvals<= (q*k/N) ])
  
  if(plotit){
    sig <- factor(pvals<=alpha)
    o <- order(pvals)
    plot(pvals[o], col=c("grey60","red")[sig[o]], pch=20, ..., 
         ylab="p-values", xlab="tests ordered by p-value", main = paste('FDR =',q))
    lines(1:N, q*(1:N)/N)
  }
  
  return(alpha)
}

# Hamilton & Madison have different usage of the words
fdr_cut(pvals_two, 0.1, plotit = TRUE) # 0.002117374
fdr_cut(pvals_two, 0.05, plotit = TRUE) # 0.002117374

# Hamilton's document is more likely to include more of this words
fdr_cut(pvals_geq, 0.1, plotit = TRUE) # 0.0004581803

# Hamilton's document is more likely to include less of this words
fdr_cut(pvals_leq, 0.1, plotit = TRUE) # 0.0009466518

sort(pvals_two[pvals_two <= 0.002117374]) # 117 words selected with FDR <= 0.1
sort(pvals_two[pvals_two <= 0.0006524924]) # 78 words selected with FDR <= 0.05
sort(pvals_geq[pvals_geq <= 0.0004581803]) # include "upon","while"...
sort(pvals_leq[pvals_leq <= 0.0009466518]) # include "on","whilst"

sort(pvals_two[MW_final])
MW_final[!(MW_final %in% names(sort(pvals_two[pvals_two <= 0.002117374])))]
# Bonferroni?
sort(pvals_two[pvals_two <= 0.1/length(pvals_two)])
sort(pvals_two[pvals_two <= 0.05/length(pvals_two)])


sort(pvals_two[pvals_two <= 0.002117374]) 

bh =data.frame(name = names(pvals_two[pvals_two <= 0.002117374]), 
           pval = pvals_two[pvals_two <= 0.002117374])%>%
  mutate(size = -log10(pval))

bonferroni = data.frame(name = names(pvals_two[pvals_two <= 0.1/length(pvals_two)]),
                        pval = pvals_two[pvals_two <= 0.1/length(pvals_two)])%>%
  mutate(size = -log10(pval))



wordcloud(words = HC$feature, freq = -log10(HC$pval), min.freq = 1,
          max.words = 200, random.order = FALSE, rot.per = 0.35, 
          colors = brewer.pal(8, "Dark2"))

wordcloud(words = bh$name, freq = bh$size, min.freq = 1,
          max.words = 200, random.order = FALSE, rot.per = 0.35, 
          colors = brewer.pal(8, "Dark2"))


wordcloud(words = bonferroni$name, freq = bonferroni$size, min.freq = 1,
          max.words = 200, random.order = FALSE, rot.per = 0.35, 
          colors = brewer.pal(8, "Dark2"))

dim(bonferroni)
dim(bh)
