library(tidyverse)
library(tidytext)
library(gutenbergr)

# id = 18 : with table of contents
# federalist_papers <- gutenberg_download(18) 
federalist_papers <- gutenberg_download(1404)
head(federalist_papers, n = 10)

## List format by each paper (86 lists)
## Two versions of No. 70
data("federalistPapers", package='syllogi')
str(federalistPapers)
library(jsonlite)
write_json(federalistPapers, "federalist.json")