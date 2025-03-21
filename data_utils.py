import json
import pandas as pd
import nltk
from pathlib import Path
from nltk.tokenize import sent_tokenize, word_tokenize
import rpy2.robjects as ro
import rpy2.robjects.packages as rpackages
from rpy2.robjects import pandas2ri


nltk.download('punkt')

def load_federalist_py():
    with open("data/federalist.json") as f:
        federalist_dict = json.load(f)


    # Initialize an empty list to store the flattened data
    flattened_data = []

    # Iterate over each dictionary in the list
    for item in federalist_dict:
        # Extract metadata and content
        meta = item['meta'][0]  # Assuming there's only one meta per item
        paper = item['paper'][0] if item['paper'] else ''  # Assuming content is a list

        # Flatten the data
        flattened_data.append({
            'meta_number': meta.get('number', ''),
            'meta_author': meta.get('author', ''),
            'meta_title': meta.get('title', ''),
            'meta_journal': meta.get('journal', ''),
            'meta_body': paper
        })

    # Create a DataFrame
    df = pd.DataFrame(flattened_data)
    federalist_df = df.drop(df.index[69])

    federalist_df['meta_sentence'] = federalist_df['meta_body'].apply(sent_tokenize)
    federalist_df['meta_words'] = federalist_df['meta_sentence'].apply(lambda sentences: [word_tokenize(sentence) for sentence in sentences])

    
    # Create a list to store sentence-level data
    sentence_data = []

    # Iterate over each row in the original DataFrame
    for index, row in federalist_df.iterrows():
        for i, sentence in enumerate(row['meta_sentence']):
            # Count the number of words in the sentence
            word_count = len(word_tokenize(sentence))

            # Append the data to the sentence_data list
            sentence_data.append({'meta_number': row['meta_number'],
                                'meta_author': row['meta_author'],
                                'meta_sentence_number': i + 1,
                                'meta_sentence': sentence,
                                'meta_num_of_words': word_count})

    # Convert the list of dictionaries to a DataFrame
    sentence_df = pd.DataFrame(sentence_data)

    return federalist_df, sentence_df


def load_rdata(file_path):
    pandas2ri.activate()

    # Load the R data file
    ro.r['load'](file_path)

    # Get the list of objects in the R environment
    r_objects = ro.r('ls()')

    # Convert R objects to Python objects
    data = {}
    for obj in r_objects:
        data[obj] = ro.r[obj]

    return data

def read_lines(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]



def load_MADStat():
    madstat_raw = load_rdata("~/txt-analysis/MADStat/MADStaText/4-Raw data/Raw-data-2019-12-version.RData")
    # paper has an abstract here
    # paper_raw = paper
    madstat_clean = load_rdata("~/txt-analysis/MADStat/MADStaText/1-Text abstracts/TextCorpusFinal.RData")
    # CleanAbstracts
    # TDM
    madstat_aupap = load_rdata("~/txt-analysis/MADStat/AuthorPaperInfo.RData")
    # AuPapMat
    # PapPapMat
    madstat_bib = load_rdata("~/txt-analysis/MADStat/BibtexInfo.RData")
    # journal
    # paper
    # paper_author

    madstat_author = read_lines("MADStat/author_name.txt")
    return madstat_raw, madstat_clean, madstat_aupap, madstat_bib, madstat_author
                              