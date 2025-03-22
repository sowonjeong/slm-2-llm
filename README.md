# Small to Large Language Models: Revisiting the Federalist Papers

This repository contains the code used for the experiments in the paper: [From Small to Large Language Models: Revisiting the Federalist Papers](https://arxiv.org/abs/2503.01869)


## 📁 Project Structure

```
.
├── data/                  # Contains raw and processed Federalist Papers data
├── eda/                   # Notebooks and scripts for exploratory data analysis
├── llm-embeddings/        # Scripts for generating and saving LLM embeddings
├── postprocess/           # Postprocessing and analysis of embeddings
├── utils/                 # Shared utility functions
└── README.md              # Project documentation
```

---

## 1️⃣ Exploratory Data Analysis (EDA)

Located in the `eda/` directory, this step includes:

- Parsing and cleaning the Federalist Papers text
- Visualizing word frequency, document lengths, and topic assignment

---

## 2️⃣ Embedding Generation

We generate embeddings using both small and large language model.

### 🧠 Large Model Embeddings

Tested models:
- **BERT** (`bert-base-uncased`)
- **RoBERTa** (`roberta-base`)
- **BART** (`facebook/bart-base`)
- **LLaMA** (via `transformers`, if locally supported or via Hugging Face inference endpoints)

You can adapt the following template to load and generate embeddings for any Hugging Face model.

👉 Example: [`example-llama.ipynb`](llm-embeddings/example-llama.ipynb)

```python
from transformers import  BertModel, BertTokenizer
from transformers import pipeline
import torch

# Load pre-trained BERT model and tokenizer
model = BertModel.from_pretrained("bert-base-uncased)
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

text = "To the People of the State of New York"

# Load the fine-tuned BERT model for feature extraction
bert_extractor = pipeline(
    task="feature-extraction",
    model=model,
    tokenizer=takenizer,
    device=0  # Use GPU if available
)
bert_extractor(text.astype(str).tolist(), return_tensors=True)
```

### ☁️ GPT API Embeddings

We also generate embeddings using OpenAI's `text-embedding-ada-002` and other available GPT models via API.

👉 Example: [`GPT-API-embedding.ipynb`](llm-embeddings/GPT-API-embedding.ipynb)

Requires:
- OpenAI API key
- Adherence to rate limits and token constraints

---

## 3️⃣ Postprocessing Embeddings

Postprocessing steps (in `postprocess/` directory) include:

Thanks for the clarification! Here's the corrected and polished version of the **Postprocessing** section for your `README.md`:

---

## 3️⃣ Postprocessing Embeddings

Located in the `postprocess/` directory, this stage includes:

1. **Word2Vec Embedding Generation**  
   - Code for generating Word2Vec-based embeddings from the Federalist Papers.  
   - 📥 Requires downloading the pretrained Google News Word2Vec model [here](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?resourcekey=0-wjGZdNAUop6WykTtMip30g).

2. **Classification Models (BART & LASSO)**  
   - Scripts to train and evaluate classifiers (e.g., BART, LASSO) on various embeddings:
     - Continuous LLM embeddings (e.g., from BERT, GPT)
     - Bag-of-Words (BoW) embeddings (e.g., from LDA, LSA or NMF)

3. **Benjamini-Hochberg (BH)** procedure for selecting words

---

## 🧪 Requirements

You’ll need:
- OpenAI API key (if using GPT embedding)
- Hugging Face Token (if using open-source LLMs)
- Access to GPU for large-scale embedding generation (optional but recommended)
---

## 🤖 Credits

This README and were generated and refined with the help of [ChatGPT](https://openai.com/chatgpt).
