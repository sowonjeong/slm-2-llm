# Small to Large Language Models: Revisiting the Federalist Papers

This repository contains the code used for all experiments in the paper: [From Small to Large Language Models: Revisiting the Federalist Papers](https://arxiv.org/abs/2503.01869)


## 📁 Project Structure

```
.
├── data/                  # Contains raw and processed Federalist Papers data
├── eda/                   # Notebooks and scripts for exploratory data analysis
├── embeddings/            # Scripts for generating and saving embeddings
├── postprocess/           # Postprocessing and analysis of embeddings
├── utils/                 # Shared utility functions
└── README.md              # Project documentation
```

---

## 1️⃣ Exploratory Data Analysis (EDA)

Located in the `eda/` directory, this step includes:

- Parsing and cleaning the Federalist Papers text
- Visualizing word frequency, document lengths, and authorship distribution
- Basic linguistic and stylistic features across authors (Hamilton, Madison, Jay)

Tools used: `pandas`, `matplotlib`, `seaborn`, `spaCy`

---

## 2️⃣ Embedding Generation

We generate embeddings using both transformer-based models and API-based methods.

### 🧠 Local Model Embeddings

Implemented models:
- **BERT** (`bert-base-uncased`)
- **RoBERTa** (`roberta-base`)
- **BART** (`facebook/bart-base`)
- **LLaMA** (via `transformers`, if locally supported or via Hugging Face inference endpoints)

Each paper is embedded at the sentence or document level by mean pooling the token embeddings.

### ☁️ GPT API Embeddings

We also generate embeddings using OpenAI's `text-embedding-ada-002` and other available GPT models via API.

Requires:
- OpenAI API key
- Adherence to rate limits and token constraints

---

## 3️⃣ Postprocessing Embeddings

Postprocessing steps (in `postprocess/` directory) include:

- Download the pretrained Word2Vec model [here](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?resourcekey=0-wjGZdNAUop6WykTtMip30g)
- Normalization and dimensionality reduction (e.g., PCA, t-SNE, UMAP)
- Clustering (e.g., KMeans, Agglomerative)
- Similarity heatmaps and distance-based analysis
- Authorship inference based on clustering structure or nearest-neighbor methods

---

## 🧪 Environment & Requirements

You’ll need:
- Python 3.8+
- OpenAI API key (if using GPT embedding)
- Hugging Face Token (if using open-source LLMs)
- Access to GPU for large-scale embedding generation (optional but recommended)
---

## 🤖 Credits

This README and were generated and refined with the help of [ChatGPT](https://openai.com/chatgpt).
