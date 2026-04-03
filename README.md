# NLP Neural N-gram Language Model

This project implements a neural 5-gram language model using PyTorch, based on the model proposed by Bengio et al. (2003).

It also includes a tool for inspecting learned word embeddings using cosine similarity.

---

## 📂 Files

- `neural_n_gram.py`  
  Trains a neural language model using a 4-word context to predict the next word.

- `wvv.py`  
  Word Vector Viewer – finds similar words using cosine similarity.

- `tokenizer.py`  
  Simple tokenizer for splitting text into words and punctuation.

---

## ⚙️ Requirements

- Python 3
- PyTorch

Install PyTorch:

```bash
pip install torch
