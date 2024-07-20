# Word2Vec with PyTorch

This repository contains a Word2Vec implementation using PyTorch. The project includes the necessary modules to train a Word2Vec model on a given text corpus, evaluate the model, and save the training logs. It also includes early stopping functionality to prevent overfitting.

## Features

- **Custom Dataset Handling**: Easily read and preprocess text data.
- **Training Loop with Early Stopping**: Implemented to train the model and stop early if validation loss doesn't improve.
- **Negative Sampling**: Used for efficient training.
- **Cosine Similarity**: To find similar words in the embedding space.

## Installation

Clone the repository:
 ```sh
 git clone https://github.com/Vadimbuildercxx/word2vec-pytorch.git
 cd word2vec-pytorch
```

## Usage

### Training the Model

To train the model, run the following command:

```sh
python train.py --epochs 5 --lr 0.0001 --emb_dim 100 --batch_size 256 --dataset_path data/wikitext-103/wiki.train.tokens --min_words_count 1 --window_size 5 --early_stopper_patience 2 --early_stopper_min_delta 0.15 --train_val_split_coeff 0.9
```

#### Parameters
- `--epochs`: Number of epochs to train.
- `--lr`: Learning rate for the optimizer.
- `--emb_dim`: Dimensionality of word embeddings.
- `--batch_size`: Size of each batch during training.
- `--dataset_path`: Path to the dataset.
- `--min_words_count`: Minimum count of words to be considered.
- `--window_size`: Context window size for Word2Vec.
- `--early_stopper_patience`: Patience for early stopping.
- `--early_stopper_min_delta`: Minimum change in validation loss to qualify as an improvement.
- `--train_val_split_coeff`: Train-validation split coefficient.

```commandline
python train.py --epochs 5 --lr 0.001 --emb_dim 100 --batch_size 128 --dataset_path data/sample_corpus.txt --min_words_count 1 --window_size 5 --early_stopper_patience 1 --early_stopper_min_delta 0.01 --train_val_split_coeff 0.9
```

#### Example of usage

```python
import torch
from model import Word2Vec
from utils import EarlyStopper
from data import Reader, Word2VecDataset

checkpoint = torch.load('w2v_model.pt')
model = Word2Vec(num_tokens=len(checkpoint['id2label']), embedding_dim=100)
model.load_state_dict(checkpoint['model_state_dict'])
w2v = Word2VecModel(model, checkpoint['label2id'], checkpoint['id2label'])
```

Get k similar vectors:
```python
similar_words = w2v["example"], k=5)
```

Get vector by key or id:
```python
example_vector_a = w2v["example"]
example_vector_b = w2v[6012]
```

Operations with vectors:
```python
example_vector = w2v.get_k_nearest_word(w2v["king"] - w2v["man"] + w2v["woman"], 1)
```

Compare two words:
```python
example_vector = w2v.compare_vectors("negative", "positive")
```

## Acknowledgements
Inspired by the Word2Vec paper by Mikolov et al.