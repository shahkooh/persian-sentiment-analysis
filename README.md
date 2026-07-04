# Persian Sentiment Analysis using Bidirectional LSTM

An end-to-end Natural Language Processing (NLP) project for binary sentiment classification of Persian text using Deep Learning. This project analyzes Persian product reviews collected from Digikala and classifies them into **Positive** or **Negative** sentiments using a Bidirectional LSTM neural network.

---

## Overview

Sentiment Analysis is one of the most important tasks in Natural Language Processing. This project presents a complete pipeline for Persian sentiment classification, including data collection, preprocessing, word embedding, model training, evaluation, and prediction.

The dataset was collected from Digikala using a custom web crawler and contains more than **23,000 Persian product reviews** labeled according to user ratings.

---

## Features

* Persian text preprocessing
* Custom Digikala review dataset
* Data cleaning and normalization
* Stop-word removal
* Persian stemming using Hazm
* Tokenization and sequence padding
* Word2Vec embedding support
* Bidirectional LSTM architecture
* Model evaluation using multiple metrics
* Confusion Matrix visualization
* Real-time sentiment prediction

---

## Dataset

The dataset contains approximately **23,000** Persian product reviews collected from Digikala.

Each sample includes:

* Review text
* Sentiment label (Positive / Negative)

The dataset was automatically labeled based on users' product ratings. Since the labels were generated from rating scores, a small amount of label noise is expected.

Dataset Split:

* Training Set: 80%
* Test Set: 20%

---

## Text Preprocessing

The preprocessing pipeline includes:

* Persian character normalization
* Removing English letters and numbers
* Removing punctuation
* Tokenization using Hazm
* Stop-word removal
* Optional stemming
* Sequence padding

---

## Word Representation

Two embedding approaches were evaluated:

* Keras Embedding Layer (trainable)
* Word2Vec using Gensim

Experiments were conducted with embedding dimensions of **32**, **100**, and **300**. The best performance was achieved with **100-dimensional embeddings**.

---

## Model Architecture

The neural network consists of:

1. Embedding Layer
2. Bidirectional LSTM (32 units)
3. Global Max Pooling
4. Dense Layer (20 neurons, ReLU)
5. Dropout Layer
6. Output Layer (Softmax)

Architecture Summary:

Embedding → BiLSTM → GlobalMaxPooling → Dense → Dropout → Softmax

---

## Technologies

* Python
* TensorFlow / Keras
* Gensim
* Hazm
* NumPy
* Pandas
* Scikit-learn
* Matplotlib

---

## Results

| Dataset      | Accuracy |
| ------------ | -------: |
| Training Set |      98% |
| Test Set     |   89–90% |

The model achieved approximately **90% test accuracy** on unseen Persian product reviews.

Evaluation metrics include:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

---

## Project Structure

```text
persian-sentiment-analysis/
│
├── notebook/
│   └── Persian_Sentiment_Analysis.ipynb
│
├── dataset/
│   └── comment.csv
│
├── images/
│
├── requirements.txt
│
└── README.md
```

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/persian-sentiment-analysis.git

cd persian-sentiment-analysis

pip install -r requirements.txt
```

---

## Usage

Open the notebook and execute the cells sequentially.

The project supports:

* Dataset preprocessing
* Model training
* Model evaluation
* Sentiment prediction for custom Persian text

Example:

```python
text = "این گوشی فوق‌العاده است"

prediction = predict(text)
```

Output:

```
Positive
```

---

## Future Improvements

* Fine-tuning ParsBERT
* Transformer-based sentiment classification
* Multi-class sentiment analysis
* Attention mechanisms
* Hyperparameter optimization
* Deployment using FastAPI

---

## Acknowledgements

This project uses the following open-source libraries:

* Hazm
* TensorFlow
* Gensim
* Scikit-learn
* Pandas
* NumPy
* Matplotlib

---

## License

This project is released under the MIT License.
