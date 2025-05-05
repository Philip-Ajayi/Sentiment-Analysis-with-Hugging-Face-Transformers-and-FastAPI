# Sentiment Analysis using Deep Learning

This project implements sentiment analysis using deep learning. The model is trained on a dataset of movie reviews (IMDB dataset) to classify reviews as positive or negative based on the text content.

## Project Overview

The project uses an **LSTM-based model** for sentiment classification. The model takes a sequence of words (review text) as input and predicts whether the sentiment is positive or negative. We use the IMDB dataset to train the model.

## Requirements

1. Python 3.x
2. TensorFlow (>=2.0)
3. Keras
4. Numpy
5. Matplotlib (for visualizing results)

You can install the required packages using:

```bash
pip install -r requirements.txt
```

## Dataset

The **IMDB dataset** contains 50,000 movie reviews labeled as positive or negative. This dataset is available through Keras and can be loaded using:

```python
from tensorflow.keras.datasets import imdb
```

## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/sentiment-analysis.git
   cd sentiment-analysis
   ```

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## How to Train

1. Preprocess the data and tokenize the reviews:

   ```bash
   python preprocess.py
   ```

2. Train the model:

   ```bash
   python train.py
   ```

   This will start training the LSTM model. The model will be trained to classify the sentiment of movie reviews.

## How to Predict Sentiment

Once the model is trained, you can use it to predict the sentiment of a new review:

```bash
python predict_sentiment.py --review "This movie was amazing!"
```

The output will be either **positive** or **negative** based on the sentiment of the review.

## Example

```bash
python predict_sentiment.py --review "I hated this movie, it was terrible."
```

Output:

```bash
Sentiment: Negative
```
