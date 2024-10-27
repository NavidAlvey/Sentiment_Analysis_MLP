# Sentiment Analysis with PyTorch MLP

## Description
### Data Loading and Preprocessing

- Loads labeled sentiment data from three text files (`amazon_cells_labelled.txt`, `imdb_labelled.txt`, and `yelp_labelled.txt`) using `pandas`.
- Combines the datasets into a single dataframe, checks for missing values, duplicates, and label distribution.
- Preprocesses the text data by converting it to lowercase, removing punctuation and digits, and eliminating stop words using NLTK’s English stopwords list.

### Text Vectorization

- Builds a vocabulary of the 1000 most common words across all tokens in the dataset.
- Defines a function to vectorize each sentence based on the vocabulary, where each word is represented as an index in the vector if it’s in the vocabulary.

### Dataset Splitting:

- Splits the dataset into training (80%) and testing (20%) sets, shuffling the indices to randomize the data.

### Model Definition (MLP Architecture):

- Defines an MLP model class SentimentMLP, consisting of an input layer, two hidden layers with ReLU activations and dropout for regularization, and an output layer with a Sigmoid activation for binary classification.

### Training Loop:

- Converts training and testing data into `PyTorch tensors`.
- Defines a Binary Cross Entropy Loss function and an Adam optimizer with a learning rate of 0.01.
- Runs a training loop over 20 epochs, printing the loss after each epoch.

### Model Evaluation:

- Switches to evaluation mode and makes predictions on the test set.
- Applies a threshold of 0.5 to generate binary predictions.
- Computes the model's accuracy by comparing predicted and actual labels.

## Project Structure

- `p1.py`: Main script containing data loading, preprocessing, model definition, training, and evaluation functions.
- `amazon_cells_labelled.txt`, `imdb_labelled.txt`, `yelp_labelled.txt`: Datasets containing sentences and their sentiment labels (positive/negative).

## Requirements

The script requires the following Python packages:
- `pandas`
- `numpy`
- `nltk`
- `torch`

Install the required packages with:
```bash
pip install pandas numpy nltk torch
