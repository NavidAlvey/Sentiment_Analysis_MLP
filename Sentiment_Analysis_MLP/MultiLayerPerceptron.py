# Sentiment Analysis using PyTorch MLP
# ------------------------------------------------------

# Import necessary libraries
import pandas as pd
import numpy as np
import os
import re
import nltk
import torch
import torch.nn as nn
import torch.optim as optim
from nltk.corpus import stopwords
from collections import Counter

# Ensure reproducibility
import random
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Download NLTK data (if not already downloaded)
nltk.download('stopwords')

# ------------------------------------------------------
# Part 1: Data Loading and Preprocessing
# ------------------------------------------------------

# Step 1: Load the Dataset
# Instructions:
# - Use `pandas.read_csv` to load the three text files: 'amazon_cells_labelled.txt', 'imdb_labelled.txt', 'yelp_labelled.txt'.
# - Each file contains sentences and labels separated by a tab.
# - Assign column names 'sentence' and 'label' using the `names` parameter.

dir_path = os.path.dirname(os.path.realpath(__file__))
try:
    amazon_df = pd.read_csv(os.path.join(dir_path, 'amazon_cells_labelled.txt'), delimiter='\t', header=None, names=['sentence', 'label'])
    imdb_df = pd.read_csv(os.path.join(dir_path, 'imdb_labelled.txt'), delimiter='\t', header=None, names=['sentence', 'label'])
    yelp_df = pd.read_csv(os.path.join(dir_path, 'yelp_labelled.txt'), delimiter='\t', header=None, names=['sentence', 'label'])

    pass
except FileNotFoundError:
    print("Dataset files not found. Please ensure the dataset files are in the current directory.")
    exit()

# Step 2: Data Analysis
# Instructions:
# - Use `pandas.concat` to combine the three dataframes into one.
# - Display the first 5 rows using `DataFrame.head()`.
# - Check for missing values using `DataFrame.isnull().sum()`.
# - Check for duplicates using `DataFrame.duplicated().sum()`.
# - Print the distribution of the labels using `DataFrame['label'].value_counts()`.

# Combine the datasets using pd.concat and assign it to variable df
#################
df = pd.concat([amazon_df, imdb_df, yelp_df], ignore_index=True)
#################


# Display the first few rows using df.head()
print("First 5 rows of the dataset:")

#################
print(df.head())
#################

# Check for missing values using df.isnull().sum()
print("\nMissing values in each column:")

#################
print(df.isnull().sum())
#################

# Check for duplicates using df.duplicated().sum()
duplicates = df.duplicated().sum()
print("\nNumber of duplicate rows:", duplicates)
#################
# Your code here #
#################

# Visualize label distribution using df['label'].value_counts()
print("\nLabel distribution:")

#################
print(df['label'].value_counts())
#################

# Step 3: Data Cleaning and Preprocessing
# Instructions:
# - Use `nltk.corpus.stopwords.words('english')` to initialize stop words.
# - Define a function `preprocess_text` that:
#   - Converts text to lowercase using `str.lower()`.
#   - Removes punctuation and digits using `re.sub()`.
#   - Splits text into words (tokens) using `str.split()`.
#   - Removes stop words by checking if each word is not in the stop words set.
# - Apply the preprocessing function to the 'sentence' column using `DataFrame.apply()` and store the tokens in a new column 'tokens'.

# Initialize stop words
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation and special characters using re.sub()
    text = re.sub(r'[^\w\s]', '', text)
    # Remove digits using re.sub()
    text = re.sub(r'\d+', '', text)
    # Split text into words (tokens) using str.split()
    words = text.split()
    # Remove stop words
    words = [word for word in words if word not in stop_words]
    return words

# Apply preprocessing to the 'sentence' column using df['sentence'].apply()
#################
df['tokens'] = df['sentence'].apply(preprocess_text)
#################

# Step 4: Build Vocabulary and Vectorize Text
# Instructions:
# - Use `collections.Counter` to build a vocabulary of the most common 1000 words from the tokens.
#   - Flatten the list of tokens using a list comprehension.
#   - Use `Counter.most_common()` to get the most common words.
# - Create a word-to-index mapping `word_to_idx` using a dictionary comprehension.
# - Define a function `vectorize_tokens` that:
#   - Initializes a zero vector of size equal to the vocabulary using `np.zeros()`.
#   - For each token in the tokens list:
#     - If the token is in `word_to_idx`, get its index and increment the corresponding position in the vector.
# - Apply the vectorization function to the 'tokens' column using `DataFrame.apply()` and store the vectors in a new column 'vector'.

# Build vocabulary from the most common words
all_tokens = [token for tokens in df['tokens'] for token in tokens]
vocab_size = 1000
most_common_tokens = Counter(all_tokens).most_common(vocab_size)
word_to_idx = {word: idx for idx, (word, _) in enumerate(most_common_tokens)}

def vectorize_tokens(tokens, word_to_idx):
    # Initialize zero vector
    vector = np.zeros(len(word_to_idx), dtype=np.float32)
    # For each token, if it is in word_to_idx, increment the corresponding index
    #################
    for token in tokens:
        if token in word_to_idx:
            vector[word_to_idx[token]] += 1  
    #################
    return vector

# Vectorize all sentences
df['vector'] = df['tokens'].apply(lambda x: vectorize_tokens(x, word_to_idx))

# Step 5: Split the Dataset Manually
# Instructions:
# - Use `np.arange()` to create an array of indices equal to the length of the dataset.
# - Use `np.random.shuffle()` to shuffle the indices.
# - Calculate the split index based on an 80-20 split.
# - Use array slicing to split `X` and `y` into training and testing sets.

# Prepare feature and label arrays
X = np.stack(df['vector'].values)
y = df['label'].values

# Shuffle indices
indices = np.arange(len(df))
np.random.shuffle(indices)

# Calculate split index
split_ratio = 0.8
split_index = int(len(df) * split_ratio)

# Split data
train_indices = indices[:split_index]
test_indices = indices[split_index:]

X_train = X[train_indices]
X_test = X[test_indices]
y_train = y[train_indices]
y_test = y[test_indices]

# ------------------------------------------------------
# Part 2: Model Definition
# ------------------------------------------------------

# Instructions:
# - Define a class `SentimentMLP` that inherits from `nn.Module`.
# - In the `__init__` method:
#   - Initialize the parent class using `super().__init__()`.
#   - Define the layers of the MLP using `nn.Sequential()`:
#     - Input layer: `nn.Linear(input_size, 64)`
#     - Activation: `nn.ReLU()`
#     - Dropout: `nn.Dropout(p=0.5)`
#     - Hidden layer: `nn.Linear(64, 32)`
#     - Activation: `nn.ReLU()`
#     - Dropout: `nn.Dropout(p=0.5)`
#     - Output layer: `nn.Linear(32, 1)`
#     - Activation: `nn.Sigmoid()`
# - In the `forward` method:
#   - Pass the input `x` through the model using `self.model(x)`.

import torch.nn as nn

class SentimentMLP(nn.Module):
    #################
    def __init__(self, input_size):
        super(SentimentMLP, self).__init__()
        
        # Define the layers of the MLP using nn.Sequential
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),    
            nn.ReLU(),                  
            nn.Dropout(p=0.5),             
            nn.Linear(64, 32),            
            nn.ReLU(),                   
            nn.Dropout(p=0.5),            
            nn.Linear(32, 1),              
            nn.Sigmoid()                   
        )
    #################
    pass
    def forward(self, x):
        return self.model(x)


# Instantiate the model
# Instructions:
# - Determine the input size using `X_train.shape[1]`.
# - Create an instance of `SentimentMLP` with the input size.

input_size = X_train.shape[1]
model = SentimentMLP(input_size)

# ------------------------------------------------------
# Part 3: Training Loop
# ------------------------------------------------------

# Instructions:
# - Convert the training and testing data into PyTorch tensors using `torch.from_numpy()`.
#   - Ensure that the labels (`y_train`, `y_test`) are converted to `float` tensors and reshaped to have shape `(n_samples, 1)` using `.view(-1, 1)`.
# - Define the loss function using `nn.BCELoss()`.
# - Define the optimizer using `optim.Adam()` with a learning rate of `0.01`.
# - Implement the training loop:
#   - Set the number of epochs to `20`.
#   - For each epoch:
#     - Set the model to training mode using `model.train()`.
#     - Zero the gradients using `optimizer.zero_grad()`.
#     - Perform the forward pass to get outputs.
#     - Compute the loss using `criterion(outputs, y_train_tensor)`.
#     - Perform the backward pass using `loss.backward()`.
#     - Update the model parameters using `optimizer.step()`.
#     - Print the loss at each epoch.

# Convert data to PyTorch tensors
X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).float().view(-1, 1)
X_test_tensor = torch.from_numpy(X_test).float()
y_test_tensor = torch.from_numpy(y_test).float().view(-1, 1)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
epochs = 20
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    # Forward pass
    #################
    outputs = model(X_train_tensor)
    #################

    # Compute loss
    #################
    loss = criterion(outputs, y_train_tensor)
    #################

    # Backward pass and optimization
    #################
    loss.backward()
    optimizer.step()    
    #################
    loss = criterion(outputs, y_train_tensor)

    # Print loss every epoch
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# ------------------------------------------------------
# Part 4: Evaluation
# ------------------------------------------------------

# Instructions:
# - Set the model to evaluation mode using `model.eval()`.
# - Perform the forward pass on the test data without tracking gradients using `torch.no_grad()`.
# - Apply a threshold of `0.5` to the model outputs to get binary predictions.
# - Implement a function `calculate_metrics` to compute accuracy:
#   - Compare the predicted labels with the true labels using element-wise comparison.
#   - Calculate the number of correct predictions.
#   - Compute the accuracy as the ratio of correct predictions to total predictions.
# - Print the accuracy.

# Evaluate the model on the test set
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    predicted = (outputs > 0.5).float()

# Convert tensors to numpy arrays for metric calculations
y_test_np = y_test_tensor.numpy()
predicted_np = predicted.numpy()

def calculate_metrics(y_true, y_pred):
    #################
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    correct_predictions = (y_true == y_pred).sum()
    accuracy = correct_predictions / len(y_true)
    #################
    return accuracy

# Calculate metrics
accuracy = calculate_metrics(y_test_np, predicted_np)

print(f'\nEvaluation Metrics:')
print(f'Accuracy  : {accuracy * 100:.2f}%')