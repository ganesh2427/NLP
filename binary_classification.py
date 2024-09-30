import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Loading the data from the source
with open('/Users/shashack/Downloads/rt-polaritydata/rt-polaritydata/rt-polarity.pos', 'r', encoding='ISO-8859-1') as f:
    positive_reviews = f.readlines()

with open('/Users/shashack/Downloads/rt-polaritydata/rt-polaritydata/rt-polarity.neg', 'r', encoding='ISO-8859-1') as f:
    negative_reviews = f.readlines()

# Labeling the dataset: Assigning binary labels to the reviews,
# where positive reviews are labeled as 1 and negative reviews as 0.
positive_labels = [1] * len(positive_reviews)
negative_labels = [0] * len(negative_reviews)

# Combine the data: Merging positive and negative reviews into a single dataset
# along with their corresponding labels for further processing.

reviews = positive_reviews + negative_reviews
labels = positive_labels + negative_labels

# Ensure dataset size consistency: Verifying that the number of positive and negative reviews 
# matches the specified sizes for training, validation, and test sets.

assert len(positive_reviews) == len(positive_labels)
assert len(negative_reviews) == len(negative_labels)

# Create training, validation, and test sets based on the specified sizes: 
# Splitting the dataset into three distinct sets:
# - Training set: First 4000 positive and 4000 negative reviews for model training.
# - Validation set: Next 500 positive and 500 negative reviews for hyperparameter tuning and model evaluation.
# - Test set: Final 831 positive and 831 negative reviews for assessing the model's performance.

train_reviews = positive_reviews[:4000] + negative_reviews[:4000]
train_labels = [1] * 4000 + [0] * 4000

val_reviews = positive_reviews[4000:4500] + negative_reviews[4000:4500]
val_labels = [1] * 500 + [0] * 500

test_reviews = positive_reviews[4500:5331] + negative_reviews[4500:5331]
test_labels = [1] * 831 + [0] * 831

# Create a pipeline for TF-IDF transformation and Naive Bayes classifier:
# This combines TfidfVectorizer for feature extraction (unigrams and bigrams) 
# and MultinomialNB for classification into a single streamlined process.

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),  # Unigram and bigram features
    ('nb', MultinomialNB(alpha=0.1))  # Naive Bayes classifier
])

# Train the model on the training set
pipeline.fit(train_reviews, train_labels)

# Validate the model on the validation set (optional, but useful)
val_preds = pipeline.predict(val_reviews)
val_accuracy = accuracy_score(val_labels, val_preds) * 100
val_precision = precision_score(val_labels, val_preds)
val_recall = recall_score(val_labels, val_preds)
val_f1 = f1_score(val_labels, val_preds)

print(f"Validation Accuracy: {val_accuracy:.2f}%")
print(f"Validation Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1-Score: {val_f1:.4f}")

# Test the model on the test set
test_preds = pipeline.predict(test_reviews)

# Evaluate the model on the test set
test_accuracy = accuracy_score(test_labels, test_preds) * 100
tn, fp, fn, tp = confusion_matrix(test_labels, test_preds).ravel()
precision = precision_score(test_labels, test_preds)
recall = recall_score(test_labels, test_preds)
f1 = f1_score(test_labels, test_preds)

# Output results
print(f"Test Accuracy: {test_accuracy:.2f}%")
print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
print(f"Test Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

# Create a table to present results
metrics = pd.DataFrame({
    'Metric': ['True Positives', 'True Negatives', 'False Positives', 'False Negatives', 'Precision', 'Recall', 'F1-Score'],
    'Value': [tp, tn, fp, fn, precision, recall, f1]
})

# Plot the table
plt.figure(figsize=(8, 4))
sns.heatmap(metrics.set_index('Metric').T, annot=True, cmap='Blues', fmt='.4f')
plt.title(f"Model Performance Metrics (Test Accuracy: {test_accuracy:.2f}%)")
plt.show()