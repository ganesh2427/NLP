Sentiment Analysis with Naive Bayes Classifier
Overview
This project aims to classify text reviews as positive or negative using a Naive Bayes classifier. It involves using the TF-IDF vectorization technique for feature extraction and the Multinomial Naive Bayes model for classification. The data is preprocessed, split into training, validation, and test sets, and the performance of the model is evaluated on the test set using various metrics like accuracy, precision, recall, F1-score, and confusion matrix.

Files
rt-polarity.pos: Contains positive reviews.
rt-polarity.neg: Contains negative reviews.
These files are located at: /Users/shashack/Downloads/rt-polaritydata/rt-polaritydata/.

Requirements
To run the script, you will need the following Python libraries:

bash
Copy code
pip install numpy pandas scikit-learn matplotlib seaborn
Approach
Data Preparation
Loading the Data: The reviews are loaded from two files:
rt-polarity.pos: Positive reviews
rt-polarity.neg: Negative reviews These reviews are labeled with binary values (1 for positive and 0 for negative).
Data Splitting:
The dataset is divided into three sets:
Training Set: 4,000 positive and 4,000 negative reviews.
Validation Set: 500 positive and 500 negative reviews for model tuning.
Test Set: 831 positive and 831 negative reviews for final evaluation.
Text Preprocessing
TF-IDF Vectorization: The text is transformed into numerical features using TF-IDF vectorization, capturing both unigrams and bigrams.
The maximum number of features is set to 10,000.
Model
Naive Bayes Classifier (MultinomialNB): This classifier is effective for text data, especially with TF-IDF features.
Alpha Smoothing Parameter: Set to 0.1 to handle zero probabilities in feature space.
Pipeline Construction
The TF-IDF transformation and model training are combined into a pipeline, ensuring a streamlined process for training and evaluation.

Model Training and Evaluation
The model is trained on the training set and validated using the validation set. This helps tune the model and provides early insights into its performance.
Final performance is evaluated on the test set using various metrics.
Performance Metrics
Confusion Matrix: Used to capture true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN).
Precision: Measures the accuracy of positive predictions.
Recall: Measures the model’s ability to identify positive reviews.
F1-Score: Harmonic mean of precision and recall.
Accuracy: Overall accuracy of the model.
Script Breakdown
Loading and Labeling the Data
python
Copy code
# Loading positive and negative reviews and assigning labels (1 = positive, 0 = negative)
with open('/path_to/rt-polarity.pos', 'r', encoding='ISO-8859-1') as f:
    positive_reviews = f.readlines()

with open('/path_to/rt-polarity.neg', 'r', encoding='ISO-8859-1') as f:
    negative_reviews = f.readlines()

positive_labels = [1] * len(positive_reviews)
negative_labels = [0] * len(negative_reviews)
Dataset Splitting
python
Copy code
# Training Set (4000 positive + 4000 negative)
train_reviews = positive_reviews[:4000] + negative_reviews[:4000]
train_labels = [1] * 4000 + [0] * 4000

# Validation Set (500 positive + 500 negative)
val_reviews = positive_reviews[4000:4500] + negative_reviews[4000:4500]
val_labels = [1] * 500 + [0] * 500

# Test Set (831 positive + 831 negative)
test_reviews = positive_reviews[4500:5331] + negative_reviews[4500:5331]
test_labels = [1] * 831 + [0] * 831
Model Pipeline and Training
python
Copy code
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Create the pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
    ('nb', MultinomialNB(alpha=0.1))
])

# Train the model
pipeline.fit(train_reviews, train_labels)
Validation
python
Copy code
# Validate the model
val_preds = pipeline.predict(val_reviews)
val_accuracy = accuracy_score(val_labels, val_preds)
val_precision = precision_score(val_labels, val_preds)
val_recall = recall_score(val_labels, val_preds)
val_f1 = f1_score(val_labels, val_preds)

print(f"Validation Accuracy: {val_accuracy:.2f}%")
print(f"Validation Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1-Score: {val_f1:.4f}")
Testing and Evaluation
python
Copy code
# Test the model
test_preds = pipeline.predict(test_reviews)

# Calculate metrics
test_accuracy = accuracy_score(test_labels, test_preds) * 100
tn, fp, fn, tp = confusion_matrix(test_labels, test_preds).ravel()
precision = precision_score(test_labels, test_preds)
recall = recall_score(test_labels, test_preds)
f1 = f1_score(test_labels, test_preds)

# Output results
print(f"Test Accuracy: {test_accuracy:.2f}%")
print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
print(f"Test Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
Visualization
python
Copy code
import seaborn as sns
import matplotlib.pyplot as plt

# Visualize performance metrics
metrics = pd.DataFrame({
    'Metric': ['True Positives', 'True Negatives', 'False Positives', 'False Negatives', 'Precision', 'Recall', 'F1-Score'],
    'Value': [tp, tn, fp, fn, precision, recall, f1]
})

plt.figure(figsize=(8, 4))
sns.heatmap(metrics.set_index('Metric').T, annot=True, cmap='Blues', fmt='.4f')
plt.title(f"Model Performance Metrics (Test Accuracy: {test_accuracy:.2f}%)")
plt.show()
Results
The model's performance on the test set will be displayed, including key metrics such as accuracy, precision, recall, F1-score, and confusion matrix.

Conclusion
This project showcases a basic sentiment analysis pipeline, combining text vectorization and classification using Naive Bayes. The results highlight the model’s ability to classify text reviews effectively, and further improvements could be made by incorporating more advanced models or additional preprocessing techniques.
