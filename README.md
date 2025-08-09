# Climate Change Text Classification with Multi-Layer Perceptron (MLP)

## 📖 Project Overview

This project implements a **text classification system** that analyzes written content related to climate change and predicts its category (e.g., causes, impacts, solutions, or unrelated).
Using **Natural Language Processing (NLP)** techniques and a **Multi-Layer Perceptron (MLP)** neural network, the model learns to recognize patterns in language that are indicative of specific climate-related topics.

The work involved **data exploration, preprocessing, model development, training, and evaluation**, with a focus on building an effective classification pipeline using **TF-IDF features**.

---

## 🎯 Goal

The primary goal was to create an automated text analysis model that can:

* Identify if a given statement is related to climate change.
* Categorize climate-related statements into specific, meaningful classes.
* Serve as a proof-of-concept for text classification in environmental datasets.

---
📂 Dataset & Example Files
This repository includes two supplementary files that support the climate change text classification project:

climate_change.txt – A plain text dataset containing climate change–related statements and articles. Used for training and testing the text classification model after preprocessing with tokenization, stopword removal, and TF-IDF vectorization.

chat_history_sample.json – A small, anonymized sample of a conversation log demonstrating how a text classification or Q&A system might interact with climate-related inputs. Useful for illustrating model behavior and example outputs in a conversational setting.

Both files are provided for demonstration purposes and are safe to publish. If using larger or proprietary datasets, ensure licensing and privacy compliance before distribution.

---

## 🔍 Approach

### 1. Data Acquisition & Exploration

* Loaded climate change–related text data from structured datasets.
* Examined text distribution across categories to identify imbalance and trends.
* Reviewed sample entries to understand linguistic patterns in each category.

### 2. Preprocessing

* Converted all text to lowercase for consistency.
* Removed punctuation, special characters, and stopwords.
* Tokenized text into words and applied **TF-IDF vectorization** to create numerical feature representations.

### 3. Model Architecture

* Implemented a **Multi-Layer Perceptron** in PyTorch.
* Architecture included:

  * Input layer matching the TF-IDF feature size.
  * Fully connected hidden layers with ReLU activation.
  * Dropout layers to prevent overfitting.
  * Softmax output layer for multi-class classification.

### 4. Training & Optimization

* Used **cross-entropy loss** and the **Adam optimizer**.
* Monitored training and validation loss over multiple epochs.
* Applied early stopping based on validation performance trends.

### 5. Evaluation

* Generated classification reports including **accuracy, precision, recall, and F1-score**.
* Created a confusion matrix to visualize class-specific performance.
* Documented common misclassification patterns (e.g., overlap between “impacts” and “causes”).

---

## 📊 Key Results

* Achieved approximately **85% accuracy** on the validation dataset.
* Balanced performance across categories with precision and recall both above 80%.
* Validation metrics indicated a stable model without significant overfitting.

---

## 📝 Conclusion

The MLP combined with TF-IDF features provided a strong baseline for climate change text classification. The model demonstrated the feasibility of automated topic detection in environmental datasets with straightforward preprocessing and architecture.

### Potential Improvements:

* Integrating pre-trained embeddings such as **GloVe** or **Word2Vec** for richer semantic representation.
* Exploring **transformer-based models** (e.g., BERT) for improved contextual understanding.
* Expanding the dataset with more diverse sources to improve generalization.

---

## 🛠 Technology Stack

* **Programming Language:** Python 3.10+
* **Libraries:** PyTorch, NumPy, Pandas, scikit-learn, NLTK, Matplotlib
* **Feature Engineering:** TF-IDF vectorization
* **Evaluation Tools:** Classification reports, confusion matrix, metric visualizations
