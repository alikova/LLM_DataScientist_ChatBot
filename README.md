# LLM_DataScientist_ChatBot
Analyze and classify messages from a dataset, then develop a chatbot that can respond to natural language queries with meaningful insights.

User Feedback Analysis and Chatbot System
This repository contains a comprehensive system for analyzing user feedback messages from LiveChat and Telegram, classifying them into actionable categories, and providing a natural language chatbot interface for querying the data.

Project Overview
The system processes user feedback messages to:

Clean and preprocess text data
Generate embeddings using BERT
Classify messages into actionable categories
Provide a conversational interface for analyzing the data

Features
Message Classification: Automatically categorizes user messages into predefined categories
Dynamic Filtering: Supports filtering by category, time range, and source
Statistical Insights: Provides metrics like message counts, unique users, and trend detection
Conversational Memory: Maintains context across multiple queries

Getting Started
Prerequisites
Python 3.7+
Google Colab (for easy execution)
Access to Google Drive (optional, for storing large datasets)
Installation

Clone this repository:
git clone https://github.com/alikova/LLM_DataScientist_ChatBot.git

Open the notebooks in Google Colab:
Navigate to Google Colab
File → Open notebook → GitHub
Enter your repository URL
Follow the step-by-step instructions in each notebook

Project Structure
├── notebooks/
│   ├── 1_Data_Preprocessing.ipynb   # Data cleaning and preparation
│   ├── 2_Model_Training.ipynb       # BERT embeddings and classification
│   └── 3_Chatbot_Interface.ipynb    # Interactive chatbot demo
├── src/
│   ├── preprocessor.py              # Data preprocessing functions
│   ├── embeddings.py                # BERT embedding functions
│   ├── classifier.py                # Classification model
│   └── chatbot.py                   # Chatbot implementation
└── README.md                        # This file

Usage
The project is organized into three main notebooks:

1. Data Preprocessing
Loads and cleans the raw data
Handles missing values
Tokenizes messages
Performs exploratory data analysis

2. Model Training
Generates BERT embeddings for messages
Performs clustering to identify natural categories
Trains a classification model
Evaluates model performance

3. Chatbot Interface
Implements a natural language interface
Supports querying the dataset
Maintains conversational context
Provides statistical insights and visualizations
Evaluation Questions
How did you classify feedback?
The classification system uses a hybrid approach:

Unsupervised clustering: BERT embeddings + K-means to identify natural groupings
Supervised classification: Random Forest classifier trained on labeled examples
Rule-based refinement: Domain-specific rules to handle edge cases
This approach was chosen because it:

Leverages semantic understanding from BERT
Doesn't require large amounts of labeled data
Can be continuously improved with more data
For previously unseen issues (e.g., a new wallet blocking deposits), the system can:

Detect anomalies in embedding space
Assign to closest existing category
Flag for human review when confidence is low
How does your chatbot manage conversational context?
The chatbot maintains context through:

A context dictionary tracking category, source, and time range filters
Natural language parsing to extract relevant entities
Incremental context updates (only overwriting what changes)
Explicit context reset functionality

What are the main limitations?
Vague feedback challenges: Messages with limited content are difficult to classify
Multi-category overlaps: Messages may belong to multiple categories
Conversational memory constraints: Complex multi-turn discussions may lose context
Language limitations: Currently optimized for English only

How could the system be improved?
Fine-tuning BERT: Train on domain-specific data
Multi-label classification: Allow messages to belong to multiple categories
User feedback integration: Learn from human corrections
Advanced NLP: Add entity recognition and sentiment analysis
Time-series analysis: Implement more sophisticated trend detection

How would you measure and validate classification correctness?
Human evaluation: Manual review of a representative sample
Confusion matrix analysis: Precision, recall, F1-score per category
Cross-validation: K-fold validation to ensure robustness
Active learning: Prioritize review of low-confidence predictions
User feedback loop: Track when users disagree with classifications

Contact
Alenka Žumer - zumer.alenka@protonmail.com 

Project Link: https://github.com/alikova/LLM_DataScientist_ChatBot

