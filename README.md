# LLM_DataScientist_ChatBot
The project involves analyzing and classifying messages from a dataset using BERT and developing a chatbot with conversational memory to respond to natural language queries with meaningful insights.

# Feedback Analysis Chatbot

An intelligent chatbot for analyzing user feedback with temporal pattern detection and (multy)category classification.

## Features
- Filter feedback by category, source, and time range
- Detect seasonal patterns and trends
- Identify spikes in feedback volume
- Handle queries spanning different categories
- Maintain conversation context for follow-up questions

## Setup
1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the web app: `streamlit run app.py` on link: https://llmdatascientistchatbot-trzggeyddrk6hhngs8vyj9.streamlit.app/

## Dataset and Data Format
Dataset is saved in the repository under the name: final_dataset_for_chatbot.csv
The chatbot expects a CSV file with the following columns:
- id_user: Unique user identifier (int)
- message: The feedback text (str)
- category: Feedback category (str)
- source: Source of the feedback (e.g., LiveChat, Telegram) (str)
- timestamp: Date and time of the feedback (int to datetime)

# User Feedback Analysis and Chatbot System
This repository contains a system for analyzing user feedback messages from LiveChat and Telegram, classifying them into actionable categories, and providing a natural language chatbot interface for querying the data.

# Project Structure
feedback-analysis-chatbot/
│
├── data/
│   ├── sample_data.csv          # Sample feedback data
│   ├── cleaned_data.csv         # Preprocessed data
│   ├── bert_embeddings.npy      # BERT embeddings for messages
│   └── final_dataset_for_chatbot.csv  # Final processed dataset
│
├── src/
│   ├── preprocessing.py         # Data cleaning and BERT tokenization
│   ├── embeddings.py            # BERT embedding generation
│   ├── categorization.py        # Message categorization logic
│   ├── openai_enhanced_chatbot.py  # Main chatbot implementation
│   └── app_with_memory.py       # Advanced Streamlit interface with memory
│
├── app.py                       # Streamlit interface (simplified)
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation


### Getting Started
Prerequisites
Python 3.7+
Google Colab (for easy execution)
Access to Google Drive (optional, for storing large datasets)
Installation

### Clone this repository:
git clone https://github.com/alikova/LLM_DataScientist_ChatBot.git

### * Optionally - Open the notebooks in Google Colab:
Navigate to Google Colab
File → Open notebook → GitHub
Enter your repository URL
Follow the step-by-step instructions in each notebook

# Setup Instructions of the repository

### Clone the repository
git clone https://github.com/yourusername/feedback-analysis-chatbot.git
cd feedback-analysis-chatbot

### Create virtual environment and install dependencies
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

### Run preprocessing and generate embeddings from LLM_DataScientist_analyse_classify_chatbot.ipnyb
python src/preprocessing.py --input data/your_data.csv --output_dir data
python src/embeddings.py --input data/cleaned_data.csv --output_dir data
python src/categorization.py --input data/cleaned_data.csv --output data/final_dataset_for_chatbot.csv --embeddings data/bert_embeddings.npy

### Run the Streamlit app
streamlit run app.py
For the enhanced version with advanced memory features and OpenAI API key (still in development):
streamlit run src/app_with_memory.py

# Usage
After launching the app, you can ask questions like:

"Provide account issues at Live chat."
"Could you show login issues from Telegram in 2024?"
"show deposit problems for LiveChat users"
"and what about telegram?"

The chatbot will process your query, apply appropriate filters, and display relevant statistics and visualizations.

# Requirements
Python 3.8+
PyTorch
Transformers (HuggingFace)
Streamlit
Pandas
Matplotlib
NLTK

# Project Overview
The system processes user feedback messages based on the categorization system.

The project is organized into three main notebooks:

### Clean and preprocess text data
Generate embeddings using BERT
Classify messages into actionable categories
Provide a conversational interface for analyzing the data

1. Data Preprocessing
Loads and cleans the raw data
Handles missing values
Tokenizes messages
Performs exploratory data analysis

### Clean and preprocess text data
Generate embeddings using BERT
Classify messages into actionable categories
Provide a conversational interface for analyzing the data

2. Model Training
Generates BERT embeddings for messages
Performs clustering to identify natural categories
Trains a classification model
Evaluates model performance

### Features
Message Classification: Automatically categorizes user messages into predefined categories
Dynamic Filtering: Supports filtering by category, time range, and source
Statistical Insights: Provides metrics like message counts, unique users, and trend detection
Conversational Memory: Maintains context across multiple queries

3. Chatbot Interface
Implements a natural language interface
Supports querying the dataset
Maintains conversational context
Provides statistical insights and visualizations


# Evaluation Questions

### How did you classify feedback?

This system implements a hybrid feedback classification approach combining BERT embeddings with weighted rule-based pattern matching. This architecture balances the semantic understanding of transformers with the precision of explicit rules.

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

BERT's semantic capabilities can position such messages near relevant domain centroids while rule-based fallbacks catch key terminology. Regular pattern updates and occasional model fine-tuning maintain classification accuracy as new issues emerge.

### How does your chatbot manage conversational context?

Conversational context is managed through a structured memory system tracking message history, active filters, and query intent (new topic, follow-up, context switch, comparison). The system maintains relevant context across interactions, allowing natural conversation flow where filters persist appropriately between related queries.
The chatbot maintains context through:

A context dictionary tracking category, source, and time range filters
Natural language parsing to extract relevant entities
Incremental context updates (only overwriting what changes)
Explicit context reset functionality

### What are the main limitations?

Vague feedback challenges: Messages with limited content are difficult to classify
Multi-category overlaps: Messages may belong to multiple categories
Conversational memory constraints: Complex discussions may lose context, memory span is currently relatively short 
Language limitations: Currently optimized for English only
User experience: Chatbot interface is simply designed with no additional options to search the data with shortcuts, buttons, etc.

Without continuous learning, the system requires manual updates to adapt to evolving language patterns.

### How could the system be improved?

Multi-label classification: Allow messages to belong to multiple categories and sub-categories
User feedback integration: Learn from human corrections
Advanced NLP: Add entity recognition and sentiment analysis
Time-series analysis: Implement more sophisticated trend detection with additional information about the source, users, similar use of words, etc.

Enhancement opportunities include implementing active learning from user corrections, developing hierarchical classification for multi-category issues, creating vector database storage for embeddings, and adding anomaly detection for emerging issue identification.
Full conversation analysis would enable resolution pattern identification, sentiment progression tracking, and agent performance evaluation - moving beyond classification toward complete interaction optimization. In case of need for simplified version, the system would be improved to classical ML methods for lower computational power.

### Explain how the chatbot tracks and utilizes past queries to refine current requests.

The chatbot tracks and utilizes past queries through a contextual reasoning system that analyzes each new query in relation to conversation history. When processing a new query, the system analyzes whether it's a follow-up (continuing the current topic), a context switch (returning to a previous topic), or a comparison request (contrasting different contexts). The system maintains filters from relevant previous queries, allowing gradual refinement of the search scope without requiring users to repeatedly specify all parameters.

### If the entire conversation were provided (a full exchange with a support agent), would you approach this task differently? (Explain how.)

If provided with the entire conversation between users and support agents, I would approach this task quite differently. Rather than focusing solely on categorizing individual messages, I would analyze the complete interaction arc, using agent responses to better understand issue patterns. This would enable training models to create suitable resolution for different issue types based on the data they source information from. I would implement sentiment and urgency tracking throughout conversations to detect escalation patterns and successful de-escalation techniques. 
Conversation flow analysis would help identify common friction points and optimal ways of responding, clarifying or proposing possible solutions. Agent response patterns could be analyzed to develop templates based on successful interactions, and user's activity mapping would provide insights into the progress from initial contact to resolution.

### How would you measure and validate classification correctness?

Human evaluation: Manual review of a representative sample
Confusion matrix analysis: Precision, recall, F1-score per category
Cross-validation: K-fold validation to ensure robustness
Active learning: Prioritize review of low-confidence predictions
User feedback loop: Track when users disagree with classifications

Classification validation while training a classification model used precision/recall metrics, cross-validation techniques, and visualization tools to correlate classification accuracy with categorized dataset.


Project is currently still in the beta phase, so bugs and performace issues might be present.

Thank you for visiting the project.

Contact
Alenka Žumer - zumer.alenka@protonmail.com 

Project Link: https://github.com/alikova/LLM_DataScientist_ChatBot

