# Automated Categorization of Citizen Inquiries

## Project Overview

This project was developed as part of an internship at the State Corporation "Government for Citizens." The primary goal is to design, implement, and evaluate a machine learning model capable of automatically categorizing incoming citizen inquiries based on their textual content. Accurate and efficient categorization can help in routing inquiries to the appropriate departments more quickly, thereby improving response times and overall citizen satisfaction.

This repository contains the Jupyter Notebooks, (sample) data, and scripts used for this project.

## Table of Contents

* [Project Overview](#project-overview)
* [Dataset](#dataset)
* [Technologies Used](#technologies-used)
* [Project Structure](#project-structure)
* [Setup and Installation](#setup-and-installation)
* [Usage](#usage)
* [Results](#results)

## Dataset

The model was trained and evaluated on an anonymized dataset of past citizen inquiries provided by the State Corporation "Government for Citizens." For the purpose of reproducibility within this repository, a small, representative, and further anonymized sample (`sample_data/sample_inquiries.csv`) is conceptually included. The dataset contains the text of each inquiry and its manually assigned category.

**Note:** The full dataset used for the actual internship work is confidential and not included in this public-facing conceptual repository.

## Technologies Used

* **Programming Language:** Python 3.x
* **Core Libraries:**
    * Pandas: For data manipulation and analysis.
    * NumPy: For numerical operations.
    * Scikit-learn: For machine learning tasks (TF-IDF, classification algorithms, evaluation metrics).
    * NLTK (Natural Language Toolkit) / spaCy: For text preprocessing (tokenization, lemmatization, stop-word removal).
    * Matplotlib & Seaborn: For data visualization.
* **Development Environment:** Jupyter Notebooks
* **Version Control:** Git

## Project Structure

citizen_inquiry_classification/
│
├── notebooks/
│   ├── 01_Data_Loading_and_EDA.ipynb        # Data loading, initial inspection, and EDA
│   ├── 02_Text_Preprocessing.ipynb          # Text cleaning and preparation
│   └── 03_Model_Training_and_Evaluation.ipynb # Feature extraction, model training, and evaluation
│
├── scripts/                                 # Optional: .py utility scripts
│   └── (empty or utility_functions.py)
│
├── sample_data/
│   └── sample_inquiries.csv                 # Conceptual sample of anonymized data
│
├── requirements.txt                         # Python dependencies
└── README.md                                # This file

## Setup and Installation

1.  **Clone the repository (Conceptual):**
    ```bash
    git clone <repository-url>
    cd citizen_inquiry_classification
    ```

2.  **Create a Python virtual environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    If using NLTK for the first time, you might also need to download specific resources:
    ```python
    # Run this in a Python interpreter or a notebook cell
    import nltk
    nltk.download('punkt')      # For tokenization
    nltk.download('stopwords')  # For stop words
    nltk.download('wordnet')    # For lemmatization
    nltk.download('omw-1.4')    # Open Multilingual Wordnet, needed by WordNetLemmatizer
    ```

## Usage

The project is primarily organized into Jupyter Notebooks within the `notebooks/` directory. It is recommended to run them in the specified order:

1.  **Start Jupyter Notebook server:**
    ```bash
    jupyter notebook
    ```
2.  Open and run the notebooks sequentially:
    * `01_Data_Loading_and_EDA.ipynb`
    * `02_Text_Preprocessing.ipynb`
    * `03_Model_Training_and_Evaluation.ipynb`

Each notebook contains detailed explanations and code for its respective stage of the project.

## Results

The models were evaluated based on standard classification metrics such as accuracy, precision, recall, and F1-score. The `03_Model_Training_and_Evaluation.ipynb` notebook contains the detailed evaluation results and comparisons between different algorithms (e.g., Naive Bayes, Logistic Regression, SVM). Based on the evaluation, Logistic Regression demonstrated a strong balance of performance and efficiency for this task.

---

**Author:** [Dariya Khussainova] (Machine Learning Intern)
**Date:** [05.07.2025]