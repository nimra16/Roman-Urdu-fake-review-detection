# Fake Reviews Detection on E-Commerce Websites using Novel User Behavioral Features

This repository contains the implementation and experimental results for the research paper "Fake Reviews Detection on E-Commerce Websites using Novel User Behavioral Features: An Experimental Study" focusing on Roman Urdu fake review detection.

## 📋 Table of Contents

- [Overview](#overview)
- [Authors](#authors)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Features](#features)
- [Experimental Settings](#experimental-settings)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Citation](#citation)
- [License](#license)

## 🔍 Overview

The trend of writing fake reviews has recently increased with the rapid growth of e-commerce websites. This research addresses the challenge of detecting fake reviews in **Roman Urdu**, a low-resource language, by introducing novel user behavioral features alongside traditional textual and lingual features.

### Key Contributions:

- **Novel Feature Engineering**: Identification of three types of discriminative features:
  - Review textual features
  - Review lingual features
  - Review behavioral features
- **Class Imbalance Handling**: Evaluation of LSTM-based text generation, random undersampling (RUS), and oversampling (ROS) techniques
- **Comprehensive Evaluation**: Empirical comparison of machine learning and deep learning algorithms
- **Performance Enhancement**: 3% accuracy improvement over baseline studies using Gradient Boosting

## 👥 Authors

- **Nimra Mughal** - Center of Excellence for Robotics, AI and Blockchain (CRAIB), Sukkur IBA University
- **Ghulam Mujtaba** - Center of Excellence for Robotics, AI and Blockchain (CRAIB), Sukkur IBA University
- **Muhammad Hussain Mughal** - Department of Computer Science, Sukkur IBA University
- **Abdul Manaf** - Center of Excellence for Robotics, AI and Blockchain (CRAIB), Sukkur IBA University
- **Zainab Umair Kamangar** - Department of Computer Science, Sukkur IBA University

## 📊 Dataset

This repository contains processed data with extracted features from the **Roman Urdu fake review dataset** originally accessed from:

- **Source**: [Kaggle - Daraz Roman Urdu Reviews](https://www.kaggle.com/datasets/naveedhn/daraz-roman-urdu-reviews)
- **Platform**: Daraz (Leading e-commerce platform in Pakistan)
- **Language**: Roman Urdu
- **Files**:
  - `clean_train.csv` - Training dataset with extracted features
  - `clean_test.csv` - Testing dataset with extracted features

## 🔬 Methodology

The research follows a comprehensive 6-step methodology for detecting fake reviews in Roman Urdu:

![Methodology Diagram](<figures/Methodology.drawio%20(1).png>)

### Pipeline Overview:

1. **Text Preprocessing**: Remove special characters, whitespace, and single character words; Apply Roman Urdu stopword removal
2. **Feature Extraction**: Extract three types of features - Review Textual Features (RTF), Review Lingual Features (RLF), and User Behavioral Features (UBF)
3. **Data Split**: 90% for training and validation, 10% for testing
4. **Resampling**: Apply hybrid oversampling (LSTM text generation for RTF, oversampling for RLF and UBF) and undersampling techniques
5. **Classification**: Use both Deep Learning models (with RTF, RLF, UBF) and Machine Learning models (with RLF+UBF) including the proposed Gradient Boosting Algorithm
6. **Model Evaluation**: Assess performance using accuracy, precision, recall, F-score, and confusion matrix

## 🎯 Features

### 1. **Textual Features (RTF)**

- TF-IDF vectors
- N-gram features
- Text length and structure

### 2. **Lingual Features (RLF)**

- Language-specific patterns
- Roman Urdu linguistic characteristics
- Sentiment indicators

### 3. **Behavioral Features (UBF)** (Novel Contribution)

- User review patterns
- Review frequency analysis
- Rating behavior
- Temporal patterns

## 🧪 Experimental Settings

The research evaluates multiple experimental configurations:

1. **Setting I**: Default train-test split with ML algorithms
2. **Setting II**: Cross-validation evaluation
3. **Setting III**: Feature selection techniques
4. **Setting IV**: Class imbalance handling
   - Random Oversampling (ROS)
   - Random Undersampling (RUS)
   - SMOTE (Synthetic Minority Oversampling Technique)
5. **Setting V**: Feature selection + Undersampling
   - Chi-square test
   - Information Gain (IG)
   - Recursive Feature Elimination (RFE)

## 📁 Project Structure

```
├── README.md                           # Project documentation
├── clean_train.csv                     # Training dataset
├── clean_test.csv                      # Testing dataset
├── Train_and_test_baseline.ipynb       # Baseline model implementation
├── Experiment_Feature_Extraction.ipynb # Feature extraction pipeline
├── Experimental results/               # Results and visualizations
│   ├── expsettingI_ML_updated80.csv   # Setting I results (80% split)
│   ├── expsettingI_ML_updated90.csv   # Setting I results (90% split)
│   ├── expsettingIII_80.csv          # Setting III results
│   ├── expsettingIV_*.csv            # Setting IV results (ROS, RUS, SMOTE)
│   ├── expsettingV_*.csv             # Setting V results
│   └── *.png                         # Performance visualizations
├── ExperimentalCodes_updated/         # Updated experimental code
│   ├── DL_Experiments/               # Deep learning experiments
│   └── ML_experiments/               # Machine learning experiments
└── figures/                          # Research figures and plots
    ├── cloud_cleaned.png             # Word cloud (cleaned data)
    ├── cloud_raw.png                 # Word cloud (raw data)
    ├── dataset.png                   # Dataset statistics
    ├── model_*.png                   # Model architectures
    └── *.png                         # Other visualizations
```

## 🛠️ Installation

### Prerequisites

- Python 3.7+
- Jupyter Notebook
- Required Python packages (see requirements below)

### Setup

```bash
# Clone the repository
git clone https://github.com/nimra16/Roman-Urdu-fake-review-detection.git
cd Roman-Urdu-fake-review-detection

# Install required packages
pip install pandas numpy matplotlib seaborn scikit-learn
pip install nltk imbalanced-learn tensorflow keras
pip install jupyter notebook

# Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

## 🚀 Usage

### 1. Feature Extraction

```bash
jupyter notebook Experiment_Feature_Extraction.ipynb
```

### 2. Baseline Training and Testing

```bash
jupyter notebook Train_and_test_baseline.ipynb
```

### 3. Run Specific Experiments

Navigate to `ExperimentalCodes_updated/` and run the desired experiment notebooks:

- `ML_experiments/` - Machine Learning experiments
- `DL_Experiments/` - Deep Learning experiments

### 4. View Results

Check the `Experimental results/` directory for:

- CSV files with detailed metrics
- PNG files with performance visualizations

## 📈 Results

### Key Findings:

- **User behavioral features** play a vital role in detecting fake reviews
- **Text generation** is ineffective for balancing textual data
- **Gradient Boosting (GB)** outperformed other models
- **3% accuracy improvement** over baseline studies

### Best Performance:

- **Algorithm**: Gradient Boosting Classifier
- **Feature Combination**: Textual + Lingual + Behavioral features
- **Preprocessing**: Random Undersampling + Feature Selection

### Performance Metrics:

Results are available in the `Experimental results/` directory with detailed accuracy, precision, recall, and F1-score metrics for different experimental settings.

## 📚 Citation

If you use this code or dataset in your research, please cite:

```bibtex
@article{mughal2025fake,
  author = {N. Mughal and G. Mujtaba and M. H. Mughal and A. Manaf and Z. Kamangar},
  title = {Fake Reviews Detection on E-Commerce Websites Using Novel User Behavioral Features: An Experimental Study},
  journal = {ACM Transactions on Asian and Low-Resource Language Information Processing},
  year = {2025},
  month = jul,
  doi = {https://doi.org/10.1145/3748493}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

We welcome contributions! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📧 Contact

For questions or collaboration opportunities, please contact:

- Nimra Mughal: nimra.mscsf19@iba-suk.edu.pk
- Abdul Manaf: abdul.manaf@iba-suk.edu.pk
- Ghulam Mujtaba: mujtaba@iba-suk.edu.pk

---

**Keywords**: Fake Reviews, Roman Urdu, User Behavioral Features, Machine Learning, Deep Learning, LSTM-based Text Generation, Feature Selection, Class Imbalance
