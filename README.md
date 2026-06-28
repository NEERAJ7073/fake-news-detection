\# 📰 Fake News Detection



A Machine Learning pipeline to detect fake news articles using NLP techniques.



\## 🔍 Features

\- Text preprocessing (stopwords, stemming)

\- TF-IDF + Sentiment feature engineering  

\- 6 ML models compared (Logistic Regression, SVM, Naive Bayes, etc.)

\- Visual dashboard with 8 charts



\## 📊 Results (FakeNewsNet Dataset)

| Model | Accuracy | F1-Score | AUC |

|---|---|---|---|

| Logistic Regression | 84.8% | 0.904 | 0.869 |

| Naive Bayes | 83.9% | 0.896 | 0.869 |

| Linear SVM | 84.0% | 0.896 | 0.861 |

| Random Forest | 83.0% | 0.893 | 0.851 |



\## 🚀 How to Run

pip install scikit-learn nltk pandas numpy matplotlib seaborn scipy

python3 fake\_news\_detection.py



\## 📁 Dataset

Download FakeNewsNet.csv from Kaggle and place in project folder.



\## 🛠️ Tech Stack

\- Python 3.x

\- Scikit-learn, NLTK, Pandas, Matplotlib, Seaborn

