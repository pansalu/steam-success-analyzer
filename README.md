#  Steam Game Success Analyzer

I built this project because I wanted to understand what separates successful games from ones that flop on Steam. Turns out developer reputation matters way more than genre or price which honestly surprised me.

Started with a Kaggle dataset of 27,000 games, did a lot of cleaning and feature engineering, trained a few models, and ended up with 86% accuracy on a 3-class prediction problem. Also built a Streamlit app so anyone can actually use it without touching the code.

---

## 🎯 What This Project Does

Analyzes 20,000+ Steam games and predicts whether a new game will be:
 **High Success** — 85%+ positive reviews AND 2000+ total reviews
 **Medium Success** — 70%+ positive reviews AND 200+ total reviews
 **Low Success** — Everything else

---

## 📊 Key Findings

- **Developer reputation** is the single strongest predictor of success — more important than genre, price, or tags combined
- Price has surprisingly weak correlation with review score
- Steam releases exploded after 2013 when it opened to indie developers
- Adding playtime and platform count features improved accuracy by 3%

---

## Models Trained

| Model | Accuracy | High F1 | Medium F1 |
|-------|----------|---------|-----------|
| Logistic Regression | 65% | 0.34 | 0.33 |
| Random Forest | 85% | 0.60 | 0.58 |
| Tuned XGBoost | 86% | 0.63 | 0.61 |

---

##  Tech Stack

- Python, Pandas, NumPy
- Scikit-learn, XGBoost
- Matplotlib, Seaborn
- Streamlit

---

## 📁 Project Structure

```
steam-success-analyzer/
├── data/
│   ├── raw/              # Original Kaggle dataset
│   └── processed/        # Cleaned and feature engineered data
├── notebooks/
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_evaluation.ipynb
├── models/               # Saved ML models
├── app.py                # Streamlit dashboard
├── requirements.txt
└── README.md
```

##  How to Run

pip install -r requirements.txt
python -m streamlit run app.py

---

## 📬 Contact

Built by Panshul Sudhakaran — [LinkedIn](https://www.linkedin.com/in/panshul-sudhakaran-192174413/) | [GitHub](https://github.com/pansalu)