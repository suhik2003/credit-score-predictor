# Credit Score Prediction System 🔍💳

This is a full-stack machine learning web app that predicts a person's credit score — "Good", "Average", or "Poor" — based on user input.

## 🔧 Tech Stack

- **Frontend**: HTML, CSS
- **Backend**: Flask
- **ML & Data**: Python, Pandas, NumPy, scikit-learn, Matplotlib

## 📊 Features

- Input form for age, income, loan amount, and credit history
- Predicts credit score using a trained Random Forest model
- Displays prediction and a bar chart of prediction probabilities

## 🚀 How to Run Locally

```bash
git clone https://github.com/yourusername/credit-score-predictor.git
cd credit-score-predictor
pip install -r requirements.txt
python train_model.py
python app.py
```

Then open `http://127.0.0.1:5000` in your browser.

## 📂 Project Structure

```
credit-score-predictor/
│
├── model/               # Saved ML model + scaler
├── static/              # CSS & chart images
├── templates/           # HTML files
├── app.py               # Flask backend
├── train_model.py       # Script to train the model
├── requirements.txt     # Dependencies
└── README.md            # This file
```



### 👨‍💻 Author

Suhani Kumar • [GitHub](https://github.com/suhik2003)

---

### 📜 License

MIT License
