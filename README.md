
# ❤️ Heart Disease Prediction using Machine Learning

This project aims to build a machine learning model that predicts the likelihood of heart disease in patients based on clinical features. It uses several classification algorithms to evaluate model performance and help identify patients at risk.

## 📊 Dataset

The dataset contains anonymized health records with features such as:
- Resting blood pressure
- Cholesterol levels
- Maximum heart rate
- ST depression (`oldpeak`)
- Chest pain type, fasting blood sugar, and more...

The target column is `target`, indicating the presence (1) or absence (0) of heart disease.

## 🔍 Problem Statement

Cardiovascular diseases are among the leading causes of death globally. Early detection through machine learning can assist medical professionals in preventive care and treatment.

---

## 🧪 Models Used

- Logistic Regression
- Decision Tree Classifier
- Random Forest
- K-Nearest Neighbors (KNN)
- Support Vector Classifier (SVC)
- Gradient Boosting Classifier
- XGBoost Classifier

The dataset was split into training and test sets using `train_test_split` with stratification to maintain class balance.

---

## ⚙️ Preprocessing

- **Null Handling**: Missing values were handled appropriately.
- **Outlier Removal**: IQR method was used on continuous features like `cholesterol`, `resting bp s`, `oldpeak`, and `max heart rate`.
- **Feature Scaling**: Applied `StandardScaler` on numerical features for normalization.

---

## 🏆 Results

| Model                  | Accuracy Score (Test) |
|-----------------------|------------------------|
| Logistic Regression   | *~To be filled*        |
| Decision Tree         | *~To be filled*        |
| Random Forest         | *~To be filled*        |
| KNN                   | *~To be filled*        |
| SVM                   | *~To be filled*        |
| Gradient Boosting     | *~To be filled*        |
| XGBoost               | *~To be filled*        |

> 📌 *Replace the above with actual scores once your model finishes training.*

---

## ⚠️ Why Accuracy May Be Limited

- The dataset size may be relatively small or imbalanced.
- Hardware limitations (e.g., training done on a personal machine with limited GPU like GTX 1650, 4GB VRAM).
- Further hyperparameter tuning and model ensembling were not fully explored.
- Advanced feature engineering could boost results.

---

## 🚀 Future Improvements

- Perform GridSearchCV or RandomizedSearchCV for hyperparameter tuning.
- Explore feature selection techniques like Recursive Feature Elimination (RFE).
- Use cross-validation for more reliable model evaluation.
- Try deep learning models if dataset is scaled up.
- Add explainability tools like SHAP or LIME.

---

## 💻 Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/heart-disease-prediction.git
   cd heart-disease-prediction
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook:
   ```bash
   jupyter notebook Heart_Disease.ipynb
   ```

> 📝 *Make sure to place your dataset CSV in the appropriate directory or update the path.*

---

## 📁 Project Structure

```
├── Heart_Disease.ipynb
├── dataset.csv
├── README.md
└── requirements.txt
```

---

## 📌 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 🤝 Contributions

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/yourusername/heart-disease-prediction/issues) if you'd like to collaborate.
