# Predictive Modelling of Consumer Behavior in E-commerce

## 📌 Project Overview

This project develops an end-to-end AI pricing optimization engine for the Ponpare coupon platform. By analyzing millions of user interactions, we created a predictive model that forecasts purchase probability and recommends the optimal discount strategy to maximize Return on Investment (ROI).

**Goal:** Move beyond prediction to actionable prescription—recommending the exact discount (0%, 10%, 20%) that maximizes expected profit for each user.

---

## 🚀 Key Features

- **Advanced Feature Engineering**: Engineered behavioral features such as `user_conversion_rate` and `avg_time_between_visits`, which became top predictors.
- **Champion Model**: XGBoost classifier achieving **0.801 ROC-AUC**, outperforming Logistic Regression by >4%.
- **Profit Maximization Engine**: Financial simulation that weighs purchase probability vs. margin loss to find the optimal discount.
- **Explainable AI (XAI)**: Full SHAP integration for understanding model-driven discount recommendations.
- **Production API**: A FastAPI backend with an HTML dashboard for real-time prediction and discount optimization.

---

## 📂 Repository Structure

```
notebooks/         # Jupyter notebooks (Phase 1 to Phase 6)
models/            # Serialized models (.joblib, .pkl)
data/              # Condensed feature stores and processed datasets
project_api/       # FastAPI backend + frontend dashboard
```

---

## 🛠 Setup & Installation

### 1. Clone the repository

```
git clone https://github.com/Tanmaypathak18/Predictive-Modelling-of-Consumer-Behavior-in-E-commerce.git
cd Predictive-Modelling-of-Consumer-Behavior-in-E-commerce
```

### 2. Create and activate a virtual environment

```
python -m venv venv

# Windows\.\venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

---

## 🖥 Running the API Dashboard

A production-ready API is included to test the model’s predictions and discount recommendations.

### 1. Navigate to the API folder

```
cd project_api
```

### 2. Start the backend server

```
uvicorn api:app --reload
```

Server starts at: [**http://127.0.0.1:8000**](http://127.0.0.1:8000)

### 3. Launch the frontend dashboard

Open a new terminal and run:

```
python -m http.server 5500
```

Visit: [**http://localhost:5500**](http://localhost:5500)

### 4. Test with a sample user ID

Try example ID:

```
d9dca3cb44bab12ba313eaa681f663eb
```

This will generate an optimal discount suggestion based on live profit calculations.

---

## 📊 Project Results

| Metric      | Score                  |
| ----------- | ---------------------- |
| ROC-AUC     | 0.8010                 |
| Accuracy    | 71.15%                 |
| Top Feature | user\_conversion\_rate |

### 💼 Business Impact

The simulation found that loyal users often yield **higher profits at 0% discount**, helping avoid unnecessary revenue loss due to discount cannibalization.

---

## 📜 License

This project is open-source under the MIT License.

