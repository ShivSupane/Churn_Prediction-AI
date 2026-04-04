# 🚀 Ultimate Churn Prediction Dashboard

An advanced **AI-powered Customer Churn Prediction System** built using **Machine Learning (CatBoost)** and deployed with **Streamlit**.
This dashboard enables businesses to identify at-risk customers and take proactive retention actions.

---

## 📌 Project Overview

Customer churn is a critical problem in modern businesses. This project provides:

* 🔍 **Real-time churn prediction**
* 📊 **Batch customer analysis via CSV upload**
* 📈 **Interactive visual dashboards**
* 🤖 **Machine learning-driven insights**

The system uses a trained **CatBoost Classifier** to predict the probability of customer churn based on behavioral and transactional data.

---

## 🧠 Key Features

### 🔮 Individual Prediction

* Predict churn for a single customer
* Interactive sliders for input features
* 🎯 Outputs:

  * Churn Probability (%)
  * Risk Category (High / Medium / Low)
  * 📊 Gauge Chart
  * 📉 Radar Chart
  * 💡 Recommended Action

---

### 📊 Batch Analysis

* Upload CSV file of customers
* Analyze hundreds/thousands of users instantly
* Outputs:

  * Risk segmentation (High / Medium / Low)
  * Distribution charts
  * Top high-risk customers
  * 📥 Downloadable results

---

### 📈 Model Insights

* Model type: **CatBoost Classifier**
* Accuracy: ~91%
* Features used: 44 engineered features
* Fast inference & scalable

---

## 🏗️ Tech Stack

| Layer               | Technology    |
| ------------------- | ------------- |
| Frontend            | Streamlit     |
| ML Model            | CatBoost      |
| Data Processing     | Pandas, NumPy |
| Visualization       | Plotly        |
| Model Serialization | Joblib        |

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/churn-dashboard.git
cd churn-dashboard
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the App

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
📦 churn-dashboard
 ┣ 📜 app.py
 ┣ 📜 catboost_model.pkl
 ┣ 📜 feature_scaler.pkl
 ┣ 📜 requirements.txt
 ┣ 📜 runtime.txt
 ┗ 📜 README.md
```

---

## 📊 Input Features

The model uses multiple features such as:

* Customer Demographics (Age, Membership Years)
* Behavioral Data (Login Frequency, Session Duration)
* Transactional Data (Purchases, Lifetime Value)
* Engagement Metrics (Email Open Rate, Social Activity)
* Derived Features (RFM Score, Risk Flags)

---

## 📤 CSV Format (Batch Input)

Your CSV should include columns like:

```
Age, Membership_Years, Login_Frequency, Total_Purchases,
Days_Since_Last_Purchase, Cart_Abandonment_Rate,
Lifetime_Value, Email_Open_Rate, Customer_Service_Calls
```

Missing columns will be automatically handled by the system.

---

## 🎯 Use Cases

* 🛒 E-commerce platforms
* 📱 Subscription-based services
* 🏦 FinTech / Banking
* 📊 CRM & Customer Analytics

---

## 🚀 Future Enhancements

* 🔍 SHAP Explainability (Model Transparency)
* 📡 API Deployment (FastAPI / Flask)
* ☁️ Cloud Integration (AWS / Render)
* 🔔 Real-time Alerts (Email/SMS)
* 🤖 AI Agent for automated retention

---

## 🧑‍💻 Author

**Shiv (PixalFlare Project)**

* Passionate about AI, Cloud & Full Stack Development
* Focused on building scalable, real-world ML systems

---

## ❤️ Acknowledgements

* Streamlit for UI framework
* CatBoost for powerful ML modeling
* Open-source community

---

## 📜 License

This project is for educational and research purposes.

---

## ⭐ If you like this project

Give it a ⭐ on GitHub and share it!
