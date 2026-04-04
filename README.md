---
title: Ultimate Churn Prediction Dashboard
emoji: 🤖
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
python_version: 3.11
pinned: true
license: mit
---

# 🤖 Ultimate Customer Churn Prediction Dashboard

**The Most Advanced ML Deployment for Customer Churn Prediction**

[![Model](https://img.shields.io/badge/Model-XGBoost-orange)](https://xgboost.readthedocs.io/)
[![Accuracy](https://img.shields.io/badge/Accuracy-90.97%25-success)](.)
[![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.9284-blue)](.)
[![Platform](https://img.shields.io/badge/Platform-Hugging%20Face-yellow)](https://huggingface.co/)

## 🎯 Features

### 📊 Batch Analysis
- **Upload CSV**: Process thousands of customers simultaneously
- **Real-time Analytics**: Interactive Plotly charts
- **Risk Segmentation**: Automatic HIGH/MEDIUM/LOW classification
- **Download Results**: Export complete analysis as CSV
- **Visual Insights**: Pie charts, histograms, scatter plots

### 🔮 Individual Prediction
- **Interactive Sliders**: Easy customer data input
- **Instant Predictions**: <100ms inference time
- **Churn Probability Gauge**: Stunning speedometer visualization
- **Customer Health Radar**: 6-dimension analysis chart
- **Actionable Recommendations**: Retention strategies for each risk level

### 📈 Model Performance
- **Accuracy**: 90.97%
- **ROC-AUC**: 0.9284
- **Precision**: 89.25%
- **Recall**: 92.30%
- **F1-Score**: 90.75%
- **Features**: 44 engineered features

## 🚀 Technology Stack

- **ML Model**: XGBoost Classifier (actual .pkl file, NOT approximation!)
- **UI Framework**: Gradio 4.44.1
- **Visualization**: Plotly (interactive charts)
- **Feature Engineering**: RFM analysis, behavioral scoring
- **Scaling**: StandardScaler (from scikit-learn)
- **Deployment**: Hugging Face Spaces (100% FREE)

## 💡 How to Use

### Batch Prediction
1. Click "Batch Analysis" tab
2. Upload CSV with customer data
3. Click "Analyze All Customers"
4. View statistics, charts, top 20 high-risk customers
5. Download complete results

### Single Customer
1. Click "Individual Prediction" tab
2. Adjust sliders for customer details
3. Click "Predict Churn Probability"
4. View risk level, probability gauge, health radar
5. Read recommended retention action

## 📊 Model Details

### Training
- **Training Set**: 40,000 customers
- **Test Set**: 10,000 customers
- **Features**: 44 engineered features
- **Algorithm**: XGBoost with hyperparameter tuning
- **Cross-Validation**: 5-fold stratified CV

### Feature Engineering
- **RFM Features**: Recency, Frequency, Monetary (normalized)
- **Behavioral**: Purchase rate, Engagement score
- **Binary Flags**: High-value, At-risk, Frequent buyer
- **Encoding**: Label encoding for categorical variables
- **Scaling**: StandardScaler (mean=0, std=1)

### Performance Comparison
| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| 🥇 **XGBoost** | **90.97%** | **0.9284** |
| 🥈 CatBoost | 90.92% | 0.9283 |
| 🥉 LightGBM | 90.80% | 0.9261 |
| Random Forest | 90.62% | 0.9250 |
| Deep Learning | ~82% | ~0.87 |
| Logistic Regression | 76.42% | 0.8447 |

## 🎨 UI Features

- **Stunning Dark Mode**: Gradient backgrounds, glass morphism
- **Responsive Design**: Works on desktop, tablet, mobile
- **Interactive Charts**: Plotly visualizations
- **Real-time Updates**: Instant predictions
- **Professional Styling**: Custom CSS, modern fonts

## 🔒 Privacy & Security

- **No Data Storage**: Predictions processed in real-time, nothing saved
- **Open Source**: Full code transparency
- **Secure Platform**: Hosted on Hugging Face infrastructure

## 📝 License

MIT License - Free to use for educational and commercial purposes

## 👨‍💻 Author

**Shiv** - Data Scientist & ML Engineer
- Specialization: Customer Analytics, Churn Prediction
- ML Models: XGBoost, LightGBM, CatBoost, Deep Learning

## 🙏 Acknowledgments

- **Platform**: Hugging Face Spaces (100% FREE hosting!)
- **Framework**: Gradio (amazing UI framework)
- **ML Library**: XGBoost (best-in-class gradient boosting)
- **Visualization**: Plotly (interactive charts)

---

**Note**: This deployment uses the ACTUAL trained XGBoost model (.pkl file),  
NOT JavaScript approximations. All predictions are made using the real model  
with 90.97% accuracy achieved during training.

**Deployment Status**: ✅ Production-Ready | 🟢 Active | ⚡ <100ms Latency
