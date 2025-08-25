# Customer-Lifetime-Value-CLTV-Prediction-E-commerce
An end-to-end data science project predicting Customer Lifetime Value (CLTV) for an e-commerce platform using machine learning and presenting results via an interactive Streamlit dashboard.
The dataset is sourced from the Olist Brazilian E-commerce Dataset, containing detailed order, product, payment, and customer data.

1. Project Overview
This project focuses on:
Predicting future revenue potential of customers to optimize marketing spend and retention strategies.
Building ML models (XGBoost, LightGBM, Random Forest, Linear Regression) to forecast CLTV.
Explaining model predictions with SHAP for transparent, actionable insights.
Providing an interactive dashboard to explore data, model results, and business recommendations.

2. Features
- Data Processing & Feature Engineering
Combined multiple relational datasets (orders, items, payments, products, customers).
Handled missing values with KNN imputation and treated outliers using IQR.
Created RFM metrics (Recency, Frequency, Monetary), purchase intervals, repeat purchase flags, and more.

- Machine Learning
Implemented XGBoost, LightGBM, Random Forest, Linear Regression.
Evaluated using RMSE, R², and Top-N (Top 100 customers) accuracy.
Hyperparameter tuning with GridSearchCV for optimal results.

- Model Interpretability
SHAP feature importance plots and per-customer explanations.
Identified key drivers: Recency, Average Order Value, Frequency.

- Interactive Dashboard (Streamlit)
Data Overview: Dataset intro, context, and preprocessing steps.
Modelling Results: Performance metrics, Top-N customer predictions.
SHAP Insights: Feature importance, beeswarm plots, and transparency.
Business Recommendations: Marketing, retention, and revenue strategies.

3. Project Structure
.
├── data/                          
├── notebooks/                    
├── src/
│   ├── preprocessing.py         
│   ├── modeling.py                
│   ├── explainability.py       
│
├── streamlit_app/
│   ├── app.py                     
│   ├── images/                    
│
├── requirements.txt              
├── README.md                 


4. Installation
- Clone the repository
git clone https://github.com/yourusername/cltv-prediction.git
cd cltv-prediction

- Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # (Mac/Linux)
venv\Scripts\activate         # (Windows)

- Install dependencies
pip install -r requirements.txt

5. Usage
- Run the CLTV pipeline (training & evaluation)
python src/modeling.py

- Launch the Streamlit dashboard
cd streamlit_app
streamlit run app.py

- Open the provided local URL (e.g., http://localhost:8501) to view the interactive dashboard.

6. Results

- Best Model: XGBoost achieved the highest predictive performance.
- Top-100 High-Value Customer Accuracy: ~65%.
- Key Predictors: Recency, Frequency, Average Payment Amount.
- Impact: Enables data-driven customer segmentation and targeted marketing strategies.


7. Future Work
Real-time deployment for live CLTV scoring.
Incorporating customer engagement and behavioral features.
Exploring deep learning architectures for improved accuracy.

9. Tech Stack
- Python: Pandas, NumPy, scikit-learn, XGBoost, LightGBM, SHAP, Matplotlib
- Streamlit: Interactive web app
- Machine Learning & Explainable AI

10. Acknowledgements
Dataset provided by Olist
.
