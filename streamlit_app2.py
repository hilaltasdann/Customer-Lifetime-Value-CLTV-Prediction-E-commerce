import streamlit as st
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
pd.set_option('display.width', 500)
# -------------------- Load Data --------------------
item = pd.read_csv(
    "/Users/hilaltasdan/PycharmProjects/PythonProject2/.venv/datasourcee/olist_order_items_dataset.csv")
order = pd.read_csv("/Users/hilaltasdan/PycharmProjects/PythonProject2/.venv/datasourcee/olist_orders_dataset.csv")
product = pd.read_csv(
    "/Users/hilaltasdan/PycharmProjects/PythonProject2/.venv/datasourcee/olist_products_dataset.csv")
payment = pd.read_csv(
    "/Users/hilaltasdan/PycharmProjects/PythonProject2/.venv/datasourcee/olist_order_payments_dataset.csv")
customer = pd.read_csv(
    "/Users/hilaltasdan/PycharmProjects/PythonProject2/.venv/datasourcee/olist_customers_dataset.csv")
category = pd.read_csv(
    "/Users/hilaltasdan/PycharmProjects/PythonProject2/.venv/datasourcee/product_category_name_translation.csv")

# -------------------- Merge --------------------
df = customer.merge(order, on='customer_id')
df = df.merge(payment, on='order_id', validate='m:m')
df = df.merge(item, on='order_id')
df = df.merge(product, on='product_id')
df = df.merge(category, on='product_category_name', how='left')

# -------------------- Preprocess --------------------
df.dropna(subset=['order_purchase_timestamp'], inplace=True)
def outlier_thresholds(dataframe,feature,perct1=0.25,perct2=0.75):
    q1 = dataframe[feature].quantile(perct1)
    q3 = dataframe[feature].quantile(perct2)
    iqr = q3 - q1
    up_limit = q3 + 1.5 * iqr
    low_limit = q1 - 1.5 * iqr
    return low_limit, up_limit
def grab_col_names(dataframe,cat_th=10,car_th=20):
    # cat cols
    cat_cols=[col for col in dataframe.columns if dataframe[col].dtypes=='O']
    num_but_cat = [col for col in dataframe.columns if dataframe[col].dtypes in [int,float]
                   and dataframe[col].nunique() < cat_th]
    cat_but_car =[col for col in dataframe.columns if dataframe[col].dtypes == 'O' and
                  dataframe[col].nunique()>car_th]
    cat_cols=cat_cols+num_but_cat
    cat_cols=[col for col in cat_cols if col not in cat_but_car]
    # num cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ['int64', 'float64']]
    num_cols=[col for col in num_cols if col not in num_but_cat]

    print(f'Observations: {dataframe.shape[0]}')
    print(f'Variations: {dataframe.shape[1]}')
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car=grab_col_names(product,cat_th=10,car_th=15)

import matplotlib.pyplot as plt

def plot_numeric_distributions(df, num_cols):
    for col in num_cols:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(df[col], kde=True, bins=30, ax=ax)
        ax.set_title(f'{col} - Dağılım Grafiği')
        ax.set_xlabel(col)
        ax.set_ylabel("Frekans")
        st.pyplot(fig)  # ✅



def plot_correlation_matrix(df, num_cols):
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df[num_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title("Sayısal Değişkenler Korelasyon Matrisi")
    st.pyplot(fig)  # ✅

def plot_boxplot_by_state(df, num_col='freight_value', cat_col='customer_state'):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x=cat_col, y=num_col, data=df, ax=ax)
    ax.set_title(f'{num_col} dağılımı - {cat_col} bazında')
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)  # ✅

def plot_summary_stats(df, num_cols):
    desc = df[num_cols].describe().T[['mean', 'min', 'max']]
    fig, ax = plt.subplots(figsize=(12, 6))
    desc.plot(kind='bar', ax=ax)
    ax.set_title("Sayısal Değişkenler - Ortalama / Min / Maks")
    ax.set_ylabel("Değer")
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)  # ✅


def replace_w_thresholds(dataframe,variable):
    low_limit,up_limit=outlier_thresholds(dataframe,variable)
    dataframe.loc[dataframe[variable]<low_limit,variable]=low_limit
    dataframe.loc[dataframe[variable]>up_limit,variable]=up_limit

def run_cltv_pipeline_minimal():
    item = pd.read_csv(r"C:\Users\er_po\Desktop\final_proje\datasets\olist_order_items_dataset.csv")
    order = pd.read_csv(r"C:\Users\er_po\Desktop\final_proje\datasets\olist_orders_dataset.csv")
    product = pd.read_csv(r"C:\Users\er_po\Desktop\final_proje\datasets\olist_products_dataset.csv")
    payment = pd.read_csv(r"C:\Users\er_po\Desktop\final_proje\datasets\olist_order_payments_dataset.csv")
    customer = pd.read_csv(r"C:\Users\er_po\Desktop\final_proje\datasets\olist_customers_dataset.csv")
    category = pd.read_csv(r"C:\Users\er_po\Desktop\final_proje\datasets\product_category_name_translation.csv")

    #
    df = customer.merge(order, on='customer_id')
    df = df.merge(payment, on='order_id', validate='m:m')
    df = df.merge(item, on='order_id')
    df = df.merge(product, on='product_id')
    df = df.merge(category, on='product_category_name', how='left')

    #
    df.dropna(subset=['order_purchase_timestamp'], inplace=True)
    cat_cols, num_cols, cat_but_car = grab_col_names(product, cat_th=10, car_th=15)
    for col in num_cols:
        replace_w_thresholds(df, col)

    cols_to_impute = ['product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm']

    #
    df_impute = df[cols_to_impute]

    #
    imputer = KNNImputer(n_neighbors=5)
    df_imputed = pd.DataFrame(imputer.fit_transform(df_impute), columns=cols_to_impute)

    #
    df[cols_to_impute] = df_imputed
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    df['total_price'] = df['price'] * df['order_item_id']
    df.isnull().sum()

    #
    df[df.price.isnull()].head()
    columns_to_check = ['order_delivered_carrier_date', 'order_delivered_customer_date']
    df.loc[df['order_delivered_carrier_date'].isnull(), 'order_delivered_carrier_date'] = 0
    df.loc[df['order_delivered_customer_date'].isnull(), 'order_delivered_customer_date'] = 0

    #
    df.drop(axis=1, columns='product_category_name', inplace=True)
    df.isnull().sum()
    df.head(5)
    product.head()

    df.shape

    df.dropna(subset=[
        'product_name_lenght',
        'product_description_lenght',
        'product_photos_qty',
        'product_category_name_english'
    ], inplace=True)

    df.dropna(subset=['order_approved_at'], inplace=True)

    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    df['total_price'] = df['price'] * df['order_item_id']

    # --- Train/Test by cutoff ---
    cutoff_date = dt.datetime(2018, 1, 1)
    train_df = df[df['order_purchase_timestamp'] < cutoff_date]
    test_df = df[df['order_purchase_timestamp'] >= cutoff_date]

    # --- RFM Features ---
    reference_date = train_df['order_purchase_timestamp'].max()
    rfm = train_df.groupby('customer_unique_id').agg(
        recency=('order_purchase_timestamp', lambda x: (reference_date - x.max()).days),
        frequency=('order_purchase_timestamp', 'nunique'),
        monetary=('total_price', 'sum')
    ).reset_index()

    # Sıfır frequency varsa at
    rfm = rfm[rfm['frequency'] > 0]

    # Ek feature
    rfm['avg_order_value'] = rfm['monetary'] / rfm['frequency']

    total_items = train_df.groupby('customer_unique_id')['order_item_id'].count().reset_index(name='total_items')
    avg_item_price = train_df.groupby('customer_unique_id')['price'].mean().reset_index(name='avg_item_price')
    category_count = train_df.groupby('customer_unique_id')['product_category_name_english'].nunique().reset_index(
        name='category_count')
    tenure = train_df.groupby('customer_unique_id')['order_purchase_timestamp'].agg(['min', 'max']).reset_index()
    tenure['tenure_days'] = (tenure['max'] - tenure['min']).dt.days
    tenure = tenure[['customer_unique_id', 'tenure_days']]
    purchase_interval = train_df.groupby('customer_unique_id')['order_purchase_timestamp'].apply(
        lambda x: (x.max() - x.min()).days / (len(x) - 1) if len(x) > 1 else 0
    ).reset_index(name='purchase_interval')
    repeat_flag = rfm[['customer_unique_id', 'frequency']].copy()
    repeat_flag['is_repeat_customer'] = (repeat_flag['frequency'] > 1).astype(int)

    # Merge all features
    features = rfm.merge(total_items, on='customer_unique_id')
    features = features.merge(avg_item_price, on='customer_unique_id')
    features = features.merge(category_count, on='customer_unique_id')
    features = features.merge(tenure, on='customer_unique_id')
    features = features.merge(purchase_interval, on='customer_unique_id')
    features = features.merge(repeat_flag[['customer_unique_id', 'is_repeat_customer']], on='customer_unique_id')

    # Actual CLTV from test
    real_cltv = test_df.groupby('customer_unique_id')['total_price'].sum().reset_index()
    real_cltv.columns = ['customer_unique_id', 'actual_cltv']

    data = features.merge(real_cltv, on='customer_unique_id', how='inner')
    data = data[data['actual_cltv'] > 0]  # 0'dan büyük
    data['cltv_log'] = np.log1p(data['actual_cltv'])

    X = data.drop(columns=['customer_unique_id', 'actual_cltv', 'cltv_log'])
    y = data['cltv_log']

    # Cross Validation (XGBoost)
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=300,
        learning_rate=0.05,
        max_depth=7,
        random_state=42
    )

    from sklearn.model_selection import KFold, cross_val_score
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(xgb_model, X, y, scoring='neg_root_mean_squared_error', cv=cv)
    rmse_scores = -cv_scores
    mean_rmse = -cv_scores.mean()

    # Fit final
    xgb_model.fit(X, y)
    data['xgb_pred'] = xgb_model.predict(X)

    # Top 100
    top_n = 100
    top_true = set(data.sort_values('actual_cltv', ascending=False).head(top_n)['customer_unique_id'])
    top_pred = set(data.sort_values('xgb_pred', ascending=False).head(top_n)['customer_unique_id'])
    intersection = len(top_true & top_pred)

    # SHAP
    explainer = shap.Explainer(xgb_model)
    shap_values = explainer(X)

    # Sonuçları dict döndürelim
    return {
        "rmse_scores": rmse_scores,
        "mean_rmse": mean_rmse,
        "top_100_intersection": intersection,
        "shap_values": shap_values,
        "X": X,
        "data_sample": data.head(10)
    }

###############################
# Streamlit Interface
###############################

def main():

    st.title(":blue[RFM & CLTV Analysis]")
    st.subheader("Unlocking Growth of Brazilian E-commerce Brand")
    st.write("Presented by: Burak Yilkin, Erkan Polat, Hilal Tasdan")

    # SLIDE 1
    st.subheader("The Story")
    st.write("This dataset was generously provided by Olist, the largest department store in Brazilian marketplaces. Olist connects small businesses from all over Brazil to channels without hassle and with a single contract. Those merchants are able to sell their products through the Olist Store and ship them directly to the customers using Olist logistics partners. See more on the website: www.olist.com")
    st.write("The dataset has information of 100k orders from 2016 to 2018 made at multiple marketplaces in Brazil. Its features allows viewing an order from multiple dimensions: from order status, price, payment and freight performance to customer location, product attributes and finally reviews written by customers. We also released a geolocation dataset that relates Brazilian zip codes to lat/lng coordinates.")
    st.write("Example of a product listing on a marketplace:")
    st.image("/Users/hilaltasdan/PycharmProjects/PythonProject2/.venv/e-commerce_image.png")

    st.subheader("Project Objective – Predicting CLTV")
    st.write("""
        - **Goal**: Identify the most valuable customers for targeted marketing strategies.
        - **What is CLTV?:** A prediction of the total reve
        nue a customer will generate during their relationship with the business.
        - **In this project:** We use e-commerce data to build machine learning models that predict CLTV and interpret model decisions with SHAP.        
    """"")
    st.write("Why it matters:")
    st.write("""
        - Optimizes marketing spend.
        - Improves customer retention and loyalty.
        - Increases profitability.
    """"")
    st.image("/Users/hilaltasdan/PycharmProjects/PythonProject2/.venv/segmentations.jpg")

    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")

    # SLIDE 2
    st.title(" Data Sources and Overview")

    # Subtitle
    st.subheader("Datasets Used (over 9 MB)")

    # Dataset list
    st.markdown("""
    - `olist_customers_dataset.csv`
    - `olist_geolocation_dataset.csv`
    - `olist_order_items_dataset.csv`
    - `olist_order_payments_dataset.csv`
    - `olist_order_reviews_dataset.csv`
    - `olist_orders_dataset.csv`
    - `olist_products_dataset.csv`
    - `olist_sellers_dataset.csv`
    - `product_category_name_translation.csv`  
    """)

    # Data context
    st.subheader("Data Context")
    st.markdown("""
    - The data is sourced from a Brazilian e-commerce platform, spanning **2016 to 2018**.
    - It includes detailed information on:
      - Customer behavior
      - Product characteristics
      - Transaction records
    """)

    # Processing details
    st.subheader("Data Processing")
    st.markdown("""
    - All datasets were merged using common keys:
      - `order_id`
      - `customer_id`
    """)
    st.image("/Users/hilaltasdan/PycharmProjects/PythonProject2/.venv/data_image.png")
    st.write(" **Result:** A unified dataset combining customer, order, payment, and product information.")

    st.write("")
    st.write("")
    st.write("")

    # SLIDE 2
    st.title("Explatory Data Analysis(EDA)")
    st.write("We analyze and investigate data sets and summarize their main characteristics and applied data visualization methods.")

    plot_numeric_distributions(df, num_cols)

    st.subheader("Korelasyon Matrisi")
    plot_correlation_matrix(df, num_cols)

    # SLIDE 3
    st.title("Data Cleaning and Preprocessing")

    st.subheader("Missing Values")
    st.markdown("""
    - Dropped rows with missing values in the `order_purchase_timestamp` column.
    """)

    st.subheader("Data Merging")
    st.markdown("""
    - Combined multiple datasets on `order_id` and `customer_id`.
    - Removed redundant or duplicate columns to prevent data leakage or noise.
    """)

    st.subheader("Quality Check")
    st.markdown("""
    - Used a custom helper function `check_df()` to:
      - Inspect data types
      - Check for null values
      - Preview sample rows
    """)
    st.image("/Users/hilaltasdan/PycharmProjects/PythonProject2/.venv/kopke.jpg")
    st.success("**Goal:** Prepare a clean and integrated dataset suitable for feature engineering and modeling.")

    # SLIDE 4: Feature Engineering
    st.title(" Feature Engineering")

    st.subheader("Created RFM Features")
    st.markdown("""
    - `Recency`: Days since the last purchase
    - `Frequency`: Total number of completed orders
    - `Monetary`: Total amount paid by the customer
    """)

    st.subheader("Additional Features")
    st.markdown("""
    - Average payment amount per order
    - Number of unique product categories purchased
    - Total number of payment installments
    """)

    st.success("Purpose: Capture meaningful purchasing behavior patterns to enhance model performance.")

    st.write("")
    st.write("")
    st.write("")

    # SLIDE 5
    st.title("Modeling Approach")

    st.subheader("Models Used")
    st.markdown("""
    - **RandomForestRegressor**  
      Ensemble model using decision trees with bootstrapping.

    - **LinearRegression**  
      A baseline model for linear relationships.

    - **XGBoostRegressor**  
      Gradient boosting with regularization and high performance.  

    - **LightGBMRegressor**  
      Fast, efficient gradient boosting optimized for large datasets.
    """)
    st.subheader("Custom Cross-Validation")
    st.markdown("""
    - Instead of using traditional **RMSE or MAE**, we used a **custom evaluation loop** to assess performance.
    - The next slide will explain this custom metric in detail.
    """)
    
    with st.spinner("Pipeline Çalışıyor..."):
        results = run_cltv_pipeline_minimal()
    st.write(results["rmse_scores"])

    st.write(f"Mean CV RMSE: {results['mean_rmse']:.2f}")

    st.subheader("Evaluation Focus")
    st.markdown("""
       - We focused on the **accuracy of identifying top high-value customers** (Top-100 precision), not just prediction error.
       """)

    st.subheader("Top-100 Intersection")
    st.write(f"{results['top_100_intersection']} / 100")

    # 3) Data Sample
    st.subheader("Data Sample")
    st.dataframe(results["data_sample"])

    st.write("")
    st.write("")
    st.write("")

    # 4) SHAP Feature Importance (Bar)
    # SLIDE 7: Model Interpretability with SHAP
    st.title(" Model Interpretability with SHAP")

    st.subheader(" What is SHAP?")
    st.markdown("""
    - **SHAP** stands for **Shapley Additive Explanations**.
    - It explains **how much each feature contributes** to a model's prediction for each individual customer.
    - Based on cooperative game theory — assigns a "fair value" to each feature.

    """)

    st.subheader("How SHAP Was Used")
    st.markdown("""
    - Applied to the **XGBoost** and **LightGBM** models.
    - Used SHAP summary plots and decision plots to interpret feature impacts.
    - Helps **visualize and explain** why the model predicted high or low CLTV for each customer.
    """)

    st.subheader("Top Influential Features")
    st.markdown("""
    - **Average Payment Amount** – Higher average payments led to higher predicted CLTV.
    - **Recency** – More recent buyers were predicted to have higher CLTV.
    - **Frequency** – Repeat customers had stronger lifetime value predictions.
    """)

    st.subheader("SHAP Feature Importance (Bar)")
    fig, ax = plt.subplots()
    shap.plots.bar(results["shap_values"], show=False)
    st.pyplot(fig)
    st.success("Why It Matters: SHAP improves model **transparency**, builds **trust**, and supports **data-driven decisions**.")

    # 5) SHAP Beeswarm
    st.subheader("SHAP Beeswarm Plot")
    fig2, ax2 = plt.subplots()
    shap.summary_plot(results["shap_values"], results["X"], show=False)
    st.pyplot(fig2)

    st.write("")
    st.write("")
    st.write("")


    # SLIDE 8
    st.title("Conclusion and Recommendations")

    st.subheader("Best Model")
    st.markdown("""
    - **XGBoostRegressor** showed the best performance.
    - Achieved approximately **65% Top-100 customer match accuracy** using our custom evaluation metric.
    """)

    st.subheader("Business Applications")
    st.markdown("""
    - **Marketing:** Launch personalized campaigns for top-value customers.
    - **Sales:** Develop loyalty programs and upsell strategies.
    - **Product Development:** Use insights to guide product recommendations and personalization.
    """)

    st.subheader("Next Steps")
    st.markdown("""
    - **Real-time Integration:** Deploy the model for real-time customer segmentation.
    - **Feature Expansion:** Incorporate time-series data or customer interaction behavior.
    - **Advanced Modeling:** Explore deep learning models for further accuracy improvements.
    """)

    st.success("Final Takeaway: Data-driven CLTV prediction helps prioritize key customers and boost long-term business value.")


if __name__ == "__main__":
    main()
