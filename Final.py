import numpy as np
import pandas as pd
import datetime as dt
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
import xgboost as xgb
import shap as shap

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
pd.set_option('display.width', 500)
# -------------------- Load Data --------------------
item = pd.read_csv("/Users/hilaltasdan/PycharmProjects/PythonProject2/.venv/datasourcee/olist_order_items_dataset.csv")
order = pd.read_csv("/Users/hilaltasdan/PycharmProjects/PythonProject2/.venv/datasourcee/olist_orders_dataset.csv")
product = pd.read_csv("/Users/hilaltasdan/PycharmProjects/PythonProject2/.venv/datasourcee/olist_products_dataset.csv")
payment = pd.read_csv("/Users/hilaltasdan/PycharmProjects/PythonProject2/.venv/datasourcee/olist_order_payments_dataset.csv")
customer = pd.read_csv("/Users/hilaltasdan/PycharmProjects/PythonProject2/.venv/datasourcee/olist_customers_dataset.csv")
category = pd.read_csv("/Users/hilaltasdan/PycharmProjects/PythonProject2/.venv/datasourcee/product_category_name_translation.csv")

# -------------------- Merge --------------------
df = customer.merge(order, on='customer_id')
df = df.merge(payment, on='order_id', validate='m:m')
df = df.merge(item, on='order_id')
df = df.merge(product, on='product_id')
df = df.merge(category, on='product_category_name', how='left')

# -------------------- Preprocess --------------------
df.dropna(subset=['order_purchase_timestamp'], inplace=True)

def check_df(dataframe):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(3))
    print("##################### Tail #####################")
    print(dataframe.tail(3))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.select_dtypes(include=['number']).quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)
def outlier_thresholds(dataframe,feature,perct1=0.25,perct2=0.75):
    q1 = dataframe[feature].quantile(perct1)
    q3 = dataframe[feature].quantile(perct2)
    iqr = q3 - q1
    up_limit = q3 + 1.5 * iqr
    low_limit = q1 - 1.5 * iqr
    return low_limit, up_limit
def check_outlier(dataframe,feature,perct1=0.20,perct2=0.80):
    low_limit,up_limit=outlier_thresholds(dataframe,feature,perct1,perct2)
    if dataframe[(dataframe[feature]>up_limit)|(dataframe[feature]<low_limit)].any(axis=None):
        return True
    else:
        return False

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

for col in num_cols:
    print(col,check_outlier(df,col))


def replace_w_thresholds(dataframe,variable):
    low_limit,up_limit=outlier_thresholds(dataframe,variable)
    dataframe.loc[dataframe[variable]<low_limit,variable]=low_limit
    dataframe.loc[dataframe[variable]>up_limit,variable]=up_limit

for col in num_cols:
    replace_w_thresholds(df,col)

for col in num_cols:
    print(col,check_outlier(df,col))

# Missing Values
df.isnull().sum()

#
cols_to_impute = ['product_weight_g','product_length_cm','product_height_cm','product_width_cm']

#
df_impute = df[cols_to_impute]

# KNN imputing
imputer = KNNImputer(n_neighbors=5)
df_imputed = pd.DataFrame(imputer.fit_transform(df_impute), columns=cols_to_impute)

##
df[cols_to_impute] = df_imputed
df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
df['total_price'] = df['price'] * df['order_item_id']
df.isnull().sum()

# Missing Value Features
df[df.price.isnull()].head()
columns_to_check = ['order_delivered_carrier_date', 'order_delivered_customer_date']
df.loc[df['order_delivered_carrier_date'].isnull(), 'order_delivered_carrier_date'] = 0
df.loc[df['order_delivered_customer_date'].isnull(), 'order_delivered_customer_date'] = 0


# Product Cat Name - Portuguese dropped
df.drop(axis=1,columns='product_category_name',inplace=True)
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


# -------------------- Cutoff Date --------------------
cutoff_date = dt.datetime(2018, 1, 1)
train_df = df[df['order_purchase_timestamp'] < cutoff_date]
test_df = df[df['order_purchase_timestamp'] >= cutoff_date]

# -------------------- Feature Engineering --------------------
reference_date = train_df['order_purchase_timestamp'].max()
rfm = train_df.groupby('customer_unique_id').agg(
    recency=('order_purchase_timestamp', lambda x: (reference_date - x.max()).days),
    frequency=('order_purchase_timestamp', 'nunique'),
    monetary=('total_price', 'sum')
).reset_index()
rfm = rfm[rfm['frequency'] > 0]
rfm.head()
rfm['avg_order_value'] = rfm['monetary'] / rfm['frequency']
total_items = train_df.groupby('customer_unique_id')['order_item_id'].count().reset_index(name='total_items')
avg_item_price = train_df.groupby('customer_unique_id')['price'].mean().reset_index(name='avg_item_price')
category_count = train_df.groupby('customer_unique_id')['product_category_name_english'].nunique().reset_index(name='category_count')
tenure = train_df.groupby('customer_unique_id')['order_purchase_timestamp'].agg(['min', 'max']).reset_index()
tenure['tenure_days'] = (tenure['max'] - tenure['min']).dt.days
tenure = tenure[['customer_unique_id', 'tenure_days']]
purchase_interval = train_df.groupby('customer_unique_id')['order_purchase_timestamp'].apply(
    lambda x: (x.max() - x.min()).days / (len(x)-1) if len(x) > 1 else 0
).reset_index(name='purchase_interval')
repeat_flag = rfm[['customer_unique_id', 'frequency']].copy()
repeat_flag['is_repeat_customer'] = (repeat_flag['frequency'] > 1).astype(int)

# Merging all features
features = rfm.merge(total_items, on='customer_unique_id')
features = features.merge(avg_item_price, on='customer_unique_id')
features = features.merge(category_count, on='customer_unique_id')
features = features.merge(tenure, on='customer_unique_id')
features = features.merge(purchase_interval, on='customer_unique_id')
features = features.merge(repeat_flag[['customer_unique_id', 'is_repeat_customer']], on='customer_unique_id')

# -------------------- Actual CLTV  --------------------
real_cltv = test_df.groupby('customer_unique_id')['total_price'].sum().reset_index()
real_cltv.columns = ['customer_unique_id', 'actual_cltv']

# -------------------- Merge for Modeling --------------------
data = features.merge(real_cltv, on='customer_unique_id', how='inner')
data = data[data['actual_cltv'] > 0]
data['cltv_log'] = np.log1p(data['actual_cltv'])

X = data.drop(columns=['customer_unique_id', 'actual_cltv', 'cltv_log'])
y = data['cltv_log']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "LightGBM": lgb.LGBMRegressor(objective='regression', n_estimators=500, learning_rate=0.05, max_depth=7, random_state=42),
    "RandomForest": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
    "LinearRegression": LinearRegression(),
    "XGBoost": xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=7, random_state=42, objective='reg:squarederror')
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    data[f'{name}_pred'] = model.predict(X)

    # Top-N accuracy
    top_n = 100
    top_true = set(data.sort_values('actual_cltv', ascending=False).head(top_n)['customer_unique_id'])
    top_pred = set(data.sort_values(f'{name}_pred', ascending=False).head(top_n)['customer_unique_id'])
    intersection = len(top_true & top_pred)

    results[name] = {
        'RMSE': rmse,
        'R2': r2,
        'Top100_Intersection': intersection
    }

# Results
for model_name, metrics in results.items():
    print(f"\nModel: {model_name}")
    print(f"RMSE: {metrics['RMSE']:.2f}")
    print(f"R² Score: {metrics['R2']:.2f}")
    print(f"Top 100 müşteriden {metrics['Top100_Intersection']} tanesi doğru tahmin edildi.")


xgb_model = models['XGBoost']
explainer = shap.Explainer(xgb_model)
shap_values = explainer(X)

print("\nSHAP feature importance for XGBoost:")
shap.summary_plot(shap_values, X, plot_type="bar")
# -------------------- SHAP for Tuned XGBoost --------------------
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X)  # shap_values artık np.array tipinde

print("\nSHAP feature importance for Tuned XGBoost:")
shap.summary_plot(shap_values, X, plot_type="bar")
shap.dependence_plot("recency", shap_values, X)

###########################

# -------------------- CROSS VALIDATION & FINE TUNING FOR XGBOOST --------------------
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV

# XGBoost Model with cross validation method
print("\n--- CROSS VALIDATION for current XGBoost model ---")

# X ve y
xgb_baseline = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=300,
    learning_rate=0.05,
    max_depth=7,
    random_state=42
)

cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(xgb_baseline, X, y, scoring='neg_root_mean_squared_error', cv=cv)
print("XGBoost CV RMSE Scores:", -cv_scores)
print("Mean CV RMSE:", -cv_scores.mean())

# tuning
print("\n--- FINE TUNING XGBoost with GridSearchCV ---")

param_grid = {
    'n_estimators': [300, 500],
    'learning_rate': [0.01, 0.03, 0.05],
    'max_depth': [6, 8],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 1],
    'min_child_weight': [1, 5],
}

grid_xgb = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

grid_search = GridSearchCV(
    estimator=grid_xgb,
    param_grid=param_grid,
    scoring='neg_root_mean_squared_error',
    cv=3,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X, y)
best_params = grid_search.best_params_
best_score = -grid_search.best_score_
print("Best Params:", best_params)
print(f"Best CV RMSE Score: {best_score:.4f}")

#  BEST ESTIMATOR
print("\n--- Final Model with Best Params ---")
xgb_tuned = grid_search.best_estimator_
xgb_tuned.fit(X, y)

# Top-100 Customer Comparison
data['xgb_tuned_pred'] = xgb_tuned.predict(X)
top_n = 100
top_true = set(data.sort_values('actual_cltv', ascending=False).head(top_n)['customer_unique_id'])
top_pred = set(data.sort_values('xgb_tuned_pred', ascending=False).head(top_n)['customer_unique_id'])
intersection = len(top_true & top_pred)

print(f"Top {top_n} müşteriden {intersection} tanesi doğru tahmin edildi.")

print("\n--- Done: Cross Validation + Fine Tuning for XGBoost ---")


############## Top N Approach ############

print("\n--- CUSTOM CROSS VALIDATION for Top-100 Intersection ---")

# User IDs Separation
user_ids = data['customer_unique_id'].values


# X_np, y_np = numpy array; user_ids
X_np = X.values
y_np = y.values

# K-Fold Object
kf = KFold(n_splits=5, shuffle=True, random_state=42)

intersection_scores = []

for fold_idx, (train_index, test_index) in enumerate(kf.split(X_np)):

    X_train_cv, X_test_cv = X_np[train_index], X_np[test_index]
    y_train_cv, y_test_cv = y_np[train_index], y_np[test_index]
    user_test_cv = user_ids[test_index]  # hangi kullanıcılar test setinde?

    # Model
    xgb_custom = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=300,
        learning_rate=0.05,
        max_depth=7,
        random_state=42
    )


    xgb_custom.fit(X_train_cv, y_train_cv)


    y_pred_cv = xgb_custom.predict(X_test_cv)

    #  user_id, actual, pred
    df_test_cv = pd.DataFrame({
        'customer_unique_id': user_test_cv,
        'actual_cltv': np.expm1(y_test_cv),  # log'dan geri dönüş
        'pred_cltv': np.expm1(y_pred_cv)
    })

    # Top 100 gerçekte en değerli müşteriler
    top_true_cv = set(df_test_cv.sort_values('actual_cltv', ascending=False)
                      .head(100)['customer_unique_id'])
    # Modelin top 100 tahmini
    top_pred_cv = set(df_test_cv.sort_values('pred_cltv', ascending=False)
                      .head(100)['customer_unique_id'])


    intersect_count = len(top_true_cv & top_pred_cv)

    intersection_scores.append(intersect_count)
    print(f"Fold {fold_idx + 1}: Top100 Intersection = {intersect_count}")

# Result's Averages
mean_intersection = np.mean(intersection_scores)
print(f"\nMean Top-100 Intersection across folds: {mean_intersection:.2f}")



