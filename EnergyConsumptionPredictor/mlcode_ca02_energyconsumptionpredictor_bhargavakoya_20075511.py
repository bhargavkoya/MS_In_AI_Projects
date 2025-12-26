# -*- coding: utf-8 -*-
"""ML_CA02_EnergyConsumptionPredictor_BhargavaKoya_20075511.ipynb

Regression Problem: Predict continuous energy consumption (kWh) based on environmental and temporal features.

Dataset: UCI Household Power Consumption
- Features: 28+ (including temperature, humidity, wind speed, pressure, dew point, visibility, time-based features, sub-metering readings, voltage, current, etc.)

- Instances: 2+ million observations (can be sampled/aggregated)

- Target Variable: Global Active Power (continuous)

Install Libraries
"""

!pip install pandas numpy scikit-learn xgboost matplotlib seaborn scipy
print("Libraries installed!")

"""Import Libraries"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

import xgboost as xgb
import lightgbm as lgb

import warnings
warnings.filterwarnings('ignore')

print("All libraries imported!")

"""Data Understanding - Load & Inspect Data"""

df = pd.read_csv('household_power_consumption.csv',nrows=50000)
df.columns = ['Date', 'Time', 'Global_active_power', 'Global_reactive_power',
'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']

df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
df = df.drop(['Date', 'Time'], axis=1)

print(f"Dataset shape: {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"\nFirst rows:\n{df.head()}")
print(f"\nBasic statistics:\n{df.describe()}")

"""Data Understanding - EDA Distributions"""

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
numeric_cols = ['Global_active_power', 'Voltage', 'Global_intensity',
                'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']

for idx, col in enumerate(numeric_cols):
    row = idx // 3
    col_idx = idx % 3
    axes[row, col_idx].hist(df[col], bins=50, alpha=0.7, edgecolor='black')
    axes[row, col_idx].set_title(f'{col}')

plt.tight_layout()
plt.show()
print(" Distribution plots created")

"""Data Understanding - Correlation Analysis"""

corr_matrix = df.select_dtypes(include=[np.number]).corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', square=True)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()
print(" Correlation matrix created")

"""Data understanding - Temporal Patterns"""

target = 'Global_active_power'

#Convert numerical columns to numeric type, coercing errors to NaN
numeric_cols = ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity',
                'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')


fig, axes = plt.subplots(2, 2, figsize=(15, 10))

#plotting how Global active power utilized on the basis of time
hourly = df.groupby(df['datetime'].dt.hour)[target].mean()
axes[0, 0].plot(hourly.index, hourly.values, marker='o', linewidth=2)
axes[0, 0].set_title('By Hour')
axes[0, 0].grid(True, alpha=0.3)

daily = df.groupby(df['datetime'].dt.day_name())[target].mean()
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
daily = daily.reindex(day_order)
axes[0, 1].bar(range(len(daily)), daily.values)
axes[0, 1].set_xticklabels(daily.index, rotation=45)
axes[0, 1].set_title('By Day')
axes[0, 1].grid(True, alpha=0.3, axis='y')

monthly = df.groupby(df['datetime'].dt.month)[target].mean()
axes[1, 0].plot(monthly.index, monthly.values, marker='s', linewidth=2)
axes[1, 0].set_title('By Month')
axes[1, 0].grid(True, alpha=0.3)

sample = df[['datetime', target]].iloc[:2000]
axes[1, 1].plot(sample['datetime'], sample[target], linewidth=0.5)
axes[1, 1].set_title('Time Series Sample')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
print(" Temporal analysis completed")

"""Data Preprocessing - Handle Missing Values"""

df_clean = df.fillna(method='ffill').fillna(method='bfill')
print(f"Missing values before: {df.isnull().sum().sum()}")
print(f"Missing values after: {df_clean.isnull().sum().sum()}")
print(" Missing values handled")

"""Data preprocessing - Feature Engineering"""

df_feat = df_clean.copy()

#Temporal features
df_feat['hour'] = df_feat['datetime'].dt.hour
df_feat['day_of_week'] = df_feat['datetime'].dt.dayofweek
df_feat['month'] = df_feat['datetime'].dt.month
df_feat['day_of_year'] = df_feat['datetime'].dt.dayofyear
df_feat['is_weekend'] = (df_feat['day_of_week'] >= 5).astype(int)

#Cyclical encoding - Since the time calculation is circular, we are utilizing cyclic endoing instead of normal enocodings
df_feat['hour_sin'] = np.sin(2 * np.pi * df_feat['hour'] / 24)
df_feat['hour_cos'] = np.cos(2 * np.pi * df_feat['hour'] / 24)

#Lag features such as 1 minute ago, 6 monutes ago and 24 minutes ago
for lag in [1, 6, 24]:
    df_feat[f'power_lag_{lag}'] = df_feat['Global_active_power'].shift(lag)

#Rolling features to capture consumption trends
for window in [6, 24]:
    df_feat[f'power_roll_mean_{window}'] = df_feat['Global_active_power'].rolling(window).mean()
    df_feat[f'power_roll_std_{window}'] = df_feat['Global_active_power'].rolling(window).std()

#Interaction - Power=Voltage*Intensity
df_feat['voltage_intensity'] = df_feat['Voltage'] * df_feat['Global_intensity']

#Remove NaN
df_feat = df_feat.dropna()

print(f"Final shape: {df_feat.shape}")
print(f"Total features: {len(df_feat.columns)}")
print(" Feature engineering completed")

"""Data preprocessing - Remove Outliers"""

def remove_outliers_iqr(data, col, threshold=1.5):
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - threshold * IQR
    upper = Q3 + threshold * IQR
    return data[(data[col] >= lower) & (data[col] <= upper)]

df_feat = remove_outliers_iqr(df_feat, 'Global_active_power')
print(f"Shape after outlier removal: {df_feat.shape}")
print(" Outliers removed")

"""Data preprocessing - Train-Test Split & Scaling"""

X = df_feat.drop(['Global_active_power', 'datetime'], axis=1)
y = df_feat['Global_active_power']

split_idx = int(len(X) * 0.8)
X_train = X[:split_idx].values
X_test = X[split_idx:].values
y_train = y[:split_idx].values
y_test = y[split_idx:].values

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print("Data split and scaled")

"""Modelling - Dimensionality reduction - PCA & K-Means Clustering"""

#PCA(Principal Component Analysis)
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_s)
X_test_pca = pca.transform(X_test_s)

print(f"Original features: {X_train_s.shape[1]}")
print(f"PCA components: {X_train_pca.shape[1]}")

#K-Means clustering
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans.fit(X_train_s)

print(f" PCA & K-Means completed")

"""Modelling - Find Optimal n_estimators for few regression models before training with data"""

print("Finding Optimal Number Of Estimators")

#Test different n_estimators values
n_estimators_range = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
rf_cv_means = []
rf_cv_stds = []
gb_cv_means = []
gb_cv_stds = []
xgb_cv_means = []
xgb_cv_stds = []

print("\nEvaluating Random Forest...")
for n_est in n_estimators_range:
    rf = RandomForestRegressor(n_estimators=n_est, max_depth=15, random_state=42, n_jobs=-1)
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(rf, X_train[:5000], y_train[:5000], cv=3, scoring='r2', n_jobs=-1)
    rf_cv_means.append(scores.mean())
    rf_cv_stds.append(scores.std())
    print(f"  n_estimators={n_est}: R² = {scores.mean():.4f} ± {scores.std():.4f}")

print("\nEvaluating Gradient Boosting...")
for n_est in n_estimators_range:
    gb = GradientBoostingRegressor(n_estimators=n_est, max_depth=5, random_state=42)
    scores = cross_val_score(gb, X_train[:5000], y_train[:5000], cv=3, scoring='r2', n_jobs=-1)
    gb_cv_means.append(scores.mean())
    gb_cv_stds.append(scores.std())
    print(f"  n_estimators={n_est}: R² = {scores.mean():.4f} ± {scores.std():.4f}")

print("\nEvaluating XGBoost...")
for n_est in n_estimators_range:
    xgb_model = xgb.XGBRegressor(n_estimators=n_est, max_depth=5, random_state=42, verbosity=0)
    scores = cross_val_score(xgb_model, X_train[:5000], y_train[:5000], cv=3, scoring='r2', n_jobs=-1)
    xgb_cv_means.append(scores.mean())
    xgb_cv_stds.append(scores.std())
    print(f"  n_estimators={n_est}: R² = {scores.mean():.4f} ± {scores.std():.4f}")

#Find optimal values - Maximum value among list
rf_opt_idx = np.argmax(rf_cv_means)
gb_opt_idx = np.argmax(gb_cv_means)
xgb_opt_idx = np.argmax(xgb_cv_means)

#Setting up the final estimators for each model
rf_opt = n_estimators_range[rf_opt_idx]
gb_opt = n_estimators_range[gb_opt_idx]
xgb_opt = n_estimators_range[xgb_opt_idx]

print(f"\n Optimal n_estimators found:")
print(f"  Random Forest: {rf_opt} (R² = {rf_cv_means[rf_opt_idx]:.4f})")
print(f"  Gradient Boosting: {gb_opt} (R² = {gb_cv_means[gb_opt_idx]:.4f})")
print(f"  XGBoost: {xgb_opt} (R² = {xgb_cv_means[xgb_opt_idx]:.4f})")

#Visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

#Random Forest
axes[0].errorbar(n_estimators_range, rf_cv_means, yerr=rf_cv_stds, c='blue', fmt='-o', capsize=5, linewidth=2)
axes[0].axvline(x=rf_opt, color='red', linestyle='--', linewidth=2, label=f'Optimal: {rf_opt}')
axes[0].set_ylabel('R² Score')
axes[0].set_xlabel('Number of Estimators')
axes[0].set_title(f'Random Forest: Optimal n_estimators = {rf_opt}')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

#Gradient Boosting
axes[1].errorbar(n_estimators_range, gb_cv_means, yerr=gb_cv_stds, c='green', fmt='-o', capsize=5, linewidth=2)
axes[1].axvline(x=gb_opt, color='red', linestyle='--', linewidth=2, label=f'Optimal: {gb_opt}')
axes[1].set_ylabel('R² Score')
axes[1].set_xlabel('Number of Estimators')
axes[1].set_title(f'Gradient Boosting: Optimal n_estimators = {gb_opt}')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

#XGBoost
axes[2].errorbar(n_estimators_range, xgb_cv_means, yerr=xgb_cv_stds, c='purple', fmt='-o', capsize=5, linewidth=2)
axes[2].axvline(x=xgb_opt, color='red', linestyle='--', linewidth=2, label=f'Optimal: {xgb_opt}')
axes[2].set_ylabel('R² Score')
axes[2].set_xlabel('Number of Estimators')
axes[2].set_title(f'XGBoost: Optimal n_estimators = {xgb_opt}')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n n_estimators optimization graphs created")

"""Modelling - Train Models
During this training 19 Models are trained and the details mentioned below:

Total Models: 19

A. Linear Models (4)
1. Linear Regression - No regularization

2. Ridge Regression (α=1.0) - L2 regularization

3. Lasso Regression (α=0.1) - L1 regularization

4. ElasticNet (α=0.1, l1_ratio=0.5) - Combined L1+L2

B. Support Vector Regression (5)
5. SVR (Linear kernel) - Linear decision boundary

6. SVR (RBF kernel) - Radial Basis Function, non-linear

7. SVR (Polynomial kernel) - Degree=3 polynomial mapping

8. SVR (Sigmoid kernel) - Neural network-like kernel

9. SVR (Best kernel, Tuned) - Hyperparameter optimized best performer

C. Tree-Based Models (6)
Decision Tree - max_depth=10

10. Random Forest - n_estimators=50 (optimized)

11. Gradient Boosting - n_estimators=50 (optimized)

12. XGBoost - n_estimators=100 (optimized)

13. LightGBM - n_estimators=100

14. Tuned variants - RF, GB, XGB with optimized hyperparameters (3 models)

D. Distance-Based Model (1)
15. K-Nearest Neighbors (k=5) - Instance-based learning

E. Neural Network (1)
16. Multi-Layer Perceptron

"""

import time
from sklearn.utils import resample

print("Training Models")

results = {}
training_times = {}

#1. Linear Regression
print("\n1. Linear Regression...")
start = time.time()
lr = LinearRegression().fit(X_train_s, y_train)
training_times['Linear Regression'] = time.time() - start
results['Linear Regression'] = r2_score(y_test, lr.predict(X_test_s))
print(f"   R² = {results['Linear Regression']:.4f} | Time: {training_times['Linear Regression']:.2f}s")

#2. Ridge
print("2. Ridge...")
ridge = Ridge(alpha=1.0).fit(X_train_s, y_train)
results['Ridge'] = r2_score(y_test, ridge.predict(X_test_s))
print(f"   R² = {results['Ridge']:.4f}")

#3. Lasso
print("3. Lasso...")
lasso = Lasso(alpha=0.1, max_iter=1000).fit(X_train_s, y_train)
results['Lasso'] = r2_score(y_test, lasso.predict(X_test_s))
print(f"   R² = {results['Lasso']:.4f}")

#4. ElasticNet
print("4. ElasticNet...")
enet = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=1000).fit(X_train_s, y_train)
results['ElasticNet'] = r2_score(y_test, enet.predict(X_test_s))
print(f"   R² = {results['ElasticNet']:.4f}")

#5. SVR WITH MULTIPLE KERNELS
print("5. Support Vector Regression (Multiple Kernels)")

svr_kernels = ['linear', 'rbf', 'poly', 'sigmoid']
svr_models = {}

#Due to huge computation time, resampled the data to 8k data for SVR
X_svr, y_svr = resample(X_train_s, y_train, n_samples=8000, random_state=42)

for kernel in svr_kernels:
    print(f"\n   Training SVR ({kernel})...")
    start = time.time()

    if kernel == 'poly':
        svr = SVR(kernel=kernel, degree=3, gamma='scale')
    else:
        svr = SVR(kernel=kernel, gamma='scale')

    svr.fit(X_svr, y_svr)
    svr_models[kernel] = svr

    time_taken = time.time() - start
    r2 = r2_score(y_test, svr.predict(X_test_s))
    results[f'SVR ({kernel})'] = r2

    print(f"R² = {r2:.4f} | Time: {time_taken:.2f}s")

#Find best SVR kernel
best_svr_kernel = max(svr_kernels, key=lambda k: results[f'SVR ({k})'])
best_svr = svr_models[best_svr_kernel]
print(f"\n Best SVR kernel: {best_svr_kernel}")

#6. Decision Tree
print("\n6. Decision Tree...")
dt = DecisionTreeRegressor(max_depth=10, random_state=42).fit(X_train, y_train)
results['Decision Tree'] = r2_score(y_test, dt.predict(X_test))
print(f"R² = {results['Decision Tree']:.4f}")

#7. Random Forest (Using Optimal n_estimators)
print(f"\n7. Random Forest (n_estimators={rf_opt})...")
start = time.time()
rf = RandomForestRegressor(n_estimators=rf_opt, max_depth=15, max_features='sqrt', random_state=42, n_jobs=-1).fit(X_train, y_train)
training_times['Random Forest'] = time.time() - start
results['Random Forest'] = r2_score(y_test, rf.predict(X_test))
print(f"R² = {results['Random Forest']:.4f} | Time: {training_times['Random Forest']:.2f}s")

#8. Gradient Boosting (Using Optimal n_estimators)
print(f"\n8. Gradient Boosting (n_estimators={gb_opt})...")
start = time.time()
gbr = GradientBoostingRegressor(n_estimators=gb_opt, learning_rate=0.1, max_depth=5, subsample=0.8, random_state=42).fit(X_train, y_train)
training_times['Gradient Boosting'] = time.time() - start
results['Gradient Boosting'] = r2_score(y_test, gbr.predict(X_test))
print(f"R² = {results['Gradient Boosting']:.4f} | Time: {training_times['Gradient Boosting']:.2f}s")

#9. XGBoost (Using Optimal n_estimators)
print(f"\n9. XGBoost (n_estimators={xgb_opt})...")
start = time.time()
xgb_m = xgb.XGBRegressor(n_estimators=xgb_opt, learning_rate=0.1, max_depth=5,
                         tree_method='hist', random_state=42, verbosity=0).fit(X_train, y_train)
training_times['XGBoost'] = time.time() - start
results['XGBoost'] = r2_score(y_test, xgb_m.predict(X_test))
print(f"R² = {results['XGBoost']:.4f} | Time: {training_times['XGBoost']:.2f}s")

#10. LightGBM
print("\n10. LightGBM...")
start = time.time()
lgb_m = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=5,
                          random_state=42, verbose=-1).fit(X_train, y_train)
training_times['LightGBM'] = time.time() - start
results['LightGBM'] = r2_score(y_test, lgb_m.predict(X_test))
print(f"R² = {results['LightGBM']:.4f} | Time: {training_times['LightGBM']:.2f}s")

#11. Neural Network
print("\n11. Neural Network...")
start = time.time()
X_nn, y_nn = resample(X_train_s, y_train, n_samples=10000, random_state=42)
mlp = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=200, early_stopping=True,validation_fraction=0.1, random_state=42, verbose=False).fit(X_nn, y_nn)
training_times['Neural Network'] = time.time() - start
results['Neural Network'] = r2_score(y_test, mlp.predict(X_test_s))
print(f"R² = {results['Neural Network']:.4f} | Time: {training_times['Neural Network']:.2f}s")

#12. KNN
print("\n12. KNN...")
start = time.time()
X_knn, y_knn = resample(X_train_s, y_train, n_samples=10000, random_state=42)
knn = KNeighborsRegressor(n_neighbors=5, weights='distance', n_jobs=-1).fit(X_knn, y_knn)
training_times['KNN'] = time.time() - start
results['KNN'] = r2_score(y_test, knn.predict(X_test_s))
print(f"R² = {results['KNN']:.4f} | Time: {training_times['KNN']:.2f}s")

#Summary of Results
print("\nMODEL TRAINING SUMMARY")

sorted_results = sorted(results.items(), key=lambda x: x, reverse=True)
print("\nModel Performance (sorted by R²):")
for idx, (model, score) in enumerate(sorted_results, 1):
    print(f"  {idx:2d}. {model:25s}: R² = {score:.4f}")

total_train_time = sum(training_times.values())
print(f"\nTotal training time: {total_train_time:.1f}s ({total_train_time/60:.1f} min)")
print("All models trained!")

"""Modelling - Hyperparameter Tuning To Improve the Performance"""

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

print("Hyper Parameter Tuning")

#Use 20% sample for tuning
print("\nPreparing tuning sample (20% for speed)...")
sample_size = int(len(X_train) * 0.2)
X_tune, y_tune = resample(X_train, y_train, n_samples=sample_size, random_state=42)
X_tune_s = X_train_s[:sample_size]

print(f"Tuning on: {len(X_tune):,} samples\n")

#Tune SVR With Multiple Kernels
print("1. Tuning SVR Kernels...")

svr_tune_results = {}
X_svr_tune, y_svr_tune = resample(X_tune_s, y_tune, n_samples=5000, random_state=42)

for kernel in ['linear', 'rbf', 'poly', 'sigmoid']:
    start = time.time()

    if kernel == 'poly':
        param_dist = {
            'C': uniform(0.1, 100),
            'degree': randint(2, 4),
            'gamma': ['scale', 'auto']
        }
    else:
        param_dist = {
            'C': uniform(0.1, 100),
            'gamma': ['scale', 'auto']
        }

    rs_svr = RandomizedSearchCV(
        SVR(kernel=kernel),
        param_dist,
        n_iter=5,
        cv=2,
        scoring='r2',
        n_jobs=-1,
        random_state=42,
        verbose=0
    )

    rs_svr.fit(X_svr_tune, y_svr_tune)
    time_svr = time.time() - start

    r2_svr = r2_score(y_test, rs_svr.best_estimator_.predict(X_test_s))
    svr_tune_results[kernel] = {
        'model': rs_svr.best_estimator_,
        'r2': r2_svr,
        'params': rs_svr.best_params_,
        'time': time_svr
    }

    print(f"SVR ({kernel:8s}): R² = {r2_svr:.4f} | Params: {rs_svr.best_params_} | Time: {time_svr:.1f}s")

#Find best SVR
best_svr_kernel = max(svr_tune_results, key=lambda k: svr_tune_results[k]['r2'])
best_svr_tuned = svr_tune_results[best_svr_kernel]['model']
print(f"\n  Best SVR kernel: {best_svr_kernel} (R² = {svr_tune_results[best_svr_kernel]['r2']:.4f})")

#Tune Random Forest
print("\n2. Tuning Random Forest...")
start_rf = time.time()
rs_rf = RandomizedSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    {'n_estimators': randint(rf_opt-10, rf_opt+10), 'max_depth': randint(10, 16)},
    n_iter=5, cv=2, scoring='r2', n_jobs=-1, random_state=42, verbose=0
)
rs_rf.fit(X_tune, y_tune)
time_rf = time.time() - start_rf
best_rf_tuned = rs_rf.best_estimator_
r2_rf = r2_score(y_test, best_rf_tuned.predict(X_test))
print(f" Best params: {rs_rf.best_params_}")
print(f" Test R²: {r2_rf:.4f} | Time: {time_rf:.1f}s")

#Tune XGBOOST
print("\n3. Tuning XGBoost...")
start_xgb = time.time()
rs_xgb = RandomizedSearchCV(
    xgb.XGBRegressor(random_state=42, verbosity=0),
    {'n_estimators': randint(xgb_opt-10, xgb_opt+10), 'learning_rate': uniform(0.05, 0.1),
     'max_depth': randint(4, 7)},
    n_iter=6, cv=2, scoring='r2', n_jobs=-1, random_state=42, verbose=0
)
rs_xgb.fit(X_tune, y_tune)
time_xgb = time.time() - start_xgb
best_xgb_tuned = rs_xgb.best_estimator_
r2_xgb = r2_score(y_test, best_xgb_tuned.predict(X_test))
print(f" Best params: {rs_xgb.best_params_}")
print(f" Test R²: {r2_xgb:.4f} | Time: {time_xgb:.1f}s")

#Tune LIGHTGBM
print("\n4. Tuning LightGBM...")
start_lgb = time.time()
rs_lgb = RandomizedSearchCV(
    lgb.LGBMRegressor(random_state=42, verbose=-1),
    {'n_estimators': randint(80, 120), 'learning_rate': uniform(0.05, 0.1),
     'max_depth': randint(4, 7)},
    n_iter=5, cv=2, scoring='r2', n_jobs=-1, random_state=42, verbose=0
)
rs_lgb.fit(X_tune, y_tune)
time_lgb = time.time() - start_lgb
best_lgb_tuned = rs_lgb.best_estimator_
r2_lgb = r2_score(y_test, best_lgb_tuned.predict(X_test))
print(f" Best params: {rs_lgb.best_params_}")
print(f" Test R²: {r2_lgb:.4f} | Time: {time_lgb:.1f}s")

#Summary
print("TUNING SUMMARY")
tuning_df = pd.DataFrame({
    'Model': [f'SVR ({best_svr_kernel})', 'Random Forest', 'XGBoost', 'LightGBM'],
    'Before R²': [
        results[f'SVR ({best_svr_kernel})'],
        results['Random Forest'],
        results['XGBoost'],
        results['LightGBM']
    ],
    'After R²': [r2_svr, r2_rf, r2_xgb, r2_lgb],
    'Improvement': [
        r2_svr - results[f'SVR ({best_svr_kernel})'],
        r2_rf - results['Random Forest'],
        r2_xgb - results['XGBoost'],
        r2_lgb - results['LightGBM']
    ],
    'Time (s)': [svr_tune_results[best_svr_kernel]['time'], time_rf, time_xgb, time_lgb]
})

print("\n" + tuning_df.to_string(index=False))

total_time = sum(tuning_df['Time (s)'].values)
print(f"\n Total tuning time: {total_time:.1f}s ({total_time/60:.1f} min)")

#Update results
results[f'SVR ({best_svr_kernel}) - Tuned'] = r2_svr
results['RF (Tuned)'] = r2_rf
results['XGB (Tuned)'] = r2_xgb
results['LGB (Tuned)'] = r2_lgb

print(" Hyperparameter tuning completed!")

"""Modelling - Ensemble Methods - Stacking"""

print("Ensemble Methods - Stacking")

print("\nCreating Stacking Ensemble with tuned models...")

base_models = [
    ('svr_tuned', best_svr_tuned),
    ('rf_tuned', best_rf_tuned),
    ('xgb_tuned', best_xgb_tuned),
    ('ridge', ridge)
]

meta_model = Ridge(alpha=1.0)

stacking = StackingRegressor(
    estimators=base_models,
    final_estimator=meta_model,
    cv=2
)

print("Training stacking ensemble...")
stacking.fit(X_train_s, y_train)
y_pred_stacking = stacking.predict(X_test_s)

results['Stacking Ensemble'] = r2_score(y_test, y_pred_stacking)
rmse_stacking = np.sqrt(mean_squared_error(y_test, y_pred_stacking))

print(f"\nStacking Ensemble Results:")
print(f" R² Score: {results['Stacking Ensemble']:.4f}")
print(f" RMSE: {rmse_stacking:.4f} kW")

print("\n Ensemble methods completed!")

"""Evaluation - Model Comparison"""

print("Final Model Comparison")

comparison_data = []

#Prepare model predictions dict
model_predictions = {
    'Linear Regression': (lr, X_test_s),
    'Ridge': (ridge, X_test_s),
    'Lasso': (lasso, X_test_s),
    'ElasticNet': (enet, X_test_s),
    'Decision Tree': (dt, X_test),
    'Random Forest': (rf, X_test),
    'Gradient Boosting': (gbr, X_test),
    'XGBoost': (xgb_m, X_test),
    'LightGBM': (lgb_m, X_test),
    'Neural Network': (mlp, X_test_s),
    'KNN': (knn, X_test_s),
    f'SVR (Linear)': (svr_models['linear'], X_test_s),
    f'SVR (RBF)': (svr_models['rbf'], X_test_s),
    f'SVR (Poly)': (svr_models['poly'], X_test_s),
    f'SVR (Sigmoid)': (svr_models['sigmoid'], X_test_s),
    f'SVR ({best_svr_kernel}) - Tuned': (best_svr_tuned, X_test_s),
    'RF (Tuned)': (best_rf_tuned, X_test),
    'XGB (Tuned)': (best_xgb_tuned, X_test),
    'LGB (Tuned)': (best_lgb_tuned, X_test),
    'Stacking Ensemble': (stacking, X_test_s)
}

for model_name, (model, X_data) in model_predictions.items():
    y_pred = model.predict(X_data)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100

    comparison_data.append({
        'Model': model_name,
        'R²': r2,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE (%)': mape
    })

comparison_df = pd.DataFrame(comparison_data).sort_values('R²', ascending=False)

print("\nAll Models Performance (sorted by R²):")
print(comparison_df.to_string(index=False))

best_row = comparison_df.iloc[0]
best_model_name = best_row['Model']
best_r2 = best_row['R²']

print(f" BEST MODEL: {best_model_name}")
print(f"  R² Score: {best_r2:.4f}")
print(f"  RMSE: {best_row['RMSE']:.4f} kW")
print(f"  MAE: {best_row['MAE']:.4f} kW")
print(f"  MAPE: {best_row['MAPE (%)']:.2f}%")

"""Evaluation - Visualization of Model Comparison"""

#Plot top 10 models
top_10 = comparison_df.head(10)

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

#R² Scores
axes[0, 0].barh(top_10['Model'], top_10['R²'], color='steelblue', alpha=0.8)
axes[0, 0].set_xlabel('R² Score')
axes[0, 0].set_title('Top 10 Models: R² Score')
axes[0, 0].grid(True, alpha=0.3, axis='x')

#RMSE
axes[0, 1].barh(top_10['Model'], top_10['RMSE'], color='coral', alpha=0.8)
axes[0, 1].set_xlabel('RMSE (kW)')
axes[0, 1].set_title('Top 10 Models: RMSE')
axes[0, 1].grid(True, alpha=0.3, axis='x')

#MAE
axes[1, 0].barh(top_10['Model'], top_10['MAE'], color='lightgreen', alpha=0.8)
axes[1, 0].set_xlabel('MAE (kW)')
axes[1, 0].set_title('Top 10 Models: MAE')
axes[1, 0].grid(True, alpha=0.3, axis='x')

#MAPE
axes[1, 1].barh(top_10['Model'], top_10['MAPE (%)'], color='mediumpurple', alpha=0.8)
axes[1, 1].set_xlabel('MAPE (%)')
axes[1, 1].set_title('Top 10 Models: MAPE')
axes[1, 1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.show()

print("Model comparison visualization created")

"""Evaluation - Overfitting Analysis"""

print("Overfitting Analysis")

overfit_data = []

#Define all models with their scaled/unscaled data requirements
model_dict = {
    'Linear Regression': (lr, True),
    'Ridge': (ridge, True),
    'Lasso': (lasso, True),
    'ElasticNet': (enet, True),
    'SVR (linear)': (svr_models['linear'], True),
    'SVR (rbf)': (svr_models['rbf'], True),
    'SVR (poly)': (svr_models['poly'], True),
    'SVR (sigmoid)': (svr_models['sigmoid'], True),
    f'SVR ({best_svr_kernel}) - Tuned': (best_svr_tuned, True),
    'Neural Network': (mlp, True),
    'KNN': (knn, True),
    'Decision Tree': (dt, False),
    'Random Forest': (rf, False),
    'Gradient Boosting': (gbr, False),
    'XGBoost': (xgb_m, False),
    'LightGBM': (lgb_m, False),
    'RF (Tuned)': (best_rf_tuned, False),
    'XGB (Tuned)': (best_xgb_tuned, False),
    'LGB (Tuned)': (best_lgb_tuned, False),
    'Stacking Ensemble': (stacking, True)
}

print("\nCalculating train vs test performance for all models...\n")

for model_name, (model, is_scaled) in model_dict.items():
    X_tr = X_train_s if is_scaled else X_train
    X_te = X_test_s if is_scaled else X_test

    #Handle sampled models (SVR, KNN, Neural Network)
    if model_name in ['SVR (linear)', 'SVR (rbf)', 'SVR (poly)', 'SVR (sigmoid)',
                      f'SVR ({best_svr_kernel}) - Tuned']:
        #Use sample for training prediction
        X_tr_sample = X_svr if model_name != f'SVR ({best_svr_kernel}) - Tuned' else X_svr_tune
        y_tr_sample = y_svr if model_name != f'SVR ({best_svr_kernel}) - Tuned' else y_svr_tune
        y_train_pred = model.predict(X_tr_sample)
        train_r2 = r2_score(y_tr_sample, y_train_pred)
    elif model_name == 'KNN':
        y_train_pred = model.predict(X_knn)
        train_r2 = r2_score(y_knn, y_train_pred)
    elif model_name == 'Neural Network':
        y_train_pred = model.predict(X_nn)
        train_r2 = r2_score(y_nn, y_train_pred)
    else:
        y_train_pred = model.predict(X_tr)
        train_r2 = r2_score(y_train, y_train_pred)

    y_test_pred = model.predict(X_te)
    test_r2 = r2_score(y_test, y_test_pred)
    gap = train_r2 - test_r2

    overfit_data.append({
        'Model': model_name,
        'Train R²': train_r2,
        'Test R²': test_r2,
        'Gap': gap,
        'Status': 'Overfitting' if gap > 0.1 else 'Good Fit' if gap > 0.05 else 'Excellent'
    })

    print(f"  {model_name:30s}: Train R² = {train_r2:.4f}, Test R² = {test_r2:.4f}, Gap = {gap:.4f}")

overfit_df = pd.DataFrame(overfit_data).sort_values('Gap', ascending=False)

print("Overfitting Summary (sorted by gap)")
print("\n" + overfit_df.to_string(index=False))

#Identify overfitting models
overfit_models = overfit_df[overfit_df['Gap'] > 0.1]
print(f"\n Models with potential overfitting (Gap > 0.1): {len(overfit_models)}")
if len(overfit_models) > 0:
    print(overfit_models[['Model', 'Gap']].to_string(index=False))

#Visualization
plt.figure(figsize=(16, 8))
x = np.arange(len(overfit_df))
width = 0.35

plt.bar(x - width/2, overfit_df['Train R²'], width, label='Train R²', alpha=0.8, color='steelblue')
plt.bar(x + width/2, overfit_df['Test R²'], width, label='Test R²', alpha=0.8, color='coral')

plt.xlabel('Model', fontsize=12)
plt.ylabel('R² Score', fontsize=12)
plt.title('Overfitting Analysis: Train vs Test R² Scores', fontsize=14, fontweight='bold')
plt.xticks(x, overfit_df['Model'], rotation=45, ha='right')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3, axis='y')
plt.axhline(y=0.9, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Excellent (0.9)')
plt.tight_layout()
plt.savefig('09_overfitting_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

#Gap visualization
plt.figure(figsize=(14, 6))
colors = ['red' if gap > 0.1 else 'orange' if gap > 0.05 else 'green' for gap in overfit_df['Gap']]
plt.barh(overfit_df['Model'], overfit_df['Gap'], color=colors, alpha=0.7)
plt.xlabel('Overfitting Gap (Train R² - Test R²)', fontsize=12)
plt.title('Overfitting Gap by Model', fontsize=14, fontweight='bold')
plt.axvline(x=0.1, color='red', linestyle='--', linewidth=2, label='Overfitting Threshold (0.1)')
plt.axvline(x=0.05, color='orange', linestyle='--', linewidth=2, label='Warning Zone (0.05)')
plt.legend()
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.show()

print("\n Overfitting analysis completed")

"""Results - Feature Importance"""

print("Feature Importance Analysis")

#Get feature names
feature_names = X.columns.tolist()

#1. RANDOM FOREST FEATURE IMPORTANCE
print("\n1. Random Forest (Tuned) Feature Importance")
rf_importance = best_rf_tuned.feature_importances_
rf_imp_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': rf_importance
}).sort_values('Importance', ascending=False)

print("\nTop 15 Most Important Features (Random Forest):")
print(rf_imp_df.head(15).to_string(index=False))


#2. XGBOOST FEATURE IMPORTANCE
print("\n2. XGBoost (Tuned) Feature Importance")
xgb_importance = best_xgb_tuned.feature_importances_
xgb_imp_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': xgb_importance
}).sort_values('Importance', ascending=False)

print("\nTop 15 Most Important Features (XGBoost):")
print(xgb_imp_df.head(15).to_string(index=False))


#3. LIGHTGBM FEATURE IMPORTANCE
print("\n3. LightGBM (Tuned) Feature Importance")
lgb_importance = best_lgb_tuned.feature_importances_
lgb_imp_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': lgb_importance
}).sort_values('Importance', ascending=False)

print("\nTop 15 Most Important Features (LightGBM):")
print(lgb_imp_df.head(15).to_string(index=False))

#4. AGGREGATE FEATURE IMPORTANCE
print("\n4. Aggregate Feature Importance (Average across all 3 models)")

#Normalize importances
rf_norm = rf_importance / rf_importance.sum()
xgb_norm = xgb_importance / xgb_importance.sum()
lgb_norm = lgb_importance / lgb_importance.sum()

#Average importance
avg_importance = (rf_norm + xgb_norm + lgb_norm) / 3
agg_imp_df = pd.DataFrame({
    'Feature': feature_names,
    'RF Importance': rf_norm,
    'XGB Importance': xgb_norm,
    'LGB Importance': lgb_norm,
    'Average Importance': avg_importance
}).sort_values('Average Importance', ascending=False)

print("\nTop 20 Features (Aggregated):")
print(agg_imp_df.head(20)[['Feature', 'Average Importance']].to_string(index=False))

#VISUALIZATION
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

#Random Forest
top_rf = rf_imp_df.head(15)
axes[0, 0].barh(top_rf['Feature'], top_rf['Importance'], color='steelblue', alpha=0.8)
axes[0, 0].set_xlabel('Importance', fontsize=11)
axes[0, 0].set_title('Top 15 Features: Random Forest', fontsize=12, fontweight='bold')
axes[0, 0].invert_yaxis()
axes[0, 0].grid(True, alpha=0.3, axis='x')

#XGBoost
top_xgb = xgb_imp_df.head(15)
axes[0, 1].barh(top_xgb['Feature'], top_xgb['Importance'], color='coral', alpha=0.8)
axes[0, 1].set_xlabel('Importance', fontsize=11)
axes[0, 1].set_title('Top 15 Features: XGBoost', fontsize=12, fontweight='bold')
axes[0, 1].invert_yaxis()
axes[0, 1].grid(True, alpha=0.3, axis='x')

#LightGBM
top_lgb = lgb_imp_df.head(15)
axes[1, 0].barh(top_lgb['Feature'], top_lgb['Importance'], color='lightgreen', alpha=0.8)
axes[1, 0].set_xlabel('Importance', fontsize=11)
axes[1, 0].set_title('Top 15 Features: LightGBM', fontsize=12, fontweight='bold')
axes[1, 0].invert_yaxis()
axes[1, 0].grid(True, alpha=0.3, axis='x')

#Aggregate
top_agg = agg_imp_df.head(15)
axes[1, 1].barh(top_agg['Feature'], top_agg['Average Importance'], color='mediumpurple', alpha=0.8)
axes[1, 1].set_xlabel('Average Importance', fontsize=11)
axes[1, 1].set_title('Top 15 Features: Aggregate (Average)', fontsize=12, fontweight='bold')
axes[1, 1].invert_yaxis()
axes[1, 1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.show()

#FEATURE IMPORTANCE COMPARISON
#Compare top 10 features across models
top_10_features = agg_imp_df.head(10)['Feature'].values
comparison_data = []
for feat in top_10_features:
    comparison_data.append({
        'Feature': feat,
        'RF': rf_imp_df[rf_imp_df['Feature'] == feat]['Importance'].values[0],
        'XGB': xgb_imp_df[xgb_imp_df['Feature'] == feat]['Importance'].values[0],
        'LGB': lgb_imp_df[lgb_imp_df['Feature'] == feat]['Importance'].values[0]
    })

comp_df = pd.DataFrame(comparison_data)

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(comp_df))
width = 0.25

ax.bar(x - width, comp_df['RF'], width, label='Random Forest', alpha=0.8)
ax.bar(x, comp_df['XGB'], width, label='XGBoost', alpha=0.8)
ax.bar(x + width, comp_df['LGB'], width, label='LightGBM', alpha=0.8)

ax.set_xlabel('Feature', fontsize=12)
ax.set_ylabel('Importance', fontsize=12)
ax.set_title('Top 10 Features: Importance Comparison Across Models', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(comp_df['Feature'], rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

print("\n Feature importance analysis completed")

"""Results - Predictions Visualization"""

print("Prediction Visualization")

#Use best model from comparison
best_model = comparison_df.iloc[0]['Model']
print(f"\nVisualizing predictions for: {best_model}")

#Get predictions from best model
if best_model in ['Linear Regression', 'Ridge', 'Lasso', 'ElasticNet',
                  'SVR (linear)', 'SVR (rbf)', 'SVR (poly)', 'SVR (sigmoid)',
                  f'SVR ({best_svr_kernel}) - Tuned', 'Neural Network', 'KNN',
                  'Stacking Ensemble']:
    y_pred_best = model_predictions[best_model][0].predict(X_test_s)
else:
    y_pred_best = model_predictions[best_model][0].predict(X_test)

#Calculate metrics
r2_best = r2_score(y_test, y_pred_best)
rmse_best = np.sqrt(mean_squared_error(y_test, y_pred_best))
mae_best = mean_absolute_error(y_test, y_pred_best)
residuals = y_test - y_pred_best

# Visaulizations
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

#1. Actual vs Predicted Scatter Plot
ax1 = fig.add_subplot(gs[0, 0]) # Changed from gs[0, :2] to gs[0, 0]
ax1.scatter(y_test, y_pred_best, alpha=0.4, s=20, c='steelblue', edgecolors='none')
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         'r--', lw=3, label='Perfect Prediction')
ax1.set_xlabel('Actual Power Consumption (kW)', fontsize=12)
ax1.set_ylabel('Predicted Power Consumption (kW)', fontsize=12)
ax1.set_title(f'{best_model}: Actual vs Predicted (R² = {r2_best:.4f})',
              fontsize=13, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

#2. Metrics Box
ax2 = fig.add_subplot(gs[0, 1]) # Changed from gs[0, 2] to gs[0, 1]
ax2.axis('off')
metrics_text = f"""
MODEL PERFORMANCE

Model: {best_model}

Metrics:
├─ R² Score: {r2_best:.4f}
├─ RMSE: {rmse_best:.4f} kW
├─ MAE: {mae_best:.4f} kW
└─ MAPE: {mean_absolute_percentage_error(y_test, y_pred_best)*100:.2f}%

Sample Statistics:
├─ Test samples: {len(y_test):,}
├─ Mean actual: {y_test.mean():.4f} kW
└─ Mean predicted: {y_pred_best.mean():.4f} kW
"""
ax2.text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
         verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

#3. Time Series of Predictions (sample)
ax6 = fig.add_subplot(gs[1, :])
sample_size = min(500, len(y_test))
sample_indices = np.arange(sample_size)
ax6.plot(sample_indices, y_test[:sample_size], label='Actual', linewidth=2, alpha=0.7, color='blue')
ax6.plot(sample_indices, y_pred_best[:sample_size], label='Predicted', linewidth=2, alpha=0.7, color='red')
ax6.fill_between(sample_indices, y_test[:sample_size], y_pred_best[:sample_size],
                  alpha=0.2, color='gray', label='Error')
ax6.set_xlabel('Sample Index', fontsize=12)
ax6.set_ylabel('Power Consumption (kW)', fontsize=12)
ax6.set_title(f'Time Series: Actual vs Predicted (First {sample_size} samples)',
              fontsize=13, fontweight='bold')
ax6.legend(fontsize=11)
ax6.grid(True, alpha=0.3)

plt.show()

# Error Analysis
print("\nError Analysis:")
abs_errors = np.abs(residuals)
print(f"  Mean Absolute Error: {mae_best:.4f} kW")
print(f"  Median Absolute Error: {np.median(abs_errors):.4f} kW")
print(f"  Max Absolute Error: {abs_errors.max():.4f} kW")
print(f"  Min Absolute Error: {abs_errors.min():.4f} kW")
print(f"  Std of Errors: {residuals.std():.4f} kW")

# Percentage of predictions within tolerance
tolerance_levels = [0.1, 0.2, 0.5, 1.0]
print(f"\nPredictions within tolerance:")
for tol in tolerance_levels:
    within_tol = (abs_errors <= tol).sum()
    pct = (within_tol / len(abs_errors)) * 100
    print(f"  ±{tol} kW: {within_tol:,} samples ({pct:.1f}%)")

print("\n Prediction visualization completed")

"""Testing - Prediction on New Data"""

print("Prediction On New Data")

#Select random samples for prediction
n_predictions = 5
random_indices = np.random.randint(0, len(X_test), n_predictions)

print(f"\nMaking predictions on {n_predictions} random samples from test set...")

#Get best model
best_model_obj, best_model_X = model_predictions[best_model]

prediction_results = []

for i, idx in enumerate(random_indices, 1):
    print(f"Sample {i}/{n_predictions} (Index: {idx})")

    #Get input data
    if best_model in ['Linear Regression', 'Ridge', 'Lasso', 'ElasticNet',
                      'SVR (linear)', 'SVR (rbf)', 'SVR (poly)', 'SVR (sigmoid)',
                      f'SVR ({best_svr_kernel}) - Tuned', 'Neural Network', 'KNN',
                      'Stacking Ensemble']:
        input_data = X_test_s[idx:idx+1]
    else:
        input_data = X_test[idx:idx+1]

    #Make prediction
    prediction = best_model_obj.predict(input_data)[0]
    actual = y_test[idx]
    error = abs(prediction - actual)
    error_pct = (error / actual) * 100

    #Display results
    print(f"\nModel: {best_model}")
    print(f"\nInput Features (Top 5):")
    for j, feat in enumerate(feature_names[:5]):
        print(f"  {feat:25s}: {X_test[idx, j]:.4f}")
    print(f"  ... ({len(feature_names)} total features)")

    print(f"\nPrediction Results:")
    print(f"  Predicted: {prediction:.4f} kW")
    print(f"  Actual:    {actual:.4f} kW")
    print(f"  Error:     {error:.4f} kW ({error_pct:.2f}%)")

    status = " Excellent" if error_pct < 5 else "Good" if error_pct < 10 else "Fair" if error_pct < 20 else "Poor"
    print(f"  Status:    {status}")

    prediction_results.append({
        'Sample': i,
        'Actual (kW)': actual,
        'Predicted (kW)': prediction,
        'Error (kW)': error,
        'Error (%)': error_pct,
        'Status': status
    })

#Summary table
print("Prediction Summary")

pred_df = pd.DataFrame(prediction_results)
print("\n" + pred_df.to_string(index=False))

print(f"\nAverage Error: {pred_df['Error (kW)'].mean():.4f} kW ({pred_df['Error (%)'].mean():.2f}%)")
print(f"Max Error: {pred_df['Error (kW)'].max():.4f} kW ({pred_df['Error (%)'].max():.2f}%)")
print(f"Min Error: {pred_df['Error (kW)'].min():.4f} kW ({pred_df['Error (%)'].min():.2f}%)")

#Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

#Bar chart
x_pos = np.arange(len(pred_df))
axes[0].bar(x_pos - 0.2, pred_df['Actual (kW)'], 0.4, label='Actual', alpha=0.8, color='steelblue')
axes[0].bar(x_pos + 0.2, pred_df['Predicted (kW)'], 0.4, label='Predicted', alpha=0.8, color='coral')
axes[0].set_xlabel('Sample', fontsize=12)
axes[0].set_ylabel('Power Consumption (kW)', fontsize=12)
axes[0].set_title('Actual vs Predicted: Sample Predictions', fontsize=13, fontweight='bold')
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels([f'Sample {i+1}' for i in range(len(pred_df))])
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')

#Error percentage
colors = ['green' if e < 5 else 'orange' if e < 10 else 'red' for e in pred_df['Error (%)']]
axes[1].bar(x_pos, pred_df['Error (%)'], color=colors, alpha=0.7)
axes[1].axhline(y=5, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Excellent (<5%)')
axes[1].axhline(y=10, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='Good (<10%)')
axes[1].set_xlabel('Sample', fontsize=12)
axes[1].set_ylabel('Error (%)', fontsize=12)
axes[1].set_title('Prediction Error Percentage', fontsize=13, fontweight='bold')
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels([f'Sample {i+1}' for i in range(len(pred_df))])
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print("\n Prediction on new data completed")

"""Deployment Recommendation : Gradient Boosting"""