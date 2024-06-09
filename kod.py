import numpy as np
import pandas as pd 
import scipy
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.linear_model import ElasticNetCV
import seaborn as sns
from sklearn.model_selection import KFold
import itertools
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
import warnings
from sklearn.model_selection import ParameterGrid
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Wczytaj plik .csv do dataframe
x_test = pd.read_csv('/kaggle/input/sad-2024-grupa-1/X_test.csv/X_test.csv')
x_train = pd.read_csv('/kaggle/input/sad-2024-grupa-1/X_train.csv/X_train.csv')
y_train = pd.read_csv('/kaggle/input/sad-2024-grupa-1/y_train.csv')


# Zadanie 1

for col in x_train.columns:
    if not x_train[col].dtype.kind in ['i', 'f']: 
        print("Zmienna", col, "nie jest typu numeric.")

for col in y_train.columns:
    if not y_train[col].dtype.kind in ['i', 'f']:
        print("Zmienna", col, "nie jest typu numeric.")

for col in x_test.columns:
    if not x_test[col].dtype.kind in ['i', 'f']:
        print("Zmienna", col, "nie jest typu numeric.")

complete_x = x_train.dropna().shape[0] == x_train.shape[0]
complete_y = y_train.dropna().shape[0] == y_train.shape[0]
complete_test = x_test.dropna().shape[0] == x_test.shape[0]

print("Dane treningowe x_train są kompletne:", complete_x)
print("Dane treningowe y_train są kompletne:", complete_y)
print("Dane testowe są kompletne:", complete_test)

x_obs = x_train.shape[0]
x_vars = x_train.shape[1]
print("Liczba obserwacji w x_train:", x_obs)
print("Liczba zmiennych w x_train:", x_vars)

y_obs = y_train.shape[0]
y_vars = y_train.shape[1]
print("Liczba obserwacji w y_train:", y_obs)
print("Liczba zmiennych w y_train", y_vars)

test_obs = x_test.shape[0]
test_vars = x_test.shape[1]
print("Liczba obserwacji w x_test:", test_obs)
print("Liczba zmiennych w x_test:", test_vars)

meanvals = y_train.mean()
varvals = y_train.var()
medvals = y_train.median()
quantile25 = y_train.quantile(0.25)
quantile75 = y_train.quantile(0.75)

res = pd.DataFrame({
    'Mean': meanvals,
    'Variance': varvals,
    'Median': medvals,
    'Quantile_25': quantile25,
    'Quantile_75': quantile75
})

print(res)

densityEstim = gaussian_kde(y_train["CD36"])
x = np.linspace(min(y_train["CD36"]), max(y_train["CD36"]), 1000)
plt.plot(x, densityEstim(x))
plt.title("Estymator gestosci")
plt.xlabel('Wartosc')
plt.ylabel('Gestosc')
plt.show()

data = y_train["CD36"]
plt.hist(data, bins=30, density=True, alpha=0.5, color='b')
plt.title('Histogram')
plt.xlabel('Wartosc')
plt.ylabel('Liczba obserwacji')
plt.show()

variable = y_train['CD36']
print(variable.describe())

correlations = x_train.corrwith(variable, method='kendall')

abs_correlations = abs(correlations)

sorted_correlations = abs_correlations.sort_values(ascending=False)

top_250_vars = sorted_correlations.index[:250]

correlation_pairs = x_train[top_250_vars].corr(method='kendall')

sns.heatmap(correlation_pairs, center=0., cmap="crest")
plt.title("Mapa ciepła współczynników korelacji")
plt.show()


# Zadanie 2

scipy.stats.probplot(x=y_train["CD36"], dist=scipy.stats.norm(), plot=plt)
plt.title('Wykres kwantylowy')
plt.xlabel('Kwantyle')
plt.ylabel('Uporzadkowane wartosci')
plt.show()

out = scipy.stats.normaltest(y_train)

correlations.sort_values()

densityEstim = gaussian_kde(x_train["BLVRB"])
x = np.linspace(min(x_train["BLVRB"]), max(x_train["BLVRB"]), 1000)
plt.plot(x, densityEstim(x))
plt.title("Estymator gestosci")
plt.xlabel('Wartosc')
plt.ylabel('Gestosc')
plt.show()

scipy.stats.probplot(x=x_train["BLVRB"], dist=scipy.stats.halfnorm(), plot=plt)
plt.title('Wykres kwantylowy')
plt.xlabel('Kwantyle')
plt.ylabel('Uporzadkowane wartosci')
plt.show()

rand_signs = np.random.choice([-1,1], 6800)
z = x_train["BLVRB"] * rand_signs

out = scipy.stats.normaltest(z)

b_test = x_test["BLVRB"]
out = scipy.stats.ks_2samp(b_test, x_train["BLVRB"])


# Zadanie 3

alpha_values = [0., 0.25, 0.5, 0.75, 1]
l1_ratio_values = [0.1, 0.3, 0.5, 0.7, 0.9]

grid_elasticnet = itertools.product(alpha_values, l1_ratio_values)
num_folds = 3

kf = KFold(n_splits=num_folds, shuffle=True)

results_elasticnet = []

for i, params in enumerate(grid_elasticnet):
    rmse = 0
    r2 = 0

    alpha, l1 = params
    model = ElasticNet(alpha=alpha, l1_ratio=l1)
    
    folds = kf.split(x_train)
    for j, (train_index, test_index) in enumerate(folds):
               
        train_fold_x, test_fold_x = x_train.iloc[train_index], x_train.iloc[test_index]
        train_fold_y, test_fold_y = y_train.iloc[train_index], y_train.iloc[test_index]
        
        model.fit(train_fold_x, train_fold_y)

        train_pred = model.predict(train_fold_x)
        train_rmse = np.sqrt(mean_squared_error(train_fold_y, train_pred))
        train_r2 = r2 = r2_score(train_fold_y, train_pred) 
        predictions = model.predict(test_fold_x)

        rmse = np.sqrt(mean_squared_error(test_fold_y, predictions))
        r2 = r2_score(test_fold_y, predictions)    
        print("rmse = ", rmse/num_folds, "COD = ", r2/num_folds)
        results_elasticnet.append([alpha, l1, j, train_rmse, train_r2, rmse, r2])

result_df = pd.DataFrame(results_elasticnet, columns=["alpha", "l1", "fold", "Train RMSE", "Train R2", "RMSE", "R2"])

sns.violinplot(x=result_df["fold"], y=result_df["RMSE"])
sns.swarmplot(x=result_df["fold"], y= result_df["R2"])
plt.title('Wykres skrzypcowy')
plt.ylim(-0.3, 1.5)

df = result_df.groupby(["alpha", "l1"]).mean()


# Zadanie 4

max_features = [80, 95, 105]
n_estimators = [100, 200, 300]
max_depth = [5, 10, 15]

grid_rf = itertools.product(max_features, n_estimators, max_depth)

num_folds = 3
kf = KFold(n_splits=num_folds, shuffle=True)

results = []

for i, params in enumerate(grid_rf):
    rmse = 0
    r2 = 0

    max_features, n_estimators, max_depth = params
    model = RandomForestRegressor(max_features=max_features, n_estimators=n_estimators, max_depth=max_depth)
    
    folds = kf.split(x_train)
    for j, (train_index, test_index) in enumerate(folds):
               
        train_fold_x, test_fold_x = x_train.iloc[train_index], x_train.iloc[test_index]
        train_fold_y, test_fold_y = y_train.iloc[train_index], y_train.iloc[test_index]
        
        model.fit(train_fold_x, train_fold_y.values.ravel())

        train_pred = model.predict(train_fold_x)
        train_rmse = np.sqrt(mean_squared_error(train_fold_y, train_pred))
        train_r2 = r2 = r2_score(train_fold_y, train_pred) 
        predictions = model.predict(test_fold_x)

        rmse = np.sqrt(mean_squared_error(test_fold_y, predictions))
        r2 = r2_score(test_fold_y, predictions) 
        
        results.append([max_features, n_estimators, max_death, j, train_rmse, train_r2, rmse, r2])

result_df = pd.DataFrame(results, columns=["max_features", "n_estimators", "max_depth", "fold", "Train RMSE", "Train R2", "RMSE", "R2"])
result_df.sort_values('RMSE')

sns.boxplot(x=result_df["fold"], y=result_df["RMSE"])

plt.title('Wykres pudelkowy')

df = result_df.groupby(["max_features", "n_estimators", "max_depth"]).mean()


# Zadanie 5

num_folds = 3
kf = KFold(n_splits=num_folds, shuffle=True)

benchmark_results = []

benchmark_results = []
for train_index, test_index in kf.split(x_train):
 
    train_fold_x, test_fold_x = x_train.iloc[train_index], x_train.iloc[test_index]
    train_fold_y, test_fold_y = y_train.iloc[train_index], y_train.iloc[test_index]
    
    mean_train_y = train_fold_y.mean().values[0]
    
    train_pred = np.full_like(train_fold_y, fill_value=mean_train_y, dtype=np.float64)
    train_rmse = np.sqrt(mean_squared_error(train_fold_y, train_pred))
    train_r2 = r2 = r2_score(train_fold_y, train_pred) 
    predictions = np.full_like(test_fold_y, fill_value=mean_train_y, dtype=np.float64)
    
    rmse = np.sqrt(mean_squared_error(test_fold_y, predictions))
    r2 = r2_score(test_fold_y, predictions)
    
    benchmark_results.append([len(benchmark_results), mean_train_y,rmse,r2, train_rmse, train_r2])

benchmark_df = pd.DataFrame(benchmark_results, columns=[ "fold", "mean_train_y", "RMSE", "R2", "Train RMSE", "Train R2"])
benchmark_df.sort_values('RMSE', inplace=True)

benchmark_df.mean()


# Zadanie 6

model = RandomForestRegressor(max_features=105, n_estimators=200, max_depth=5)
model.fit(x_train, y_train.values.ravel())
y_test_result_RF = model.predict(x_test)
y_test_result_RF

y_test_result_RF_df = pd.DataFrame(y_test_result_RF, columns=['Prediction'])

y_test_result_RF_df.to_csv('y_test_result_Random_Forest.csv', index=False)

learning_rate = [0.01, 0.1, 0.2]
n_estimators = [100, 200, 300]
max_depth = [3, 5, 7]

grid_gbm = itertools.product(learning_rate, n_estimators, max_depth)
print(list(grid_gbm)) 

num_folds = 3
kf = KFold(n_splits=num_folds, shuffle=True)

results = []

grid_gbm = itertools.product(learning_rate, n_estimators, max_depth)

for i, params in enumerate(grid_gbm):
    lr, n_estimators, max_depth = params
    model = GradientBoostingRegressor(learning_rate=lr, n_estimators=n_estimators, max_depth=max_depth)
    
    rmse_list = []
    r2_list = []

    folds = kf.split(x_train)
    for j, (train_index, test_index) in enumerate(folds):

        train_fold_x, test_fold_x = x_train.iloc[train_index], x_train.iloc[test_index]
        train_fold_y, test_fold_y = y_train.iloc[train_index], y_train.iloc[test_index]
        
        model.fit(train_fold_x, train_fold_y.values.ravel())

        train_pred = model.predict(train_fold_x)
        train_rmse = np.sqrt(mean_squared_error(train_fold_y, train_pred))
        train_r2 = r2_score(train_fold_y, train_pred)

        predictions = model.predict(test_fold_x)
     
        rmse = np.sqrt(mean_squared_error(test_fold_y, predictions))
        r2 = r2_score(test_fold_y, predictions)
        
        rmse_list.append(rmse)
        r2_list.append(r2)
        
        results.append([lr, n_estimators, max_depth, j, train_rmse, train_r2, rmse, r2])

results_df = pd.DataFrame(results, columns=['Learning Rate', 'N Estimators', 'Max Depth', 'Fold', 'Train RMSE', 'Train R2', 'Test RMSE', 'Test R2'])

mean_results = results_df.groupby(['Learning Rate', 'N Estimators', 'Max Depth']).mean().reset_index()

sorted_results = mean_results.sort_values(by='Test RMSE')
sorted_results

model = GradientBoostingRegressor(learning_rate=0.1, n_estimators=200, max_depth=5)
model.fit(x_train, y_train.values.ravel())
y_test_result_GBM = model.predict(x_test)
y_test_result_GBM

y_test_result_GBM_df = pd.DataFrame(y_test_result_GBM, columns=['Prediction'])

y_test_result_GBM_df.to_csv('y_test_result_GBM.csv', index=False)

max_depth = [5, 10, 15]
n_estimators = [100, 200, 300]
learning_rate = [0.01, 0.1, 0.2]

grid_xgb = itertools.product(max_depth, n_estimators, learning_rate)
num_folds = 3
kf = KFold(n_splits=num_folds, shuffle=True)

results = []

for i, params in enumerate(grid_xgb):
    max_depth, n_estimators, learning_rate = params
    rmse = 0
    r2 = 0

    model = XGBRegressor(max_depth=max_depth, n_estimators=n_estimators, learning_rate=learning_rate, verbosity=0)
    rmse_list = []
    r2_list = []

    folds = kf.split(x_train)
    for j, (train_index, test_index) in enumerate(folds):

        train_fold_x, test_fold_x = x_train.iloc[train_index], x_train.iloc[test_index]
        train_fold_y, test_fold_y = y_train.iloc[train_index], y_train.iloc[test_index]
        
        model.fit(train_fold_x, train_fold_y.values.ravel())

        train_pred = model.predict(train_fold_x)
        train_rmse = np.sqrt(mean_squared_error(train_fold_y, train_pred))
        train_r2 = r2_score(train_fold_y, train_pred)

        predictions = model.predict(test_fold_x)

        rmse = np.sqrt(mean_squared_error(test_fold_y, predictions))
        r2 = r2_score(test_fold_y, predictions)
        
        rmse_list.append(rmse)
        r2_list.append(r2)
        
        results.append([learning_rate, n_estimators, max_depth, j, train_rmse, train_r2, rmse, r2])

results_df = pd.DataFrame(results, columns=['Learning Rate', 'N Estimators', 'Max Depth', 'Fold', 'Test RMSE', 'Test R2'])

mean_results = results_df.groupby(['Learning Rate', 'N Estimators', 'Max Depth']).mean().reset_index()

sorted_results = mean_results.sort_values(by='Test RMSE')
sorted_results

model = XGBRegressor(learning_rate=0.1, n_estimators=300, max_depth=5)
model.fit(x_train, y_train.values.ravel())
y_test_result_XGB = model.predict(x_test)
y_test_result_XGB

y_test_result_XGB_df = pd.DataFrame(y_test_result_XGB, columns=['Prediction'])

y_test_result_XGB_df.to_csv('y_test_result_XGB.csv', index=False)


# Przesyłanie predykcji 

RF_df = pd.read_csv("/kaggle/input/random-forest/y_test_result_Random_Forest.csv")
RF_df.rename({"Prediction":"Expected"}, axis="columns", inplace=True)
RF_df["Id"] = RF_df.index
RF_df.to_csv('submission.csv', index=False)

GBM_df = pd.read_csv("/kaggle/input/gbm-model/y_test_result_GBM.csv")
GBM_df.rename({"Prediction":"Expected"}, axis="columns", inplace=True)
GBM_df["Id"] = GBM_df.index
GBM_df.to_csv('submission.csv', index=False)

XGB_df = pd.read_csv("/kaggle/input/xgb-model/y_test_result_XGB.csv")
XGB_df.rename({"Prediction":"Expected"}, axis="columns", inplace=True)
XGB_df["Id"] = XGB_df.index
XGB_df.to_csv('submission.csv', index=False)


