

# Real Estate Pricing Prediction

This project uses Linear Regression to predict **house prices per unit area** based on multiple features such as location, age, distance from MRT station, and number of nearby facilities. The workflow involves data exploration, model training, prediction, and evaluation.

---

## Dataset Overview

The dataset used is `Real_Estate_Pricing.csv`, which includes the following features:

* X1 transaction date *(dropped during preprocessing)*
* X2 house age
* X3 distance to the nearest MRT station
* X4 number of convenience stores
* X5 latitude
* X6 longitude
* Y: house price of unit area (Target variable)

---

## Libraries Used

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
import scipy.stats as stats
import pylab
```

---

## Data Loading and Exploration

```python
df = pd.read_csv('Real_Estate_Pricing.csv')
df.head()
df.info()
df.describe()
```

* The `Transaction date` column is removed since it is not needed for prediction.

```python
del df['Transaction date']
```

---

## Feature Selection and Train-Test Split

Splitting the features and target variable:

```python
x = df.drop("House price of unit area", axis=1)
y = df["House price of unit area"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
```

---

## Model Training

```python
lm = LinearRegression()
model = lm.fit(x_train, y_train)
```

### Coefficient Interpretation

```python
cfd = pd.DataFrame(lm.coef_, x.columns, columns=['Coef'])
print(cfd)
```

---

## Model Evaluation

### Score (R²)

```python
model.score(x_test, y_test)
```

### Predictions and Plotting

```python
predications = lm.predict(x_test)
sns.scatterplot(x=y_test, y=predications)
```

### Evaluation Metrics

```python
print('MAE :', mean_absolute_error(y_test, predications))
print('MSE :', mean_squared_error(y_test, predications))
print('RMSE:', math.sqrt(mean_squared_error(y_test, predications)))
```

---

## Residual Analysis

Analyzing how the errors (residuals) are distributed:

```python
residuals = y_test - predications
sns.displot(residuals, bins=30, kde=True)
```

### Normality Check with Q-Q Plot

```python
stats.probplot(residuals, dist='norm', plot=pylab)
pylab.show()
```

---

