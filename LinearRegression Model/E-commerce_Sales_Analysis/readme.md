

# E-commerce Customer Spending Prediction

This project uses Linear Regression to predict the **Yearly Amount Spent** by customers on an e-commerce website based on their usage patterns. It involves exploratory data analysis, model training, and performance evaluation.

---

## Dataset Overview

The dataset contains customer behavior metrics from an e-commerce platform:

```
- Avg. Session Length
- Time on App
- Time on Website
- Length of Membership
- Yearly Amount Spent (Target Variable)
```

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

1. Load the dataset:

```python
df = pd.read_csv('/Datasets/Customers.csv')
```

2. Display the structure and first few records:

```python
df.head()
df.info()
```

---

## Exploratory Data Analysis

Visualizing relationships between features and the target variable:

```python
sns.jointplot(x="Time on Website", y="Yearly Amount Spent", data=df, alpha=0.5)
sns.jointplot(x="Time on App", y="Yearly Amount Spent", data=df, alpha=0.5)
sns.pairplot(df, kind='scatter', plot_kws={'alpha': 0.4})
sns.lmplot(x='Length of Membership', y='Yearly Amount Spent', data=df, scatter_kws={'alpha': 0.3})
```

---

## Feature Selection and Train-Test Split

Selecting relevant features and splitting the data:

```python
x = df[['Time on App', 'Time on Website', 'Length of Membership', 'Avg. Session Length']]
y = df['Yearly Amount Spent']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
```

---

## Model Training

Training a Linear Regression model:

```python
lm = LinearRegression()
lm.fit(x_train, y_train)
```

Viewing the coefficients:

```python
lm.coef_
cfd = pd.DataFrame(lm.coef_, x.columns, columns=['Coef'])
print(cfd)
```

---

## Predictions and Evaluation

Generating predictions on the test set:

```python
predications = lm.predict(x_test)
```

Plotting predictions vs actual values:

```python
sns.scatterplot(x=y_test, y=predications)
plt.xlabel('Actual values')
plt.ylabel('Predictions')
plt.title('Linear Regression Model - Prediction for E-commerce Website')
```

### Evaluation Metrics

```python
print('MAE :', mean_absolute_error(y_test, predications))
print('MSE :', mean_squared_error(y_test, predications))
print('RMSE:', math.sqrt(mean_squared_error(y_test, predications)))
```

---

## Residual Analysis

Analyzing the distribution of residuals:

```python
residuals = y_test - predications
sns.displot(residuals, bins=30, kde=True)
```

Checking normality of residuals using Q-Q plot:

```python
stats.probplot(residuals, dist='norm', plot=pylab)
pylab.show()
```

---
