
# Agricultural Yield Prediction

This project aims to predict **crop yield** based on various agricultural and environmental factors using **Linear Regression**. It includes data visualization, correlation analysis, and performance evaluation.

You can run this project using **Google Colab**, **Jupyter Notebook**, or any Python environment.

---

## Dataset Overview

The dataset (`Agricultural_Yield_Prediction.csv`) contains the following features:

```
Crop, Crop_Year, Season, State, Area, Production, Annual_Rainfall, Fertilizer, Pesticide, Yield
```

---

##  Libraries Used

```python
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
```

---

##  Data Preparation

1. Load the dataset:

```python
df = pd.read_csv('Agricultural_Yield_Prediction.csv')
```

2. Drop non-numeric or irrelevant columns for model training:

```python
X = df.drop(['Crop', 'Crop_Year', 'Season', 'State', 'Yield', 'Production'], axis=1)
y = df['Yield']
```

3. Split data into train and test sets:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

##  Exploratory Data Analysis

### 🔹 Correlation Matrix

Visualizes how features like Area, Fertilizer, Rainfall, and Pesticide correlate with Yield.

```python
correlation = df.corr(numeric_only=True)
px.imshow(correlation)
```

### 🔹 Feature vs Yield Relationships

Using scatter plots to visualize how different factors affect yield:

* Production vs Yield
* Area vs Yield
* Rainfall vs Yield
* Fertilizer vs Yield
* Pesticide vs Yield

Each plot uses logarithmic scaling for better visibility of variation.

---

##  Model Building

### 🔹 Model Used: **Linear Regression**

```python
lr = LinearRegression()
lr.fit(X_train, y_train)
```

---

##  Coefficients

View how much each feature impacts the predicted Yield:

```python
print(lr.coef_)
print(lr.intercept_)
```

---

##  Predictions

Make predictions on the test set:

```python
pred_lr = lr.predict(X_test)
r2_lr = r2_score(y_test, pred_lr)
print("R² Score: ", r2_lr)
```

---

##  Visualization of Predictions

Scatter plot of actual vs predicted yield:

```python
fig = px.scatter(x=y_test, y=pred_lr, trendline='ols',
                 title='Linear Regression Model')
fig.show()
```

---

##  Evaluation Metrics

Compare models (currently only Linear Regression):

```python
comparison = pd.DataFrame()
comparison['Type'] = ['Linear Regression']
comparison['r2 Score'] = [r2_lr]
print(comparison)
```

---

