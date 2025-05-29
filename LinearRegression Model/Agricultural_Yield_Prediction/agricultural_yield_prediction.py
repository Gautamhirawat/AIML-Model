

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

df = pd.read_csv('Agricultural_Yield_Prediction.csv')

df.head(10)

df.info()

df.describe()

correlation = df.corr(numeric_only = True)
correlation

fig = px.imshow(correlation, text_auto = '.3f', aspect = 'auto',
               title = '<b>Correlation Matrix</b>', color_continuous_scale = 'tropic')
fig.update_layout(title_font_color = 'olive', font_color = 'fuchsia')
fig.show()

fig = make_subplots(rows = 5, cols = 1,
                    subplot_titles = ['<b>Production vs Yield</b>', '<b>Area vs Yield</b>',
                                     '<b>Annual Rainfall vs Yield</b>', '<b>Fertilizer vs Yield</b>',
                                     '<b>Pesticide vs Yield</b>'])

fig.add_trace(go.Scattergl(x = df['Production'], y = df['Yield'], mode = 'markers',
                          marker_line_color = 'darkblue', marker_line_width = 1,
                          name = 'Production'), row = 1, col = 1)

fig.add_trace(go.Scattergl(x = df['Area'], y = df['Yield'], mode = 'markers',
                           marker_color = 'coral', marker_line_color = 'darkslategray',
                           marker_line_width = 1, name = 'Area'), row = 2, col = 1)

fig.add_trace(go.Scattergl(x = df['Annual_Rainfall'], y = df['Yield'], mode = 'markers',
                           marker_color = 'beige', marker_line_width = 1,
                           marker_line_color = 'coral',
                           name = 'Annual Rainfall'), row = 3, col = 1)

fig.add_trace(go.Scattergl(x = df['Fertilizer'], y = df['Yield'], mode = 'markers',
                            marker_line_width = 1, name = 'Fertilizer'),row = 4, col = 1)

fig.add_trace(go.Scattergl(x = df['Pesticide'], y = df['Yield'], mode = 'markers',
                            marker_line_width = 1, name = 'Pesticide'), row = 5, col = 1)
fig.update_xaxes(type = 'log')
fig.update_xaxes(title = 'Production', row = 1, col = 1)
fig.update_xaxes(title = 'Area', row = 2, col = 1)
fig.update_xaxes(title = 'Annual Rainfall', row = 3, col = 1)
fig.update_xaxes(title = 'Fertilizer', row = 4, col = 1)
fig.update_xaxes(title = 'Pesticide', row = 5, col = 1)

fig.update_yaxes(type = 'log', title = 'Yield')
fig.update_layout(title = '<b>Impact of Various Factors on Crop Yield</b>',
                  title_font_color = 'sienna', legend_font_color = 'saddlebrown',
                  height = 1700)
fig.show()

X = df.drop(['Crop', 'Crop_Year', 'Season', 'State', 'Yield','Production'], axis = 1)
y = df['Yield']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2 ,random_state = 42)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

lr = LinearRegression()
lr.fit(X_train, y_train)

pred_lr = lr.predict(X_test)
r2_lr = r2_score(y_test, pred_lr)
print("r2 Score of Linear Regression: ", r2_lr)

fig = px.scatter(x = y_test, y = pred_lr, trendline = 'ols',
                 title = '<b>  Linear Regression Model  </b>')
fig.update_traces(marker_size = 12, marker_color = 'sienna', marker_line_width = 1)
fig.update_xaxes(title = 'Yield')
fig.update_yaxes(title = 'Prediction')
fig.show()

comparison = pd.DataFrame()

comparison['Type'] = ['Linear Regression']
comparison['r2 Score'] = [r2_lr]
comparison

print(pred_lr)

