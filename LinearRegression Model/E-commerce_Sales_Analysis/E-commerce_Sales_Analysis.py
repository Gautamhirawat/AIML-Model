
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/Datasets/Customers.csv')

df.head()

df.info()

sns.jointplot(x = "Time on Website",y = "Yearly Amount Spent", data = df , alpha = 0.5)

sns.jointplot(x = "Time on App",y = "Yearly Amount Spent", data = df , alpha = 0.5)

sns.pairplot(df,kind = 'scatter',plot_kws={'alpha':0.4})

sns.lmplot(x='Length of Membership',y = 'Yearly Amount Spent', data = df, scatter_kws={'alpha':0.3})

from sklearn.model_selection import train_test_split

x = df[['Time on App', 'Time on Website', 'Length of Membership', 'Avg. Session Length']]
y = df['Yearly Amount Spent']

x_train, x_test , y_train , y_test = train_test_split(x,y,test_size = 0.3, random_state = 42)

x_train

y_train

from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(x_train,y_train)

lm.coef_

cfd = pd.DataFrame(lm.coef_ , x.columns , columns=['Coef'])
print(cfd)

predications = lm.predict(x_test)

predications

sns.scatterplot(x = y_test , y = predications)
plt.xlabel('Actual values')
plt.ylabel('Predctions')
plt.title('Linear regression model - Prediction for Ecommerse website')

from sklearn.metrics import mean_absolute_error , mean_squared_error
import math

print('MAE : ' , mean_absolute_error(y_test , predications))
print('MSE : ' , mean_squared_error(y_test , predications))
print('RMSE: ', math.sqrt(mean_squared_error(y_test , predications)))

residuals = y_test - predications

sns.displot(residuals , bins = 30 , kde = True)

import pylab
import scipy.stats as stats

stats.probplot(residuals , dist = 'norm' , plot = pylab)
pylab.show()

