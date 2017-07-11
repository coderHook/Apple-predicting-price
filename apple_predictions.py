
# coding: utf-8

# In[1]:

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from datetime import datetime

df = pd.read_csv("aapl.csv")

df.head()


# In[2]:

df['Date'] = pd.to_datetime(df['Date'])

df = df.sort_values('Date', ascending=True)


# # Indicators

# In[4]:

# Lets start by getting mean price, and standard deviation weekly, monthly and yearly

mean_price_5d = pd.rolling_mean(df['Close'], window=5).shift(1)
mean_price_30d = pd.rolling_mean(df['Close'], window=30).shift(1)
mean_price_365d = pd.rolling_mean(df['Close'], window=365).shift(1)



# In[6]:

df['mean_price_5d'] = mean_price_5d
df['mean_price_30d'] = mean_price_30d
df['mean_price_365d'] = mean_price_365d


# In[10]:

std_price_5d = pd.rolling_std(df['Close'], window=5).shift(1)
std_price_30d = pd.rolling_std(df['Close'], window=30).shift(1)
std_price_365d = pd.rolling_std(df['Close'], window=365).shift(1)

df['std_price_5d'] = std_price_5d
df['std_price_30d'] = std_price_30d
df['std_price_365d'] = std_price_365d


# In[11]:

##### Originally we had here more indicators but after a problem the code wasnt saved and the predictions show that the indicators that were reducing the error were those ones above.


# # LinearRegression: predicting our prices.

# In[18]:

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
df = df.dropna(axis=0)

# Lets separate the df in 2: train(biggining till 2013) and test(2013 till today)

train = df[df['Date'] < datetime(year=2013, month=1, day=1)]
test = df[df['Date'] >= datetime(year=2013, month=1, day=1)]

features1 = ['mean_price_5d', 'mean_price_30d', 'mean_price_365d']
features2 = ['std_price_5d', 'std_price_30d', 'std_price_365d']


# In[19]:

model = LinearRegression()
model.fit(train[features1 + features2], train['Close'])
prediction1 = model.predict(test[features1 + features2])
mae = mean_absolute_error(test['Close'], prediction1)

print('mae1', mae)


# # Predicting Tomorrows Apple's price

# In[33]:

model2 = LinearRegression()
model2.fit(df[features1 + features2], df['Close'])
predict_tomorrow = model2.predict(df[features1 + features2])

mae2 = mean_absolute_error(df['Close'], predict_tomorrow)

print('Apple price for tomorrow: ', predict_tomorrow[-1])
print('Mean Absolute Error of our prediction: ', mae2)


# ### Chart of our Model

# In[34]:

import matplotlib.pyplot as plt

plt.figure(figsize=(20, 9))
plt.plot(df['Date'], df['Close'], label='Actual prices')
plt.plot(df['Date'], predict_tomorrow, label='Predicted Prices')
plt.plot(df['Date'][1], predict_tomorrow[-1])
plt.legend(loc='lower right')
plt.show()


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



