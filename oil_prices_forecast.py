# -*- coding: utf-8 -*-
import pandas as pd
from fbprophet import Prophet

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


df = pd.read_csv('oil_prices.csv')
format = '%d/%m/%Y'
df['ts'] = pd.to_datetime(df['ts'], format=format)
df = df.set_index(pd.DatetimeIndex(df['ts']))

df = df.rename(columns={'ts': 'ds',
                        'price': 'y'})

# set the uncertainty interval to 95% (the Prophet default is 80%)
my_model = Prophet(n_changepoints=10,interval_width=0.95,daily_seasonality=False,weekly_seasonality=True,yearly_seasonality=True,uncertainty_samples=1000)

forecast_period = 12 # Number of forecasted time steps

my_model.fit(df)

# In order to obtain forecasts of our time series, we must provide Prophet with a new DataFrame 
# containing a ds column that holds the dates for which we want predictions:
future_dates = my_model.make_future_dataframe(periods=forecast_period, freq='12MS')

# The DataFrame of future dates is then used as input to the predict method of our fitted model:
forecast = my_model.predict(future_dates)

'''
Prophet returns a large DataFrame with many interesting columns, but we subset our output to the columns most relevant to forecasting, which are:
ds: the datestamp of the forecasted value
yhat: the forecasted value of our metric (in Statistics, yhat is a notation traditionally used to represent the predicted values of a value y)
yhat_lower: the lower bound of our forecasts
yhat_upper: the upper bound of our forecasts
'''

#my_model.plot(forecast, uncertainty=True)

fig = plt.figure()
plt.plot(df['ds'], df['y'])
plt.plot(forecast[-forecast_period:]['ds'], forecast[-forecast_period:]['yhat'])
plt.plot(forecast[-forecast_period:]['ds'], forecast[-forecast_period:]['yhat_upper'], 'k--')
plt.plot(forecast[-forecast_period:]['ds'], forecast[-forecast_period:]['yhat_lower'], 'k--')
plt.title('Brent crude oil price forecast')
plt.xlabel('Date (years)')
plt.ylabel('U.S. dollars per barrel')
plt.legend(['Data','Forecast','Upper limit','Lower limit'])
plt.show()