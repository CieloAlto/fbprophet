from fbprophet import Prophet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

test_df = pd.read_csv('./csv/daily.csv')

test_df['y_orig'] = test_df['y']
test_df['y'] = np.log(test_df['y'])

model = Prophet()
model.fit(test_df,algorithm='Newton')

future_data = model.make_future_dataframe(periods=12, freq = 'm')
forecast_data = model.predict(future_data)

model.plot(forecast_data, xlabel = 'Month', ylabel = 'Count/Day')
plt.show()


