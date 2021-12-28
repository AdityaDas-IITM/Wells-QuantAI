import pandas as pd
import matplotlib.pyplot as plt

train_df = pd.read_csv("data/training_data.csv")
pred_df = pd.read_csv("prediction_spatial_15.csv")
print(pred_df.head())

tenor_vals = [30*v for v in [2,3,6,9]] + [365*v for v in [1,2,3,4,5,6,7,8,9,10,15,20,25,30,40]]

# mnns = pred_df.columns[2:]
# mnns = [float(c) for c in mnns]
vol_2m_10_15 = pred_df.iloc[0, 2:]
vol_2m_01_06 = pred_df.iloc[1121, 2:]
vol_40y_10_15 = pred_df.iloc[18, 2:]
vol_40y_01_06 = pred_df.iloc[1139, 2:]
vol_train_40y_01_05 = train_df.iloc[18, 2:]

vol_10_mnns_10_15 = pred_df.iloc[0:19, 11]
vol_10_mnns_01_06 = pred_df.iloc[1121:1140, 11]
vol_19_mnns_10_15 = pred_df.iloc[0:19, 20]
vol_19_mnns_01_06 = pred_df.iloc[1121:1140, 20]
vol_01_mnns_10_15 = pred_df.iloc[0:19, 2]
vol_01_mnns_01_06 = pred_df.iloc[1121:1140, 2]
vol_train_19_mnns_01_05 = train_df.iloc[0:19, 20]
vol_train_01_mnns_01_05 = train_df.iloc[0:19, 2]
vol_train_10_mnns_01_05 = train_df.iloc[0:19, 11]

# print(vol_01_mnns_10_15[0])
''' 
# Volatility vs Moneyness Plots for 2M, 40Y Tenors
plt.figure()
plt.plot(vol_2m_10_15)
plt.title("Volatility vs Moneyness for 2M Tenor, 10/15/2019")
plt.xlabel("Moneyness")
plt.ylabel("Volatility (at 2M)")
plt.ylim([0,0.7])

plt.figure()
plt.plot(vol_2m_01_06)
plt.title("Volatility vs Moneyness for 2M Tenor, 01/06/2020")
plt.xlabel("Moneyness")
plt.ylabel("Volatility (at 2M)")
plt.ylim([0,0.7])

plt.figure()
plt.plot(vol_40y_10_15)
plt.title("Volatility vs Moneyness for 40Y Tenor, 10/15/2019")
plt.xlabel("Moneyness")
plt.ylabel("Volatility (at 40Y)")
plt.ylim([0,0.7])

plt.figure()
plt.plot(vol_40y_01_06)
plt.title("Volatility vs Moneyness for 40Y Tenor, 01/06/2020")
plt.xlabel("Moneyness")
plt.ylabel("Volatility (at 40Y)")
plt.ylim([0,0.7])

plt.figure()
plt.plot(vol_train_40y_01_05)
plt.title("Volatility vs Moneyness for 40Y Tenor, 01/05/2017")
plt.xlabel("Moneyness")
plt.ylabel("Volatility (at 40Y)")
plt.ylim([0,0.7])

plt.show()
#'''

#'''
# Volatility vs. Tenor for few moneynesses

plt.figure()
plt.plot(tenor_vals, vol_10_mnns_10_15)
plt.title("Volatility vs Tenor for 1.0 Moneyness, 10/15/2019")
plt.xlabel("Tenor")
plt.ylabel("Volatility (at 1.0 Moneyness)")
plt.ylim([0.1,0.3])

plt.figure()
plt.plot(tenor_vals, vol_10_mnns_01_06)
plt.title("Volatility vs Tenor for 1.0 Moneyness, 01/06/2020")
plt.xlabel("Tenor")
plt.ylabel("Volatility (at 1.0 Moneyness)")
plt.ylim([0.1,0.3])

plt.figure()
plt.plot(tenor_vals, vol_train_10_mnns_01_05)
plt.title("Train Volatility vs Tenor for 1.0 Moneyness, 01/05/2017")
plt.xlabel("Tenor")
plt.ylabel("Volatility (at 1.0 Moneyness)")
plt.ylim([0.1,0.3])

plt.figure()
plt.plot(tenor_vals, vol_train_01_mnns_01_05)
plt.title("Train Volatility vs Tenor for 0.1 Moneyness, 01/05/2017")
plt.xlabel("Tenor")
plt.ylabel("Volatility (at 0.1 Moneyness)")
plt.ylim([0,0.7])

plt.figure()
plt.plot(tenor_vals, vol_train_19_mnns_01_05)
plt.title("Train Volatility vs Tenor for 1.9 Moneyness, 01/05/2017")
plt.xlabel("Tenor")
plt.ylabel("Volatility (at 1.9 Moneyness)")
plt.ylim([0,0.7])

plt.figure()
plt.plot(tenor_vals, vol_01_mnns_10_15)
plt.title("Volatility vs Tenor for 0.1 Moneyness, 10/15/2019")
plt.xlabel("Tenor")
plt.ylabel("Volatility (at 0.1 Moneyness)")
plt.ylim([0,0.7])

plt.figure()
plt.plot(tenor_vals, vol_01_mnns_01_06)
plt.title("Volatility vs Tenor for 0.1 Moneyness, 01/06/2020")
plt.xlabel("Tenor")
plt.ylabel("Volatility (at 0.1 Moneyness)")
plt.ylim([0,0.7])

plt.figure()
plt.plot(tenor_vals, vol_19_mnns_10_15)
plt.title("Volatility vs Tenor for 1.9 Moneyness, 10/15/2019")
plt.xlabel("Tenor")
plt.ylabel("Volatility (at 1.9 Moneyness)")
plt.ylim([0,0.7])

plt.figure()
plt.plot(tenor_vals, vol_19_mnns_01_06)
plt.title("Volatility vs Tenor for 1.9 Moneyness, 01/06/2020")
plt.xlabel("Tenor")
plt.ylabel("Volatility (at 1.9 Moneyness)")
plt.ylim([0,0.7])

plt.show()
#'''