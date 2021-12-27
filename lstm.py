import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class volatility(Dataset):
    def __init__(self, window_len=7) -> None:
        super(volatility).__init__()
        self.window_len = window_len
        self._load_dataset()

    def _load_dataset(self):
        idx = pd.date_range('1/5/2017', '10/14/2019')
        data = pd.read_csv('Data/training_data.csv')
        final_df = pd.DataFrame()
        for g in data.groupby('Tenor').groups:
            grp = data.groupby('Tenor').get_group(g)
            grp = grp.set_index(['Date'])
            grp.index = pd.DatetimeIndex(grp.index)

            grp = grp.reindex(idx, fill_value=np.nan)
            grp = grp.interpolate()
            grp = grp.backfill()  # To Backfill tenor values
            grp = grp.reset_index()
            grp['Tenor'] = grp['Tenor'].apply(lambda x: (
                x[-1] + x[:-1]) if len(x) == 3 else (x[-1] + '0' + x[:-1]))
            grp.columns = ['Date', 'Tenor', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7',
                           '0.8', '0.9', '1', '1.1', '1.2', '1.3', '1.4', '1.5', '1.6', '1.7',
                           '1.8', '1.9']
            final_df = final_df.append(grp)

        final_df = final_df.sort_values(by=['Date', 'Tenor'])
        df = pd.DataFrame()
        grouped = final_df.groupby('Date')
        for g in grouped.groups:
            grp = grouped.get_group(g)
            grp = grp.drop(['Date', 'Tenor'], axis=1)
            df1 = grp.stack().swaplevel()
            df1.index = df1.index.map('{0[0]}_{0[1]}'.format)
            df1.reset_index(inplace=True, drop=True)
            df = df.append(df1.to_frame().T)
            print('{}/{}'.format(len(df), len(final_df)/19), end='\r')
        self.data = df

    def __getitem__(self, index):
        X = torch.tensor(self.data.iloc[index:index+10, :])
        y = self.data.iloc[index+10, :]


x = volatility()
print(x.data.head(10))

# print(x.data.info())
