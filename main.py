# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 13:01:38 2018

@author: jhdbenito
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime #, timedelta
from sklearn.linear_model import LinearRegression

# Parse file
parser = argparse.ArgumentParser()
parser.add_argument("file", help="csv file with your account stats")
args = parser.parse_args()
file = args.file

df = pd.read_csv(file, delimiter = ';')
dalist = ['Saldo', 'Importe'] 
for attr in dalist:
    df[attr] = df[attr].map(lambda x: x.replace('.','').replace(',','.')[:-4])
    df[attr]= pd.to_numeric(df[attr])

df['Fecha 1'] = pd.to_datetime(df['Fecha 1'], dayfirst=True )
df['OrdinalDate'] = df['Fecha 1'].map(lambda x: datetime.toordinal(x))
df['OrdinalDate'] = df['OrdinalDate'] - df['OrdinalDate'].min(axis=0)
pos = df.query('Importe > 0')

# Linear Regression
lm = LinearRegression()
X = df[['OrdinalDate']]
Y = df['Saldo']
lm.fit(X,Y)
Yhat = lm.predict(X)

# Polynomial regression:

X = df['OrdinalDate']
pm = np.poly1d(np.polyfit(X,Y,6))
Yhat2 = pm(X)

# Chart
font= {'fontsize': 20}
fig, ax = plt.subplots()
ax.set_title('Hope it goes up :D', fontdict=font)
ax.get_yaxis().set_visible(False)
fig.set_size_inches(10,8)
ax.plot_date(pos['Fecha 1'], pos['Saldo'],  color='tab:green')
ax.plot_date(df['Fecha 1'], df['Saldo'], ls='-', lw=1, marker='', drawstyle = 'steps-post', color='tab:red')
ax.bar(pos['Fecha 1'], pos['Importe'], width=10)

#ax.plot(pos['OrdinalDate'], pos['Saldo'], marker='P', lw=0,  color='tab:green')
#ax.plot(df['OrdinalDate'], df['Saldo'], ls='-', lw=1, marker='', drawstyle = 'steps-post', color='tab:red')#
#ax.bar(pos['OrdinalDate'], pos['Importe'], width=10)
#ax.plot(df['OrdinalDate'], Yhat, color='tab:olive')
#ax.plot(df['OrdinalDate'], Yhat2, color='tab:gray')

plt.show()