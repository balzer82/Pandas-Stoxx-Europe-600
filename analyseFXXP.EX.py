# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=1>

# Python Pandas for Financial Stuff

# <codecell>

import datetime

import pandas as pd
import pandas.io.data
from pandas import Series, DataFrame
pd.__version__

import matplotlib.pyplot as plt
import numpy as np

%pylab inline

# <headingcell level=2>

# Stoxx-Europe-600-Index

# <markdowncell>

# Thanks to this:
# http://nbviewer.ipython.org/github/twiecki/financial-analysis-python-tutorial/blob/master/1.%20Pandas%20Basics.ipynb

# <codecell>

sc='FXXP.EX'
stoxx = pd.io.data.get_data_yahoo(sc, 
                                 start=datetime.datetime(2013, 1, 1))
stoxx.head(10)

# <headingcell level=2>

# Schlusspreis

# <codecell>

plt.figure(figsize=(9,3))
stoxx['Close'].plot();
plt.ylabel('\$')
plt.title('Closing Price %s' % sc);
plt.savefig('Closing-Price-FXXP.png',bbox_inches='tight', dpi=150)

# <headingcell level=2>

# Financial Stuff

# <headingcell level=3>

# Moving Average

# <codecell>

close_px = stoxx['Adj Close']
mad = 10
mavg = pd.ewma(close_px, mad)
plt.figure(figsize=(16,4))
mavg.plot();
plt.ylabel('\$')
plt.title('Exponentially Weighted Moving Average (%i Days) %s' % (mad, sc));

# <headingcell level=3>

# Returns

# <codecell>

stoxx['rets'] = close_px.pct_change()
plt.figure(figsize=(16,4))
stoxx.rets.plot();
plt.title('Returns %s' % sc);
plt.ylabel('\$');

# <headingcell level=3>

# Relative Strength Index

# <markdowncell>

# Source: http://stackoverflow.com/a/20527056 (with Error!!)
# 
# The RSI is not calcualted with the `rolling_mean` but with the exponentially weighted moving average `ewma`!

# <codecell>

# Get daily up or down
delta = stoxx['Close'].diff()

dUp, dDown = delta.copy( ), delta.copy( )
dUp[ dUp < 0 ] = 0
dDown[ dDown > 0 ] = 0

n=14
RolUp = pd.ewma( dUp, n)
RolDown = pd.ewma( dDown, n).abs()

RS = RolUp / RolDown
RSI = 100. - 100./(1.+RS)

plt.figure(figsize=(9,3))
RSI.plot();
plt.axhline(20, color='k', alpha=0.2)
plt.annotate('oversold',xy=(0.5, 0.3), xycoords='figure fraction', fontsize=20, alpha=0.4, ha='center')
plt.axhline(80, color='k', alpha=0.2)
plt.annotate('overbought',xy=(0.5, 0.8), xycoords='figure fraction', fontsize=20, alpha=0.4,ha='center')
plt.title('RSI %s (%i days)' % (sc, n));
plt.ylim([0,100]);
plt.ylabel('%');
plt.savefig('RSI-FXXP.png',bbox_inches='tight', dpi=150)

# <headingcell level=3>

# Monte Carlo Simulation

# <codecell>

SO=stoxx['Close'][-1] # letzter Preis
vol=np.std(stoxx['rets'])*np.sqrt(252) # Historical Volatility
r=0.025 # Constant Short Rate

K = SO*1.1 # 10% OTM Call Option
T = 1.0 # Maturity 1 Year

M=364
dt=T/M # Time Steps
I = 100 # Simulation Paths

# <codecell>

S=np.zeros((M+1,I))
S[0,:]=SO
for t in range(1, M+1):
    ran = np.random.standard_normal(I)
    S[t,:]=S[t-1,:] * np.exp((r-vol**2/2)*dt + vol*np.sqrt(dt)*ran)


MC=pd.DataFrame(data=S, index=pd.date_range(start=stoxx.index[-1], periods=M+1))

ax=MC.plot(alpha=0.2, color='k');
stoxx['Close'].plot(ax=ax);
plt.legend(['Monte Carlo Simulation']);
plt.ylabel('\$');
plt.savefig('Monte-Carlo-Simulation-FXXP.png',bbox_inches='tight', dpi=150)

# <headingcell level=3>

# Option Valuation

# <codecell>

VO=np.exp(-r*T)*np.sum(np.max(S[-1]-K,0))/I
print('Call Value %8.3f' % VO)

# <headingcell level=1>

# Vergleich

# <codecell>

df = pd.io.data.get_data_yahoo(['AAPL', 'FXXP.EX', 'GOOG', 'FDAX.EX', 'TSLA'], 
                               start=datetime.datetime(2013, 1, 1))['Adj Close']
df.head()

# <headingcell level=3>

# Returns

# <codecell>

rets = df.pct_change()

# <codecell>

fig=plt.figure(figsize=(6,6));
pd.scatter_matrix(rets, diagonal='kde', figsize=(10, 10));
plt.savefig('Return-Correlation-FXXP.png',bbox_inches='tight', dpi=150)

# <headingcell level=3>

# Korrelation der Returns

# <codecell>

corr = rets.corr()
corr

# <codecell>

plt.imshow(corr, cmap='YlGn', interpolation='none')
plt.colorbar()
plt.xticks(range(len(corr)), corr.columns)
plt.yticks(range(len(corr)), corr.columns);

# <codecell>

fig=plt.figure(figsize=(6,6))
plt.scatter(rets.mean(), rets.std(), s=50)
plt.xlabel('Expected returns')
plt.ylabel('Risk')
for label, x, y in zip(rets.columns, rets.mean(), rets.std()):
    plt.annotate(
        label, 
        xy = (x, y), xytext = (20, -20),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'w', alpha = 0.5),
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
    
plt.savefig('Risk-Return-FXXP.png',bbox_inches='tight', dpi=150)

# <codecell>

print('Done.')

# <codecell>


