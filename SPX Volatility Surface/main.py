# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 11:18:40 2022

@author: Hogan
"""

import pandas as pd
import numpy as np
from datetime import datetime
import option_tools
import quandl # To get treasury yield curve
import math
from scipy.interpolate import CubicSpline, interp1d
import requests
import base64
import warnings
warnings.filterwarnings("ignore")

SPX_current_price = 4662.85
SPX_dividend_yield = 0.015
t0 = datetime(2022,1,14)

treasury_yield_data=quandl.get("USTREASURY/YIELD", authtoken="7NpQ9-AyBVRYkT2Zs5WM")/100
yield_curve = CubicSpline([1/12,2/12,3/12,6/12,1,2,3,5,7,10,20,30], treasury_yield_data.loc[t0])

# Alternatively, download the data at: https://data.nasdaq.com/data/USTREASURY/YIELD-treasury-yield-curve-rates
# yield_curve = CubicSpline([1/12,2/12,3/12,6/12,1,2,3,5,7,10,20,30], pd.read_csv('USTREASURY-YIELD.csv').iloc[0,1:].values/100)

quotes = option_tools.read_cboe_quote_data(cboe_file='spx_quotedata.csv')



#%% Some Sanity Tests
# http://www.option-price.com/implied-volatility.php
# https://www.math.drexel.edu/~pg/fin/VanillaCalculator.html

S=100
K=100
T=1
r=0.05
sigma=0.2
q=0.01
F = S*np.exp((r-q)*T)
C=3.5
P=2.2

test = option_tools.BSEuropeanOption(S=S, K=K, T=T, r=r, sigma=sigma, q=q, C=C, P=P)
test.summary()

test = option_tools.BSEuropeanOption(S=S, K=K, T=T, r=r, sigma=sigma, q=q, C=C, P=P, F=F)
test.summary()



#%% Traversing the quotes to get the option universe
option_universe={}

for i in range(len(quotes)):
    
    T = (quotes['Expiration Date'].iloc[i]-t0).days / 365
    r = yield_curve(T)
    K = quotes['Strike'].iloc[i]
    S = SPX_current_price
    C = (quotes['Bid'].iloc[i] + quotes['Ask'].iloc[i])/2
    P = (quotes['Bid.1'].iloc[i] + quotes['Ask.1'].iloc[i])/2
    
    if (T==0):# or (quotes['Bid'].iloc[i]==0) or (quotes['Bid.1'].iloc[i]==0):
        continue
    
    F = option_tools.implied_forward_price(C, P, K, T, r)    
    sigma = option_tools.implied_volatility(C, S, K, T, r, SPX_dividend_yield, F, 'call')
    
    option_universe[quotes['Expiration Date'].iloc[i].strftime('%y%m%d')+'-'+str(int(K))] = option_tools.BSEuropeanOption(S, K, T, r, sigma, SPX_dividend_yield, F, C, P)

#%% Build Volatility Surface

surface_moneyness, surface_delta = option_tools.build_volatility_surface(option_universe, smooth=False)
surface_moneyness, surface_delta = option_tools.build_volatility_surface(option_universe, smooth=True)

#%% Arbitrage-free Smoothing (Optional)

#arbitrage_free_option_universe = option_tools.arbitrage_free_smoothing(option_universe)
#surface_moneyness, surface_delta = option_tools.build_volatility_surface(arbitrage_free_option_universe) 


#%% Export Outputs
option_tools.save_option_universe_to_csv(option_universe)

surface_moneyness.columns = map(int, surface_moneyness.columns*12)
surface_delta.columns = map(int, surface_delta.columns*12)

Excelwriter = pd.ExcelWriter("volatility surface.xlsx",engine="xlsxwriter")
surface_moneyness.to_excel(Excelwriter, sheet_name='moneyness')  
surface_delta.to_excel(Excelwriter, sheet_name='delta')
Excelwriter.save()