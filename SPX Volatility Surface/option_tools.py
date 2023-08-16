# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 11:17:39 2022

@author: Hogan
"""

import numpy as np
from scipy.stats import norm
import math
from scipy.optimize import minimize_scalar   
import pandas as pd
from datetime import datetime, timedelta
from scipy.optimize import minimize_scalar
from scipy.interpolate import CubicSpline, interp1d
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import smoother_tools
import numpy.polynomial.polynomial as poly
from scipy.interpolate import lagrange

#%% Black Scholes Formula

def black_scholes_call(S, K, T, r, sigma, q=0, F=None):
    """
    Parameters
    ----------
    S : float
        Spot price.
    K : float
        Strike price.
    T : float
        Time to expiration, measured in years.
    r : float
        Risk-free interest rate.
    sigma : float
        Volatility, measured in annually standard deviation.
    q : float, optional
        Dividend yield. The default is 0.
    F : float, optional
        Forward price. If a forward price is provided, the spot price will be overriden and the model will be converted to the Black 76 Model. The default is None.

    Returns
    -------
    float
        Option price.
    """
    N = norm.cdf
    if F is None:
        d1 = (np.log(S/K) + (r - q + sigma**2/2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S*np.exp(-q*T)*N(d1) - K*np.exp(-r*T)*N(d2)
    else:
        d1 = (np.log(F/K) + sigma**2/2*T) / (sigma*np.sqrt(T))
        d2 = d1 - (sigma*np.sqrt(T))
        return F*np.exp(-r*T)*N(d1) - K*np.exp(-r*T)*N(d2)
        
def black_scholes_put(S, K, T, r, sigma, q=0, F=None):
    "Parameters same as black_scholes_call"
    N = norm.cdf
    if F is None:
        d1 = (np.log(S/K) + (r - q + sigma**2/2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma* np.sqrt(T)
        return K*np.exp(-r*T)*N(-d2) - S*np.exp(-q*T)*N(-d1)
    else:
        d1 = (np.log(F/K) + sigma**2/2*T) / (sigma*np.sqrt(T))
        d2 = d1 - (sigma*np.sqrt(T))
        return K*np.exp(-r*T)*N(-d2) - F*np.exp(-r*T)*N(-d1)

def implied_volatility(option_mkt_price, S, K, T, r, q=0, F=None, option_type='call'):
    
    def call_obj(implied_sigma):
        return abs(black_scholes_call(S, K, T, r, implied_sigma, q, F) - option_mkt_price)
    def put_obj(implied_sigma):
        return abs(black_scholes_put(S, K, T, r, implied_sigma, q, F) - option_mkt_price)
        
    if option_type == 'call':
        result = minimize_scalar(call_obj, bounds=(0.001,6), method='bounded')
        return result.x
    elif option_type == 'put':
        result = minimize_scalar(put_obj, bounds=(0.001,6), method='bounded')
        return result.x
    else:
        raise ValueError("Option type must be 'call' or 'put'")

def implied_forward_price(C, P, K, T, r):
    # Put call parity: C + K*e^(-rT) = S*e^(-qT)
    # Forward price: F = S*e^((r-q)T)
    return (C - P)*np.exp(r*T) + K

#%% OOP Implementation

class BSEuropeanOption():
    """
    S: Spot price
    K: Strike price
    T: Time to maturity, measured in years
    r: Risk-free interest rate
    sigma: Volatility, measured in standard deviation per annum
    q: Dividend yield, set to 0 for default
    F: Forward price. If a forward price is provided, we will use the Black 76 Model instead of the classic Black-Scholes Model, and the spot price S will be overriden in all calculations.
    C: Market price of call option, if any
    P: Market price of put option, if any
    """
    def __init__(self, S, K, T, r, sigma, q=0.0, F=None, C=None, P=None):
        self.K = K
        self.T = T
        self.r = r 
        self.S = S
        self.q = q
        self.F = F
        self.sigma = sigma
        self.C = C
        self.P = P
        
        self.d1 = self._d1()
        self.d2 = self._d2()
        self.get_greeks()
        self._implied_volatility()
        if F is None:
            self.moneyness = K/S
        else:
            self.moneyness = K/F        
        
    def N(self,x): # C.D.F. of standard normal distribution
        return norm.cdf(x)
    
    def n(self,x): # P.D.F. of standard normal distribution
        return norm.pdf(x)
    
    def _d1(self):
        if self.F is None:
            self.d1 = (np.log(self.S/self.K) + (self.r -self.q + self.sigma**2/2)*self.T) / (self.sigma*np.sqrt(self.T))
        else:
            self.d1 = (np.log(self.F/self.K) + self.sigma**2/2*self.T) / (self.sigma*np.sqrt(self.T))
        return self.d1
    
    def _d2(self):
        self.d2 = self._d1() - self.sigma* np.sqrt(self.T)
        return self.d2
    
    def _delta(self):
        if self.F is None:
            self.delta = {'call':np.exp(-self.q*self.T)*self.N(self._d1()),
                          'put':-np.exp(-self.q*self.T)*self.N(-self._d1())}
        else:
            self.delta = {'call':np.exp(-self.r*self.T)*self.N(self._d1()),
                          'put':-np.exp(-self.r*self.T)*self.N(-self._d1())}              
        return self.delta
    
    def _gamma(self):
        if self.F is None:
            self.gamma = np.exp(-self.q*self.T)*self.n(self._d1()) / (self.S*self.sigma*np.sqrt(self.T))
        else:
            self.gamma = np.exp(-self.r*self.T)*self.n(self._d1()) / (self.F*self.sigma*np.sqrt(self.T))
        return self.gamma
    
    def _vega(self):
        if self.F is None:
            self.vega = self.S*np.exp(-self.q*self.T)*self.n(self._d1())*np.sqrt(self.T)
        else:
            self.vega = self.F*np.exp(-self.r*self.T)*self.n(self._d1())*np.sqrt(self.T)
        return self.vega
    
    def _theta(self):
        if self.F is None:
            p1 = -np.exp(-self.q*self.T)*self.S*self.n(self._d1())*self.sigma / (2*np.sqrt(self.T))
            p2 = -self.r*self.K*np.exp(-self.r*self.T)*self.N(self._d2()) + self.q*self.S*np.exp(-self.q*self.T)*self.N(self._d1())
            p3 = self.r*self.K*np.exp(-self.r*self.T)*self.N(-self._d2()) - self.q*self.S*np.exp(-self.q*self.T)*self.N(-self._d1())
        
        else:
            p1 = -self.F*np.exp(-self.r*self.T)*self.n(self._d1())*self.sigma/(2*np.sqrt(self.T))
            p2 = -self.r*self.K*np.exp(-self.r*self.T)*self.N(self._d2()) + self.r*self.F*np.exp(-self.r*self.T)*self.N(self._d1())
            p3 = self.r*self.K*np.exp(-self.r*self.T)*self.N(-self._d2()) - self.r*self.F*np.exp(-self.r*self.T)*self.N(-self._d1())
            
        self.theta = {'call':p1+p2,'put':p1+p3}
        
        return self.theta
    
    def _rho(self):
        if self.F is None:
            call_rho = self.K*self.T*np.exp(-self.r*self.T)*self.N(self._d2())
            put_rho = -self.K*self.T*np.exp(-self.r*self.T)*self.N(-self._d2())
        else:
            call_rho = -self.T*np.exp(-self.r*self.T) * (self.F*self.N(self._d1()) - self.K*self.N(self._d2()))
            put_rho = -self.T*np.exp(-self.r*self.T) * (self.K*self.N(-self._d2()) - self.F*self.N(-self._d1()))
        self.rho = {'call':call_rho,'put':put_rho}
        return self.rho
    
    def get_greeks(self):
        self.delta = self._delta()
        self.gamma = self._gamma()
        self.vega = self._vega()
        self.theta = self._theta()
        self.rho = self._rho()
        return
    
    def _call_value(self):
        return black_scholes_call(self.S, self.K, self.T, self.r, self.sigma, self.q, self.F)
    
    def _put_value(self): 
        return black_scholes_put(self.S, self.K, self.T, self.r, self.sigma, self.q, self.F)
    
    def _implied_volatility(self):
        implied_vol_call, implied_vol_put = np.nan, np.nan 
        if self.C is not None:
            implied_vol_call = implied_volatility(self.C, self.S, self.K, self.T, self.r, self.q, self.F, 'call')
        if self.P is not None:
            implied_vol_put = implied_volatility(self.P, self.S, self.K, self.T, self.r, self.q, self.F, 'put')
        self.implied_volatility = {'call':implied_vol_call,
                                   'put':implied_vol_put}
        return self.implied_volatility
    
    def summary(self, precision=5):
        info = pd.DataFrame(data=np.nan, index=['BS Price','Delta','Gamma','Vega','Rho','Theta','IV'], columns=['Call Option','Put Option'])
        info.loc['BS Price'] = [np.round(self._call_value(),precision),np.round(self._put_value(),precision)]
        info.loc['Delta'] = [np.round(self.delta['call'],precision),np.round(self.delta['put'],precision)]
        info.loc['Gamma'] = [np.round(self.gamma,precision),np.round(self.gamma,precision)]
        info.loc['Vega'] = [np.round(self.vega,precision),np.round(self.vega,precision)]
        info.loc['Theta'] = [np.round(self.theta['call'],precision),np.round(self.theta['put'],precision)]
        info.loc['Rho'] = [np.round(self.rho['call'],precision),np.round(self.rho['put'],precision)]
        info.loc['IV'] = [np.round(self.implied_volatility['call'],precision),np.round(self.implied_volatility['put'],precision)]
        print(info)
        return
    

#%% Volatility Surface

def option_dict_to_df(option_universe):
    """
    Parameters
    ----------
    option_universe : dictioinary
        A dictionary of BSEuropeanOption() objects.

    Returns
    -------
    res : pd.DataFrame
        Each row corresponds to BSEuropeanOption() object.
    """
    res = pd.DataFrame(columns=['Name','T','S','K','r','q','Moneyness','Call Mkt Price','Put Mkt Price','Call Delta','Put Delta','Gamma','Vega','Call Theta','Put Theta','Call Rho','Put Rho','Call IV','Put IV'])
    for key, item in option_universe.items():
        row = len(res.index)
        res.loc[row] = [key, item.T, item.S, item.K, item.r, item.q, item.moneyness, item.C, item.P, item.delta['call'], item.delta['put'], item.gamma, item.vega, item.theta['call'], item.theta['put'], item.rho['call'], item.rho['put'], item.implied_volatility['call'], item.implied_volatility['put']]
    return res

def option_df_to_dict(option_df): # The reverse of option_dict_to_df()
    option_dict = {}
    for i in range(len(option_df.index)):
        name = option_df['Name'].iloc[i]
        S = option_df['S'].iloc[i]
        K = option_df['K'].iloc[i]
        T = option_df['T'].iloc[i]
        r = option_df['r'].iloc[i]
        q = option_df['q'].iloc[i]
        C = option_df['Call Mkt Price'].iloc[i]
        P = option_df['Put Mkt Price'].iloc[i]
        F = implied_forward_price(C, P, K, T, r) 
        sigma = implied_volatility(C, S, K, T, r, q, F, 'call')
        option_dict[name] = BSEuropeanOption(S, K, T, r, sigma, q, F, C, P)
    return option_dict


def volatility_surface_interpolation(volatility_surface, strikes, tenors, smooth=False):
    """
    Parameters
    ----------
    volatility_surface : pd.DataFrame
        index=strike(moneyness or delta), columns=tenor, value=IV.
    strikes : list
        A list of strikes to be interpolated.
    tenors : list
        A list of tenors to be interpolated.

    Returns
    -------
    interpolated_surface : pd.DataFrame
        The interpolated surface with strikes on index and tenors on columns.
    """
    
    "Step 1: Interpolate stikes with cubic spline interplolation"
    # Strikes along the iso-tenor lines are interpolated with cublic spline.
    step1 = pd.DataFrame(data=np.nan, index=strikes, columns=volatility_surface.columns)
    for j in range(len(step1.columns)):
        temp = volatility_surface.iloc[:,j].dropna() 
        interpolation = CubicSpline(temp.index, temp.values)
        interpolated_values = pd.Series(interpolation(strikes), index=strikes)
        # Warning: If the strikes' range is larger than the data, the cubic spline interpolater will be extrapolating and this will lead to serious errors. 
        # To avoid such errors, we simply delete anything outside the data's range. 
        # Ideally, the option universe should be large enough to cover every strike and tenor, but sometimes the more extreme tenors or strikes are not actively traded.
        # Extraploation will be done in step two with tenor.
        interpolated_values[interpolated_values.index>max(temp.index)]=np.nan
        interpolated_values[interpolated_values.index<min(temp.index)]=np.nan   
        step1.iloc[:,j] = interpolated_values
            
    "Step 2: Interpolate on implied variance along iso-strike lines"
    # Tenors are interpolated using linear interpolation on T*sigma^2
    interpolated_surface = pd.DataFrame(data=np.nan, index=step1.index, columns=tenors)
    for i in range(len(step1.index)):
        variances = [0] + list(step1.iloc[i]**2*step1.columns) # adding T=0 to extrapolate
        t = [0] + list(step1.columns)
        temp = pd.Series(variances, index=t).dropna()
        interpolation = interp1d(temp.index, temp.values)
        interpolated_surface.iloc[i] = interpolation(tenors)/tenors
        interpolated_surface.iloc[i] = interpolated_surface.iloc[i].apply(lambda x: math.sqrt(x))
    
    "Step 3: Smoothing"
    if smooth:
        smoother = smoother_tools.CubicSplineSmoother(n_knots=int(len(interpolated_surface.index)/4))
        for j in range(len(tenors)):
            smoother.fit(pd.Series(interpolated_surface.index), interpolated_surface.iloc[:,j])
            interpolated_surface.iloc[:,j] = smoother.Y_smoothed.values
    
    return interpolated_surface

def plot_volatility_surface(volatility_surface, figure_size=[16,12], x_label=None, y_label=None, z_label=None, title=None):
    plt.style.use('seaborn')
    fig = plt.figure(figsize = figure_size)
    ax = plt.axes(projection ='3d')
 
    # Creating color map
    my_cmap = plt.get_cmap('hot')
 
    # Creating plot
    temp = volatility_surface.copy()
    temp['Strike']=temp.index
    temp=temp.melt('Strike', var_name='T', value_name='IV')
    temp = temp.dropna(axis=0)
    
    surf = ax.plot_trisurf(temp['Strike'], temp['T'], temp['IV'], cmap = my_cmap, edgecolor ='none')
 
    fig.colorbar(surf, ax = ax, shrink = 0.5, aspect = 5)
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)
    ax.set_title(title)
    
    plt.show()
    return fig


def build_volatility_surface(option_universe, 
                             tenors=[1/12, 2/12, 3/12, 6/12, 1, 1.5, 2], 
                             moneyness=[np.round(0.6+0.05*i,2) for i in range(17)], 
                             deltas=[np.round(0.05*i,2) for i in range(1,20)],
                             smooth=True,
                             plot=True, figure_size=(16,16)):
    """
    Parameters
    ----------
    option_universe : dictionary
        A dictionary of BSEuropeanOption objects.
    tenors : list, optional
        A list of tenors measured in years. The default is [1/12, 2/12, 3/12, 6/12, 1, 2].
    moneyness : list, optional
        A list of moneyness, i.e. K/S or K/F. The default is [np.round(0.6+0.05*i,2) for i in range(17)].
    deltas : list, optional
        A list of deltas. The default is [np.round(0.05*i,2) for i in range(1,20)].
    plot : Boolean, optional
        Set to false if you do not want to plot the surface. The default is True.
    figure_size : tuple, optional
        Size of the chart. The default is (16,16).

    Returns
    -------
    surface1 : pd.DataFrame
        index=moneyness, columns=tenor, value=IV.
    surface2 : pd.DataFrame
        index=delta, columns=tenor, value=IV.
    """
    
    option_df = option_dict_to_df(option_universe)
    option_df['IV'] = (option_df['Call IV'] + option_df['Put IV'])/2
    
    "Surface 1: Tenor and Moneyness"
    if tenors is None or moneyness is None:
        surface1=None
    else:
        IV_pivot_table = option_df.pivot_table(index="Moneyness",columns="T",values="IV",aggfunc='median')
        surface1 = volatility_surface_interpolation(IV_pivot_table, moneyness, tenors, smooth)
        
    "Surface 2: Tenor and Delta"
    if tenors is None or deltas is None:
        surface2=None
    else:
        IV_pivot_table = option_df.pivot_table(index="Call Delta",columns="T",values="IV",aggfunc='median')
        surface2 = volatility_surface_interpolation(IV_pivot_table, deltas, tenors, smooth)
    
    
    "Plot the volatility surfaces"
    if plot:
        if surface1 is not None:
            plot_volatility_surface(surface1, x_label='Moneyness', y_label='Tenor', z_label='IV', title='Volatility Surface')
        if surface2 is not None:
            plot_volatility_surface(surface2, x_label='Delta', y_label='Tenor', z_label='IV', title='Volatility Surface')
        
    return surface1, surface2


#%% No-Arbitrage Smoothing

def arbitrage_free_smoothing(option_universe):
    option_df = option_dict_to_df(option_universe)
    tenors = option_df['T'].unique()
    for t in tenors:
        subset = option_df[option_df['T']==t]
        smoother = smoother_tools.CubicSplineSmoother(n_knots=10)
            
        smoother.fit(subset['K'], subset['Call Mkt Price'])
        subset['Call Mkt Price'] = smoother.Y_smoothed.apply(lambda x: max(x,0)).values
        
        smoother.fit(subset['K'], subset['Put Mkt Price'])
        subset['Put Mkt Price'] = smoother.Y_smoothed.apply(lambda x: max(x,0)).values
        
        option_df.iloc[subset.index] = subset
        
    return option_df_to_dict(option_df)


#%% Binomial Tree

def combos(n, i):
    return math.factorial(n) / (math.factorial(n-i)*math.factorial(i))

def binom_EU1(S0, K , T, r, sigma, N, type_ = 'call'):
    dt = T/N
    u = np.exp(sigma * np.sqrt(dt))
    d = np.exp(-sigma * np.sqrt(dt))
    p = (np.exp(r*dt) - d)/(u - d)
    value = 0 
    for i in range(N+1):
        node_prob = combos(N, i)*p**i*(1-p)**(N-i)
        ST = S0*(u)**i*(d)**(N-i)
        if type_ == 'call':
            value += max(ST-K,0) * node_prob
        elif type_ == 'put':
            value += max(K-ST, 0)*node_prob
        else:
            raise ValueError("type_ must be 'call' or 'put'" )
    
    return value*np.exp(-r*T)


#%% Read CBOE Data Source

def read_cboe_quote_data(cboe_file='spx_quotedata.csv', skiprows=3):
    quotes = pd.read_csv(cboe_file, skiprows=skiprows)
    quotes['Expiration Date']=quotes['Expiration Date'].apply(lambda x: datetime.strptime(x.split(' ',1)[1],'%b %d %Y'))
    return quotes

#%% Connect to CBOE API

cboe_client_id = "hogan.tong@hotmail.com_api_client_1642256761"
cboe_client_secret = "5465fa2d218f414eb7de677e7524ada4"

def connect_to_cboe_api(client_id, client_secret):
    pass
    return



#%% Save Files

def save_option_universe_to_csv(option_universe, path='option_universe.csv'):
    option_df = option_dict_to_df(option_universe)
    option_df.to_csv(path)
    return

