# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 23:11:19 2024

@author: XYZW
"""

import QuantLib as ql
import black_scholes_pricers as bsp
import scipy.optimize as opt
import numpy as np
#import pandas as pd
#import xlwings as xw
#import yfinance as yf
#import datetime as dt
import sys
sys.path.append(r'C:\Users\XYZW\Documents\Python Scripts\equity exotics')
sys.path.append(r'C:\Users\XYZW\Documents\Python Scripts\Quantlib_tests')
#import option_price_BS as opt_BS
#import heston_calibrator as HC
#import collections 

#%%
def sabr_pricer(S0,rf,dvd,ref_date,T,strike,alpha,beta,nu,rho,
                option_type,exercise_type = "european",structure_type="vanilla",details = "no"):
    """
    Option Type: ql.Option.Call or ql.Option.Put
    
    Structure Type: must be vanilla/
    
    RETURNS:
        Value
        
        SABR vol
        
        Optionally: Delta + Gamma. 
    """
    if isinstance(option_type,str)==1 and str.lower(option_type)=="call":
        option_type = ql.Option.Call
    
    if isinstance(option_type,str)==1 and str.lower(option_type)=="put":
        option_type = ql.Option.Put
        
    sabr_black_engine = bsp.black_constant_SABRvol_engine(S0, rf, dvd, 
                                    ref_date, T, strike, alpha, beta, nu, rho)
    
    option1 = bsp.option(strike,T,ref_date,option_type,exercise_type = exercise_type,
                         structure = structure_type)
    option1.setPricingEngine(sabr_black_engine)
    forward = S0*np.exp(rf*T)
    
    if exercise_type == 'european' and details=="yes":
        sabr_vol = ql.sabrVolatility(strike,forward,T,alpha,beta,nu,rho)
        return option1.NPV(),sabr_vol,option1.delta(),option1.gamma()
    else:
        return option1.NPV()


def sabr_calibrator(S0,rf,T,strikes,mkt_prices,init_sol = [0.5]*4,dvd = 0.0,ref_date = None):
    """
    Given a smile of market prices, find the parameters of SABR model matching the market pricesof options.
    """
    if ref_date == None:
        ref_date = ql.Date().todaysDate()
    sabr_func = lambda z:[sabr_pricer(S0, rf,0, ref_date,T,x,z[0],z[1],z[2],z[3],"call")
                          for x in strikes]
    sabr_calib = lambda z:np.linalg.norm(np.array(sabr_func(z))-np.array(mkt_prices),ord = 2)
    if init_sol == [0.5]*4:
        init_sol = [0.2,0.9,0.1,0.5]
    bnds = [(0,1)]*3+[(-1,1)]
    final_sol = opt.minimize(sabr_calib,init_sol,bounds = bnds)
    params =  final_sol.x
    sabr_prices = sabr_func(params)
    rel_errors = sabr_prices/np.array(mkt_prices)-1
    return dict({'Parameters':params,'Sabr prices':sabr_prices,'Rel errors':rel_errors})    

def sabr_calibrator2(S0,rf,expiries,strikes,mkt_prices,init_sol = [0.5]*4,dvd = 0.0,ref_date = None,
                     bnd_corr = None):
    """
    Given an entire surface (strike x expiries) of market prices find the sabr 
    model parameters. 
    
    Parameters:
        ref_date: reference date
        
        dvd: dividend rate
        
        strikes: strike prices
        
        bnd_corr: bounds over correlation
    """
    if ref_date == None:
        ref_date = ql.Date().todaysDate()
    sabr_func = lambda z:[[sabr_pricer(S0, rf,0, ref_date,y,x,z[0],z[1],z[2],z[3],"call")
                          for y in expiries] for x in strikes]
    sabr_calib = lambda z:np.linalg.norm(np.array(sabr_func(z))-np.array(mkt_prices),ord = 2)
    if init_sol == [0.5]*4:
        init_sol = [0.2,0.9,0.1,0.5]
    if bnd_corr ==None:
        bnd_corr = [(-0.99,0.99)]
    bnds = [(0,1)]*3+bnd_corr
    final_sol = opt.minimize(sabr_calib,init_sol,bounds = bnds)
    params =  final_sol.x
    sabr_prices = sabr_func(params)
    rel_errors = sabr_prices/np.array(mkt_prices)-1
    return dict({'Parameters':params,'Sabr prices':sabr_prices,'Rel errors':rel_errors})    