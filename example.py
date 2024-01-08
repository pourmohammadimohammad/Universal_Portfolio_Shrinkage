import numpy as np
import pandas as pd
from upsa import UPSA


# I use the (cleaned) 153 long-short characteristic sorted portfolios from https://jkpfactors.com
factors = pd.read_pickle('bryan_monthly_vw_cap.p')


# train period
cut_off = int(len(factors)/2)
factors_ins = factors.iloc[:cut_off,:]

# define shrinkage grid
z_list = np.logspace(-10,-1,10)
upsa = UPSA(z_list = z_list).fit(returns=factors_ins)


# test period
factors_oos = factors.iloc[cut_off:,:]
ridge = factors_oos@upsa.get_ridge_weights()
CUPSA = factors_oos@upsa.get_cupsa_weights()
