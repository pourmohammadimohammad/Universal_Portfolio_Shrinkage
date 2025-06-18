import numpy as np
import pandas as pd
from upsa.upsa import UPSA

# Load your 153 long–short characteristic portfolios (from jkpfactors.com)
factors = pd.read_pickle("jkp_monthly_vw_cap.p")

# Split into in-sample and out-of-sample halves
n_obs = len(factors)
cut_off = n_obs // 2
factors_ins = factors.iloc[:cut_off]
factors_oos = factors.iloc[cut_off:]

# Define a shrinkage grid
z_list = np.logspace(-10, -1, 10)

# 1) Fit with leave-one-out CV
model_loo = UPSA(z_list=z_list).fit(returns=factors_ins, cv_method="loo")
weights_loo = model_loo.get_ridge_weights()

# 2) (Alternatively) fit with 5-fold CV
model_kf = UPSA(z_list=z_list).fit(returns=factors_ins, cv_method=5)
weights_kf = model_kf.get_ridge_weights()

# Compute out-of-sample portfolio returns
port_ret_loo = factors_oos.values @ weights_loo
port_ret_kf  = factors_oos.values @ weights_kf

# Annualized Sharpe ratios
sharpe_loo = np.sqrt(12) * port_ret_loo.mean() / port_ret_loo.std()
sharpe_kf  = np.sqrt(12) * port_ret_kf.mean()  / port_ret_kf.std()

print(f"LOO Sharpe: {sharpe_loo:.3f}")
print(f"5-fold Sharpe: {sharpe_kf:.3f}")
