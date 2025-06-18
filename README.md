# Universal Portfolio Shrinkage (UPSA)

A Python implementation of the **Universal Portfolio Shrinkage Approximator (UPSA)**, a flexible spectral shrinkage method that directly optimizes out-of-sample portfolio performance, as introduced in:

> Kelly, Bryan T., Semyon Malamud, Mohammad Pourmohammadi, and Fabio Trojani. Universal portfolio shrinkage. No. w32004. National Bureau of Economic Research, 2024.

## Motivation

Classical Markowitz portfolios suffer from severe estimation noise when the number of assets or factors (N) is large relative to sample size (T), leading to large gaps between in-sample and out-of-sample performance . Traditional shrinkage methods impose restrictive forms or optimize statistical proxies rather than the portfolio objective, limiting efficacy. UPSA overcomes these limitations by providing a universal spectral approximator for nonlinear shrinkage functions and tuning shrinkage directly on expected out-of-sample performance via cross-validation.

## Key Features

* **Universal spectral approximation**: Represents a broad class of nonlinear shrinkage functions as a positive linear combination of basic ridge shrinkages, based on a Stone–Weierstrass argument (Lemma 1).
* **Objective-aligned tuning**: Chooses shrinkage weights by maximizing expected out-of-sample quadratic utility using leave-one-out or k-fold CV (Lemma 2).
* **Efficient closed-form computation**: Employs spectral formulas to compute LOO estimators for ridge portfolios without refitting eigen-decomposition for each leave-out.
* **Constraint support**: Optional nonnegativity and sum-to-one enforcement on UPSA ensemble weights for economic interpretability.
* **Flexible CV interface**: Single `cv_method` parameter accepts `'loo'` or integer >1 for k-fold CV.
* **Scalable**: Handles high-dimensional settings (N ≫ T or T ≫ N) via efficient eigen-decomposition routines.
* **Empirical robustness**: Outperforms ridge, Ledoit–Wolf, PCA-based, and benchmark factor models in anomaly portfolio tests, achieving higher Sharpe and lower pricing errors.

## Installation

Requires Python 3.7+ and dependencies: `numpy`, `scikit-learn`, `cvxopt`, `pandas` (optional).

```bash
# Install from PyPI
to=python
pip install universal-upsa
# Or install development version
git clone https://github.com/yourusername/universal-upsa.git
cd universal-upsa
pip install -e .
```

## Quickstart

```python
import numpy as np
import pandas as pd
from upsa.upsa import UPSA

# 1) Load returns (T×P) as DataFrame or ndarray
returns_df = pd.read_csv("returns.csv", index_col="date")
returns = returns_df.values

# 2) Define shrinkage grid (e.g., logspace spanning empirical eigenvalues)
z_list = np.logspace(-4, 2, 20)

# 3a) Fit with leave-one-out CV (default)
model_loo = UPSA(z_list=z_list).fit(returns, cv_method='loo', constraint=False)
w_loo = model_loo.get_upsa_weights()

# 3b) Or fit with 5-fold CV for larger T
model_kf = UPSA(z_list=z_list).fit(returns, cv_method=5, constraint=False)
w_kf = model_kf.get_upsa_weights()

# 4) Apply out-of-sample: compute portfolio returns
oos_df = pd.read_csv("oos_returns.csv", index_col="date")
oos = oos_df.values
port_ret = oos @ w_loo

# 5) Annualized Sharpe ratio
sharpe = np.sqrt(12) * port_ret.mean() / port_ret.std()
print(f"Annualized Sharpe: {sharpe:.3f}")
```

To enforce economic constraints:

```python
model_c = UPSA(z_list=z_list).fit(returns, cv_method='loo', constraint=True)
w_c = model_c.get_upsa_weights()
```

## API Reference

* **`UPSA(z_list: np.ndarray = None)`**: Initialize with optional shrinkage grid. If `None`, defaults to log-spaced grid spanning smallest to largest empirical eigenvalue fileciteturn3file12.
* **`.fit(returns, cv_method='loo', constraint=False)`**: Fit UPSA model.

  * `returns`: T×P array or DataFrame.
  * `cv_method`: `'loo'` or integer >1 for k-fold.
  * `constraint`: if `True`, enforce nonnegativity and sum-to-one on UPSA weights.
    Returns fitted instance with attributes `best_z`, `upsa`, and `eff_port` (matrix of ridge portfolios).
* **`.get_upsa_weights() -> np.ndarray`**: Returns UPSA portfolio weight vector (length P).
* **`.get_ridge_weights() -> np.ndarray`**: Returns ridge portfolio weights vector (length P).

## Empirical Evidence Summary

On 153 anomaly portfolios from Jensen et al. (1971–2022), UPSA achieves out-of-sample Sharpe ≈ 1.92 vs. 1.59 for best ridge, 1.31 for Ledoit–Wolf, and 1.45 for PCA-based portfolios; Cross sectional R² ≈ 66% vs. 39% for ridge.
Robustness holds across subsamples and additional regularization scenarios.

## Testing

Run basic sanity tests with pytest:

```bash
pip install pytest
pytest tests/test_upsa.py
```

## Citation

Please cite the paper when using UPSA:

```bibtex
@article{kelly2025universal,
  title={Universal portfolio shrinkage},
  author={Kelly, Bryan T and Malamud, Semyon and Pourmohammadi, Mohammad and Trojani, Fabio},
  year={2024},
  institution={National Bureau of Economic Research}
}
```

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please open issues or pull requests on GitHub: [https://github.com/yourusername/universal-upsa](https://github.com/yourusername/universal-upsa).
