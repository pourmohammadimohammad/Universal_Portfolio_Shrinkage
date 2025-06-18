import numpy as np
import pandas as pd
import pytest
from upsa.upsa import UPSA

@pytest.fixture
def sample_factors(tmp_path):
    # Simulate a small dataset: 20 observations × 5 portfolios
    df = pd.DataFrame(
        np.random.standard_normal((20, 5)),
        columns=[f"P{i}" for i in range(5)]
    )
    # Write to pickle to mirror real file flow
    path = tmp_path / "factors.pkl"
    df.to_pickle(path)
    return pd.read_pickle(path)


def test_upsa_basic(sample_factors):
    # Split into in-sample and out-of-sample
    cut = len(sample_factors) // 2
    ins = sample_factors.iloc[:cut]
    oos = sample_factors.iloc[cut:]

    # Fit with leave-one-out CV
    z_list = np.logspace(-3, 0, 5)
    model = UPSA(z_list=z_list).fit(ins, cv_method='loo')

    # Check weight vector shape
    w_ridge = model.get_ridge_weights()
    assert w_ridge.shape == (ins.shape[1],)

    # Apply to out-of-sample and compute Sharpe
    port = oos.values @ w_ridge
    sharpe = np.sqrt(12) * port.mean() / port.std()

    # Ensure Sharpe is a finite float
    assert isinstance(sharpe, float)
    assert np.isfinite(sharpe)


@pytest.mark.parametrize("folds", [2, 5])
def test_upsa_kfold(sample_factors, folds):
    # Split into in-sample
    cut = len(sample_factors) // 2
    ins = sample_factors.iloc[:cut]

    # Fit with k-fold CV
    model = UPSA(z_list=[0.1, 1.0, 10.0]).fit(ins, cv_method=folds)

    # Check weight vector shape remains correct
    w_ridge = model.get_ridge_weights()
    assert w_ridge.shape == (ins.shape[1],)
