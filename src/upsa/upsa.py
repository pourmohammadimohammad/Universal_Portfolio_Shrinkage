import numpy as np
from sklearn.model_selection import KFold
from cvxopt import matrix, solvers

class UPSA:
    def __init__(self, z_list: np.ndarray = None):
        self.z_list = z_list
        self.eff_port = None
        self.best_z = None
        self.upsa = None

    def fit(self,
            returns,
            constraint: bool = False,
            cv_method='loo'):
        """
        Fit UPSA model using either LOO or k-fold cross‐validation.
        Accepts NumPy arrays or pandas DataFrames for returns.

        :param returns:   (T x P) asset returns array or DataFrame
        :param constraint: enforce positivity + sum-to-1 in Markowitz
        :param cv_method: 'loo' for leave-one-out or integer >1 for k-fold
        """
        # coerce to NumPy array if pandas DataFrame
        try:
            returns = returns.values
        except AttributeError:
            returns = np.asarray(returns)

        eigval, eigvec = UPSA._eigen_decomposition(returns)

        # default z_list based on eigenvalues
        if self.z_list is None:
            self.z_list = np.logspace(
                np.log10(eigval.min()),
                np.log10(eigval.max()),
                10
            )

        # choose CV routine
        if isinstance(cv_method, int) and cv_method > 1:
            folds = cv_method
            cv_rets, full_ports = self._kfold_and_eff_port_estimators(
                returns, eigval, eigvec, folds
            )
        else:
            cv_rets, full_ports = self._loo_and_eff_port_estimators(
                returns, eigval, eigvec
            )

        self.eff_port = full_ports
        self.best_z, self.upsa = UPSA._compute_upsa(cv_rets, constraint)
        return self

    @staticmethod
    def _compute_upsa(oos_rets: np.ndarray, constraint: bool = False):
        means = oos_rets.mean(axis=0)
        cov = (oos_rets.T @ oos_rets) / oos_rets.shape[0]

        upsa = UPSA._markowitz_constrained(means, cov, constraint)
        best_z = int(np.argmax(means / np.sqrt(np.diag(cov))))
        return best_z, upsa

    def get_upsa_weights(self) -> np.ndarray:
        return self.eff_port @ self.upsa

    def get_ridge_weights(self) -> np.ndarray:
        return self.eff_port[:, self.best_z]

    def _loo_and_eff_port_estimators(self,
                                     returns: np.ndarray,
                                     eigval: np.ndarray,
                                     eigvec: np.ndarray):
        full_ports = self._compute_ridge_eff_portfolios(returns, eigval, eigvec)
        smoothers = self._compute_smoothers(returns, eigval, eigvec)
        loo_rets, _ = self._calculate_loo_returns(smoothers)
        return loo_rets, full_ports

    def _kfold_and_eff_port_estimators(self,
                                       returns: np.ndarray,
                                       eigval: np.ndarray,
                                       eigvec: np.ndarray,
                                       k_folds: int):
        T, _ = returns.shape
        cv_rets = np.zeros((T, len(self.z_list)))
        kf = KFold(n_splits=k_folds, shuffle=False)

        for train_idx, test_idx in kf.split(returns):
            r_train = returns[train_idx, :]
            r_test = returns[test_idx, :]
            ev_train, evec_train = UPSA._eigen_decomposition(r_train)
            ports_train = self._compute_ridge_eff_portfolios(r_train, ev_train, evec_train)
            cv_rets[test_idx, :] = r_test @ ports_train

        full_ports = self._compute_ridge_eff_portfolios(returns, eigval, eigvec)
        return cv_rets, full_ports

    def _compute_ridge_eff_portfolios(self,
                                      returns: np.ndarray,
                                      eigval: np.ndarray,
                                      eigvec: np.ndarray) -> np.ndarray:
        mu = returns.mean(axis=0)
        ridge_list = []
        for z in self.z_list:
            denom = eigval + z
            w_z = (eigvec * (1.0 / denom)) @ (eigvec.T @ mu)
            ridge_list.append(w_z)
        return np.column_stack(ridge_list)

    def _compute_smoothers(self, returns, eigval, eigvec):
        T, _ = returns.shape
        proj_R = eigvec.T @ returns.T
        return [
            proj_R.T @ ((1/(eigval.reshape(-1,1)+z)) * proj_R) / T
            for z in self.z_list
        ]

    def _calculate_loo_returns(self, smoothers, w_diag=None):
        if w_diag is None:
            w_diag = [np.diag(S) for S in smoothers]
        w_bar = [S.sum(axis=1) for S in smoothers]
        loo = [(w_bar[i] - w_diag[i]) / (1 - w_diag[i]) for i in range(len(smoothers))]
        return np.array(loo).T, w_diag

    @staticmethod
    def _eigen_decomposition(features: np.ndarray,
                             kill_eig: bool = False):
        T, P = features.shape
        cov = (features @ features.T if P > T else features.T @ features) / T
        eigval, eigvec = np.linalg.eigh(cov)
        if kill_eig:
            mask = eigval > 1e-10
            eigval, eigvec = eigval[mask], eigvec[:, mask]
        if P > T:
            eigvec = features.T @ eigvec * (eigval**(-0.5)).reshape(1, -1) / np.sqrt(T)
        return eigval, eigvec

    @staticmethod
    def _markowitz_constrained(mean: np.ndarray,
                               cov: np.ndarray,
                               constraint: bool = False) -> np.ndarray:
        solvers.options['show_progress'] = False
        G = matrix(-np.eye(len(mean)))
        h = matrix(np.zeros((len(mean), 1)))
        q = matrix(-mean.astype(np.double))
        P = matrix(cov.astype(np.double))
        if constraint:
            A = matrix(np.ones((1, len(mean))))
            b = matrix(1.0)
            sol = solvers.qp(P, q, G, h, A, b)
        else:
            sol = solvers.qp(P, q, G, h)
        return np.array(sol['x']).flatten()
