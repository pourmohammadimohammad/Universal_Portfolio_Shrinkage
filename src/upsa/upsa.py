import numpy as np
from sklearn.model_selection import KFold
from cvxopt import matrix, solvers

class UPSA:
    def __init__(self, z_list: np.ndarray = None):
        # Initialize UPSA with optional list of shrinkage parameters (z_list)
        self.z_list = z_list
        self.eff_port = None  # will hold efficient portfolios for each z
        self.best_z = None    # index of best shrinkage parameter
        self.upsa = None      # final UPSA weights
        self.eigenvalues = None
        self.eigenvectors = None

    def fit(self,
            returns,
            constraint: bool = False,
            cv_method='loo',
            ):
        """
        Fit UPSA model using either leave-one-out or k-fold cross-validation.

        :param returns:     (T x P) array/DataFrame of asset returns
        :param constraint:  enforce positivity + sum-to-1 in Markowitz solver
        :param cv_method:   'loo' for leave-one-out or integer >1 for k-fold splits
        """
        # Convert pandas DataFrame input to NumPy array if necessary
        try:
            returns = returns.values
        except AttributeError:
            returns = np.asarray(returns)

        # Perform eigen decomposition on returns to get eigenvalues and eigenvectors
        eigval, eigvec = UPSA._eigen_decomposition(returns)
        self.eigenvalues = eigval
        self.eigenvectors = eigvec

        # If no shrinkage grid provided, generate default z_list spanning eigenvalues range
        if self.z_list is None:
            self.z_list = np.logspace(
                np.log10(eigval.min()),  # smallest eigenvalue
                np.log10(eigval.max()),  # largest eigenvalue
                10                        # number of grid points
            )

        # Choose cross-validation routine based on cv_method
        if isinstance(cv_method, int) and cv_method > 1:
            # Use k-fold CV
            folds = cv_method
            cv_rets, full_ports = self._kfold_and_eff_port_estimators(
                returns, eigval, eigvec, folds
            )
        else:
            # Default to leave-one-out CV
            cv_rets, full_ports = self._loo_and_eff_port_estimators(
                returns, eigval, eigvec
            )

        # Store all efficient portfolios and compute UPSA selection
        self.eff_port = full_ports
        self.best_z, self.upsa = UPSA._compute_upsa(cv_rets, constraint)
        return self

    @staticmethod
    def _compute_upsa(oos_rets: np.ndarray, constraint: bool = False):
        # Compute mean and covariance of out-of-sample returns
        means = oos_rets.mean(axis=0)
        cov = (oos_rets.T @ oos_rets) / oos_rets.shape[0]

        # Solve Markowitz (with or without constraints) for final weights
        upsa = UPSA._markowitz_constrained(means, cov, constraint)
        # Select shrinkage parameter that maximizes mean / volatility
        best_z = int(np.argmax(means / np.sqrt(np.diag(cov))))
        return best_z, upsa

    def get_upsa_weights(self) -> np.ndarray:
        # Compute combined UPSA portfolio weights
        return self.eff_port @ self.upsa

    def get_ridge_weights(self) -> np.ndarray:
        # Retrieve weights for the best shrinkage parameter only
        return self.eff_port[:, self.best_z]

    def _loo_and_eff_port_estimators(self,
                                     returns: np.ndarray,
                                     eigval: np.ndarray,
                                     eigvec: np.ndarray):
        # Compute full-sample ridge-efficient portfolios
        full_ports = self._compute_ridge_eff_portfolios(returns, eigval, eigvec)
        # Compute smoother matrices for each z
        smoothers = self._compute_smoothers(returns, eigval, eigvec)
        # Calculate leave-one-out returns given smoothers
        loo_rets, _ = self._calculate_loo_returns(smoothers)
        return loo_rets, full_ports

    def _kfold_and_eff_port_estimators(self,
                                       returns: np.ndarray,
                                       eigval: np.ndarray,
                                       eigvec: np.ndarray,
                                       k_folds: int):
        T, _ = returns.shape
        # Initialize matrix to store CV returns for each observation and z
        cv_rets = np.zeros((T, len(self.z_list)))
        kf = KFold(n_splits=k_folds, shuffle=False)

        # Loop over folds, fit on train, test on hold-out
        for train_idx, test_idx in kf.split(returns):
            r_train = returns[train_idx, :]
            r_test = returns[test_idx, :]
            ev_train, evec_train = UPSA._eigen_decomposition(r_train)
            ports_train = self._compute_ridge_eff_portfolios(r_train, ev_train, evec_train)
            # Record test-set returns for each z
            cv_rets[test_idx, :] = r_test @ ports_train

        # Compute full-sample efficient portfolios as well
        full_ports = self._compute_ridge_eff_portfolios(returns, eigval, eigvec)
        return cv_rets, full_ports

    def _compute_ridge_eff_portfolios(self,
                                      returns: np.ndarray,
                                      eigval: np.ndarray,
                                      eigvec: np.ndarray) -> np.ndarray:
        # Estimate expected returns vector
        mu = np.mean(returns, axis=0)
        T, P = returns.shape
        # Project mean onto eigenvector space
        projected_mean_vec = eigvec.T @ mu  # U' mu
        if P <= T:
            # No adjustment needed when covariance is full-rank
            port_adj = 0
        else:
            # Adjust for degenerate covariance (zero eigenvalues)
            port_adj = mu - eigvec @ projected_mean_vec  # (I - UU') * mu

        # Compute ridge-efficient portfolios across all z values
        eff_portfolios = [
            # (U * diag(1/(eigval+z))) @ U' mu + adjustment/z
            np.squeeze((eigvec * (1 / (eigval.reshape(1, -1) + z_)))
                       @ projected_mean_vec + port_adj / z_)
            for z_ in self.z_list
        ]

        # Return portfolios as P x len(z_list) array
        return np.array(eff_portfolios).T

    def _compute_smoothers(self, returns, eigval, eigvec):
        T, P = returns.shape
        # Project time-series returns into eigen space
        projected_ret_vec = eigvec.T @ returns.T  # U' R'
        if P <= T:
            smoother_adj = 0
        else:
            # Adjustment for zero eigenvalues in covariance
            smoother_adj = (returns @ returns.T -
                            projected_ret_vec.T @ projected_ret_vec)

        # Compute smoother matrices for each z value
        list_of_smoothers = [
            (projected_ret_vec.T @ ((1 / (eigval.reshape(-1, 1) + z_)) * projected_ret_vec)
             + smoother_adj / z_) / T
            for z_ in self.z_list
        ]

        return list_of_smoothers

    def _calculate_loo_returns(self, smoothers, w_diag=None):
        # Compute leave-one-out returns from smoother matrices
        if w_diag is None:
            # Diagonal elements of each smoother
            w_diag = [np.diag(S) for S in smoothers]
        # Sum of weights for each smoother
        w_bar = [S.sum(axis=1) for S in smoothers]
        # LOO returns adjustment formula for each z
        loo = [(w_bar[i] - w_diag[i]) / (1 - w_diag[i])
               for i in range(len(smoothers))]
        return np.array(loo).T, w_diag

    @staticmethod
    def _eigen_decomposition(features: np.ndarray,
                             cap: bool = True):
        # Compute covariance in primal or dual space depending on P and T
        T, P = features.shape
        cov = (features @ features.T if P > T else features.T @ features) / T
        eigval, eigvec = np.linalg.eigh(cov)
        if cap:
            eigval = np.maximum(eigval, 1e-8)  # cap all eigenvalues from below
        if P > T:
            # Convert dual eigenvectors to primal space
            eigvec = (features.T @ eigvec) * \
                     (eigval**(-0.5)).reshape(1, -1) / np.sqrt(T)
        return eigval, eigvec

    @staticmethod
    def _markowitz_constrained(mean: np.ndarray,
                               cov: np.ndarray,
                               constraint: bool = False) -> np.ndarray:
        # Solve quadratic program for mean-variance optimization
        solvers.options['show_progress'] = False
        # No-short constraint: G x >= 0
        G = matrix(-np.eye(len(mean)))
        h = matrix(np.zeros((len(mean), 1)))
        # Objective: minimize (1/2)x'P x - q'x
        q = matrix(-mean.astype(np.double))
        P = matrix(cov.astype(np.double))
        if constraint:
            # Add sum-to-1 equality constraint
            A = matrix(np.ones((1, len(mean))))
            b = matrix(1.0)
            sol = solvers.qp(P, q, G, h, A, b)
        else:
            sol = solvers.qp(P, q, G, h)
        return np.array(sol['x']).flatten()
