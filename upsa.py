import numpy as np
from cvxopt import matrix, solvers


class UPSA:
    def __init__(self, z_list: np.ndarray = None):

        self.cupsa_scale = None
        self.best_z = None
        self.cupsa = None
        self.eff_port = None
        self.z_list = z_list

    def fit(self,
            returns: np.ndarray):

        eigenvalues, eigenvectors = self.smart_eigenvalue_decomposition_covariance(returns)

        if self.z_list is None:
            log_eig = np.log10(eigenvalues)
            # if shrinkage list is not specified use eigenvalues
            self.z_list = np.logspace(min(log_eig) - 1, max(log_eig) + 1, 10)

        # LOO returns and efficient portfolios
        estimated_loo_rets, self.eff_port \
            = self._leave_one_out_and_eff_port_estimators(returns=returns,
                                                          eigenvalues=eigenvalues,
                                                          eigenvectors=eigenvectors)

        # Scale so that all ridge shrinkages have imply the same trace
        scale_for_z = np.array([np.mean(eigenvalues + z) / np.mean(eigenvalues) for z in self.z_list])
        self.eff_port = self.eff_port * scale_for_z
        estimated_loo_rets = estimated_loo_rets * scale_for_z

        # get best z (int), and CUPSA weights (between 0-1)
        self.best_z, self.cupsa = UPSA.compute_upsa(estimated_loo_rets)

        # Calculate scale to ensure scale invariance
        self.cupsa_scale = np.sum(1 / ((self.cupsa * scale_for_z) @
                                       (1 / (eigenvalues.reshape(1, -1) + self.z_list.reshape(-1, 1))))) / np.sum(
            eigenvalues)

        return self

    def get_cupsa_weights(self):
        # CUPSA is Ridge porfolios multiplied by optimal weights
        return self.eff_port @ self.cupsa * self.cupsa_scale

    def get_ridge_weights(self):
        # Recover the best ridge portfolio with cross validation
        return self.eff_port[:, self.best_z]

    @staticmethod
    def compute_upsa(estimated_loo_rets):
        """
        Use cross validated returns to derive shrinakge functions
        :param estimated_loo_rets: loo returns
        :return: best z position (int), and CUPSA weights (between 0-1)
        """

        mean = np.mean(estimated_loo_rets, 0)
        t_ = estimated_loo_rets.shape[0]
        cov = estimated_loo_rets.T @ estimated_loo_rets / t_
        cvx_upsa = UPSA.markowitz_constrained(mean, cov)

        var_pred = np.diag(cov)
        all_sharpes = mean / np.sqrt(var_pred)
        best_z = np.argmax(all_sharpes)

        return best_z, cvx_upsa

    def _leave_one_out_and_eff_port_estimators(self,
                                               returns: np.ndarray,
                                               eigenvalues: np.ndarray,
                                               eigenvectors: np.ndarray):
        """
        :param returns:
        :param eigenvalues:
        :param eigenvectors:
        :return: LOO returns and efficient portfolios
        """

        list_of_ridge_smoother, eff_portfolios = \
            self.w_and_lambda(
                returns=returns, eigenvalues=eigenvalues, eigenvectors=eigenvectors)

        leave_one_out_returns = self.calculate_loo_returns(list_of_ridge_smoother=list_of_ridge_smoother)

        return leave_one_out_returns, eff_portfolios

    def w_and_lambda(self,
                     returns: np.ndarray,
                     eigenvalues: np.ndarray,
                     eigenvectors: np.ndarray):
        """
        This is the key element of UPSA: We first compute the ridge smoother matrices
        using the identity (z+R'R)^{-1} = U (z+lambda)^{-1}U' + z^{-1} (I - UU')
        we compute R(z+R'R)^{-1} R'= R (U (z+lambda)^{-1}U' + z^{-1} (I - UU'))R'

        we also compute the efficient portfolio using the same trick:
        (z+R'R)^{-1}mu = (U (z+lambda)^{-1}U' + z^{-1} (I - UU')) mu

        :param returns:
        :param eigenvalues:
        :param eigenvectors:
        :return: Ridge smoothers and Ridge shrunk efficient portfolios
        """

        T, P = returns.shape
        projected_returns = eigenvectors.T @ returns.T  # this is U'R'
        normalized_mean = np.mean(returns, axis=0)  # mu
        projected_mean_vec = eigenvectors.T @ normalized_mean  # U' mu

        if P <= T:
            l_adj = 0
            w_adj = 0

        else:
            # complexity adjustments for when the covariance matrix is degenerate and there are zero eigenvalues.
            w_adj = (returns @ returns.T - projected_returns.T @ projected_returns) / T
            l_adj = normalized_mean - eigenvectors @ projected_mean_vec  # this is (I-UU') * mu (R\odot rf_f)

        # The quadratic form of returns and covariances for different shrinkage levels
        list_of_ridge_smoother \
            = [projected_returns.T @ ((1 / (eigenvalues.reshape(-1, 1) + z)) * projected_returns) / T + w_adj / z
               for z in self.z_list]

        # these are the standard ridge efficient portfolios
        ridge_eff_portfolios \
            = [(eigenvectors * (1 / (eigenvalues.reshape(1, -1) + z))) @ projected_mean_vec + l_adj / z
               for z in self.z_list]

        # these are efficient portfolios, organized as a P x len(z_grid) matrix
        ridge_eff_portfolios = np.array(ridge_eff_portfolios).T

        return list_of_ridge_smoother, ridge_eff_portfolios

    def calculate_loo_returns(self, list_of_ridge_smoother, w_diag=None):
        """
         We now use the formula for leave-one-out returns on the efficient portfolio.
        Given the ridge_smoother_matrix W(z) is T x T, we know that
        (\bar W - W_{t,t})/(1-W_{t,t}) is the realized return.

        :param list_of_ridge_smoother: The quadratic form of returns and covariances for different shrinkage levels
        :param w_diag:
        :return: T leave-one-out realized returns, for each z, organized as a matrix. To be used for cross-validation.
        """

        if w_diag is None:
            w_diag = [np.diag(w) for w in list_of_ridge_smoother]

        w_bar = [np.sum(w, axis=1) for w in list_of_ridge_smoother]

        leave_one_out_returns = [(w_bar[i] - w_diag[i]) / (1 - w_diag[i]) for i in
                                 range(len(list_of_ridge_smoother))]

        leave_one_out_returns = np.array(leave_one_out_returns).T

        return leave_one_out_returns

    @staticmethod
    def smart_eigenvalue_decomposition_covariance(features: np.ndarray):

        """
        Efficient Eigenvalue decomposition
        :param features: features used to create covariance matrix times x P
        :return: Left eigenvectors PxT and nls_eigenvalues without zeros
        """

        [T, P] = features.shape

        if P > T:
            covariance = features @ features.T / T

        else:
            covariance = features.T @ features / T

        eigval, eigvec = np.linalg.eigh(covariance)

        # remove small and noisy eigenvalues
        eigvec = eigvec[:, eigval > 10 ** (-10)]
        eigval = eigval[eigval > 10 ** (-10)]

        if P > T:
            # project features on normalized eigenvectors
            eigvec = features.T @ eigvec * (eigval ** (-1 / 2)).reshape(1, -1) / np.sqrt(T)

        return eigval, eigvec

    @staticmethod
    def markowitz_constrained(mean, cov):

        """
        Constrained Markowitz
        :param mean: vector of asset means
        :param cov: matrix of asset covariances
        :return:  Efficient portfolio with positive weights that sum to 1.
        """

        g_mat = matrix(-np.eye(len(mean)))
        h_mat = matrix(np.zeros((len(mean), 1)))
        a_mat = matrix(np.ones((1, len(mean))))
        b_mat = matrix(1.0)
        nu_mat = matrix(-mean.astype(np.double))
        m_mat = matrix(cov.astype(np.double))

        solvers.options['show_progress'] = False
        sol_mse = solvers.qp(m_mat, nu_mat, g_mat, h_mat, a_mat, b_mat)
        cvx_upsa = np.array(sol_mse['x']).flatten()

        return cvx_upsa
