import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from scipy.stats import norm
from termcolor import colored
import warnings

class GaussianProcess:

    # initialising GP object
    def __init__(self, X, y_fpref, gp_type, number_of_restarts_likelihood,
                 len_scale_bounds, char_len_scale, signal_variance, sigma_noise, num_features,
                 max_iterations, eta, tolerance, return_y_std, Xs, ys, Xs_scaled):
        self.X = X
        self.y_pref = y_fpref
        self.gp_type = gp_type
        self.number_of_restarts_likelihood = number_of_restarts_likelihood
        self.len_scale_bounds = len_scale_bounds
        self.char_len_scale = char_len_scale
        self.signal_variance = signal_variance
        self.sigma_noise = sigma_noise
        self.num_features = num_features
        self.max_iterations = max_iterations
        self.eta = eta
        self.tolerance = tolerance
        self.return_y_std = return_y_std
        self.Xs = Xs
        self.ys = ys
        self.Xs_scaled = Xs_scaled

        # Reserved variables
        self.y_fmap = None
        self.C = None
        self.aux_gp = False
        self.ys_scaled = None

    def fit_data(self, X, y_pref):

        self.X = X

        if self.gp_type == "preference_gp":
            # # Compute kernel matrix for the data points in X
            K_xx = self.compute_kernel(X, X, self.char_len_scale)
            K_xx = K_xx + self.sigma_noise * np.eye(len(X))
            f_prior = np.zeros(shape=(X.shape[0], 1))
            self.y_fmap, self.C = self.compute_posterior_laplace(f_prior, K_xx, y_pref)
        else:
            self.y_fmap = y_pref
            self.y_pref = y_pref

        y_min = self.y_fmap.min()
        y_max = self.y_fmap.max()
        self.y_fmap = np.divide((self.y_fmap - y_min), (y_max - y_min))

        if self.num_features == 1:
            self.ys_scaled = np.divide((self.ys - y_min), (y_max - y_min))


    def compute_posterior_laplace(self, f_prior, K_xx, preference):
        """Laplace approximation of P(f|M) and a Newton-Raphson descent is used to approximate f at the MAP """

        # Newton-Raphson descent method
        f_post = np.zeros(shape=(f_prior.shape[0] + 1, 1))
        epsilon = self.tolerance + 1
        iter_count = 0
        while iter_count < self.max_iterations and epsilon > self.tolerance:
            grads = self.compute_firstorder_derivatives(f_prior, K_xx, preference)
            hess = self.compute_secondorder_derivatives(f_prior, K_xx, preference)
            f_post = f_prior - (self.eta * np.linalg.solve(hess, grads).reshape(-1, 1))
            epsilon = np.linalg.norm(f_post - f_prior, ord=2)
            f_prior = f_post
            iter_count += 1

        final_hess = self.compute_secondorder_derivatives(f_post, K_xx, preference)
        C = final_hess - np.linalg.inv(K_xx)

        return f_post, C

    def compute_firstorder_derivatives(self, f, K_xx, preference):

        # Gradients of function p(f|M) with respect to f.
        b = np.zeros(len(f))
        for i in range(len(f)):
            # obtain list of preferences that has ith instance (both good and bad)
            v_indices = preference[:, 0] == i
            u_indices = preference[:, 1] == i

            # Compute likelihoods of a preference using CDF.
            zv = self.compute_preference_likelihood(f, preference[v_indices, :])
            zu = self.compute_preference_likelihood(f, preference[u_indices, :])
            pos_r = norm.pdf(zv) / norm.cdf(zv)
            neg_c = norm.pdf(zu) / norm.cdf(zu)
            b[i] = (sum(pos_r) - sum(neg_c)) / np.sqrt(2 * self.sigma_noise)
        Kinv_f = np.linalg.solve(K_xx, f)
        grads = Kinv_f.flatten() - b
        return grads

    def compute_secondorder_derivatives_arx(self, f, K_xx, preference):
        c = np.zeros((len(f), len(f)))

        for i in range(len(f)):
            for j in range(len(f)):

                total_sum = 0
                for k in range(len(preference)):
                    each_pref = np.array([preference[k]])
                    z_vu = self.compute_preference_likelihood(f, each_pref)
                    pdf_z_vu = norm.pdf(z_vu)
                    cdf_z_vu = norm.cdf(z_vu)
                    value = ((pdf_z_vu / cdf_z_vu) ** 2) + ((pdf_z_vu) / cdf_z_vu) * z_vu
                    if each_pref[0, 0] == i:
                        indic1 = 1
                    elif each_pref[0, 1] == i:
                        indic1 = -1
                    else:
                        indic1 = 0
                    if each_pref[0, 0] == j:
                        indic2 = 1
                    elif each_pref[0, 1] == j:
                        indic2 = -1
                    else:
                        indic2 = 0
                    total_sum += indic1 * indic2 * value

                constant = 1 / (2 * (self.sigma_noise ** 2))
                c[i][j] = constant * total_sum

        hess = np.linalg.inv(K_xx) + c
        return hess

    def compute_secondorder_derivatives(self, f, K_xx, pref):
        c = np.zeros((len(f), len(f)))
        diag_obs_c = ((norm.pdf(0) / norm.cdf(0)) ** 2) / (2 * self.sigma_noise)
        for i in range(len(pref)):
            m, n = pref[i, 0], pref[i, 1]
            z_mn = self.compute_preference_likelihood(f, pref[[i], :])
            pdf_z_mn = norm.pdf(z_mn)
            cdf_z_mn = norm.cdf(z_mn)
            c_mn = (pdf_z_mn / cdf_z_mn) ** 2 + (pdf_z_mn / cdf_z_mn) * z_mn
            c[m][n] -= c_mn / (2 * self.sigma_noise)
            c[n][m] -= c_mn / (2 * self.sigma_noise)
            c[m][m] += diag_obs_c
            c[n][n] += diag_obs_c

        hess = np.linalg.inv(K_xx) + c
        return hess

    def compute_preference_likelihood(self, f, pref):
        v = pref[:, 0]
        u = pref[:, 1]
        numerator = f[v] - f[u]
        denominator = np.sqrt(2 * self.sigma_noise)
        return (numerator/denominator).flatten()

    def gp_predict(self, Xs):

        K_xx = self.compute_kernel(self.X, self.X, self.char_len_scale)
        K_xx = K_xx + self.sigma_noise * np.eye(len(self.X))
        L_xx = np.linalg.cholesky(K_xx)
        K_xxs = self.compute_kernel(self.X, Xs, self.char_len_scale)

        LK_xsx = np.linalg.solve(L_xx, K_xxs)
        Lf = np.linalg.solve(L_xx, self.y_fmap)
        mean = np.dot(LK_xsx.T, Lf)

        K_xsxs = self.compute_kernel(Xs, Xs, self.char_len_scale)
        K_xsxs = K_xsxs + self.sigma_noise * np.eye(len(Xs))

        if not self.return_y_std:
            if self.gp_type == "preference_gp":
                K_inv = np.linalg.solve(self.C @ K_xx + np.identity(K_xx.shape[0]), self.C)
                cov = K_xsxs - K_xxs.T @ K_inv @ K_xxs
            elif self.gp_type == "conventional_gp":
                cov = K_xsxs - np.dot(LK_xsx.T, LK_xsx)
            return mean, cov
        else:
            if self.gp_type == "preference_gp":
                K_inv = np.linalg.solve(self.C @ K_xx + np.identity(K_xx.shape[0]), self.C)
                var = np.diag(K_xsxs) - \
                        np.einsum('ij,jk,ki->i', K_xxs.T, K_inv, K_xxs)
                var_negative = var < 0
                if np.any(var_negative):
                    warnings.warn("Predicted variances smaller than 0. "
                                  "Setting those variances to 0.")
                    var[var_negative] = 0.0
            elif self.gp_type == "conventional_gp":
                var_mat = K_xsxs - np.dot(LK_xsx.T, LK_xsx)
                var = np.diag(var_mat)
            return mean, np.sqrt(var).reshape(-1, 1)

    def compute_kernel(self, data_point1, data_point2, char_len_scale):
        # Element wise squaring the vector of given length scales
        char_len_scale = np.array(char_len_scale) ** 2

        # Creating a Diagonal matrix with squared l values
        sq_dia_len = np.diag(char_len_scale)

        # Computing inverse of a diagonal matrix by reciprocating each item in the diagonal
        inv_sq_dia_len = np.linalg.inv(sq_dia_len)
        kernel_mat = np.zeros(shape=(len(data_point1), len(data_point2)))

        for i in np.arange(len(data_point1)):
            for j in np.arange(len(data_point2)):
                difference = data_point1[i, :] - data_point2[j, :]
                product1 = np.dot(difference, inv_sq_dia_len)
                final_product = np.dot(product1, difference.T)
                each_kernel_val = (self.signal_variance ** 2) * (np.exp((-1 / 2.0) * final_product))
                kernel_mat[i, j] = each_kernel_val

        return kernel_mat

    # # Compute mean and variance required for the calculation of posteriors
    def compute_mean_var(self, Xs, X, y):

        K_xx = self.compute_kernel(X, X, self.char_len_scale)
        noise = 1e-3
        K_noise = K_xx + noise * np.eye(len(X))

        # # # Seldom ill-conditioned, thus commented to speed up. Uncomment in production
        # no_attempts = 1
        # while np.linalg.cond(K_noise) > 10000 and no_attempts < 10:
        #     noise = noise/100
        #     K_noise = K_xx + noise * np.eye(len(X))
        # if no_attempts > 10:
        #     print("Number of stabilising attempts exhausted!!")

        L_xx = np.linalg.cholesky(K_noise)

        # Apply the kernel function to find covariances between the unseen data points and the observed samples
        K_x_xs = self.compute_kernel(X, Xs, self.char_len_scale)
        factor1 = np.linalg.solve(L_xx, K_x_xs)
        factor2 = np.linalg.solve(L_xx, y)
        mean = np.dot(factor1.T, factor2)

        # Applying kernel function to find covariances between the unseen datapoints to find variance
        K_xs_xs = self.compute_kernel(Xs, Xs, self.char_len_scale)
        variance = K_xs_xs - np.dot(factor1.T, factor1)

        return mean, variance, factor1

    def optimize_log_marginal_likelihood_l(self, input_param):
        # 0 to n-1 elements represent the nth eleme
        init_charac_length_scale = np.array(input_param[: self.num_features])
        K_xx = self.compute_kernel(self.X, self.X, init_charac_length_scale)
        Knoise = K_xx + self.sigma_noise * np.eye(len(self.X))
        L_xx = np.linalg.cholesky(Knoise)
        factor = np.linalg.solve(L_xx, self.y_fmap)
        log_marginal_likelihood = -0.5 * (np.dot(factor.T, factor) +
                                          len(self.X) * np.log(2 * np.pi) +
                                          np.log(np.linalg.det(Knoise)))
        return log_marginal_likelihood

    def compute_max_loglikelihood(self):

        # Estimating Length scale itself
        x_max_value = None
        log_like_max = - 1 * float("inf")

        # Data structure to create the starting points for the scipy.minimize method
        random_points = []
        starting_points = []

        # Depending on the number of dimensions and bounds, generate random multi-starting points to find maxima
        for dim in np.arange(self.num_features):
            random_data_point_each_dim = np.random.uniform(self.len_scale_bounds[dim][0],
                                                           self.len_scale_bounds[dim][1],
                                                           self.number_of_restarts_likelihood). \
                reshape(1, self.number_of_restarts_likelihood)
            random_points.append(random_data_point_each_dim)

        # Vertically stack the arrays of randomly generated starting points as a matrix
        random_points = np.vstack(random_points)

        # Reformat the generated random starting points in the form [x1 x2].T for the specified number of restarts
        for sample_num in np.arange(self.number_of_restarts_likelihood):
            array = []
            for dim_count in np.arange(self.num_features):
                array.append(random_points[dim_count, sample_num])
            starting_points.append(array)
        starting_points = np.vstack(starting_points)

        total_bounds = self.len_scale_bounds.copy()

        for ind in np.arange(self.number_of_restarts_likelihood):

            init_len_scale = starting_points[ind]

            # print("Initial length scale: ", init_len_scale)
            maxima = opt.minimize(lambda x: -self.optimize_log_marginal_likelihood_l(x),
                                  init_len_scale,
                                  method='L-BFGS-B',
                                  tol=0.01,
                                  options={'maxfun': 20, 'maxiter': 20},
                                  bounds=total_bounds)

            len_scale_temp = maxima['x'][:self.num_features]
            log_likelihood = self.optimize_log_marginal_likelihood_l(len_scale_temp)

            if log_likelihood > log_like_max:
                print("New maximum log likelihood ", log_likelihood, " found for l= ",
                      maxima['x'][: self.num_features])
                x_max_value = maxima
                log_like_max = log_likelihood

        print(colored("Maximum likelihood found is {}".format(log_like_max), "green"))
        return log_like_max

    def estimateparams_refit(self, X, pref_obj):

        # Estimating Length scale itself
        x_max_value = None
        log_like_max = - 1 * float("inf")

        # Data structure to create the starting points for the scipy.minimize method
        random_points = []
        starting_points = []

        # Depending on the number of dimensions and bounds, generate random multi-starting points to find maxima
        for dim in np.arange(self.num_features):
            random_data_point_each_dim = np.random.uniform(self.len_scale_bounds[dim][0],
                                                           self.len_scale_bounds[dim][1],
                                                           self.number_of_restarts_likelihood). \
                reshape(1, self.number_of_restarts_likelihood)
            random_points.append(random_data_point_each_dim)

        # Vertically stack the arrays of randomly generated starting points as a matrix
        random_points = np.vstack(random_points)

        # Reformat the generated random starting points in the form [x1 x2].T for the specified number of restarts
        for sample_num in np.arange(self.number_of_restarts_likelihood):
            array = []
            for dim_count in np.arange(self.num_features):
                array.append(random_points[dim_count, sample_num])
            starting_points.append(array)
        starting_points = np.vstack(starting_points)

        total_bounds = self.len_scale_bounds.copy()

        for ind in np.arange(self.number_of_restarts_likelihood):

            init_len_scale = starting_points[ind]

            # print("Initial length scale: ", init_len_scale)
            maxima = opt.minimize(lambda x: -self.optimize_log_marginal_likelihood_l(x),
                                  init_len_scale,
                                  method='L-BFGS-B',
                                  tol=0.01,
                                  options={'maxfun': 20, 'maxiter': 20},
                                  bounds=total_bounds)

            len_scale_temp = maxima['x'][:self.num_features]
            log_likelihood = self.optimize_log_marginal_likelihood_l(len_scale_temp)

            if log_likelihood > log_like_max:
                print("New maximum log likelihood ", log_likelihood, " found for l= ",
                      maxima['x'][: self.num_features])
                x_max_value = maxima
                log_like_max = log_likelihood

        self.char_len_scale = x_max_value['x'][:self.num_features]
        print(colored("Optimal hyperparameters found at: {}".format(self.char_len_scale), "green"))

        # #refit data with new hyperparams
        self.fit_data(X, pref_obj)

    def plot_1d_gp(self, itr_count, title, X_objective, y_obj, num_testpoints):

        identifier = str(itr_count)+"-"+title
        if self.num_features != 1:
            print("multi-dimensional GP(s) found, skipping plots for: ", identifier)
            return

        if self.aux_gp:
            print("Aux GP found skipping plots")
            return

        plt.figure(identifier)

        f_prior_mean = np.zeros(num_testpoints)
        # plt.ylim([-0.5, 0.5])
        plt.plot(self.Xs_scaled, f_prior_mean, color="blue", linestyle="dashdot", label='Prior Mean')

        mean, std_dev = self.gp_predict(self.Xs_scaled)
        plt.plot(self.Xs_scaled, mean, color="black", linestyle="solid", label='Posterior Mean')

        if title.startswith("expobj") or title.startswith("Base-BO"):
            plt.plot(self.Xs_scaled, self.ys_scaled, color="brown", linestyle="dashed", label='True Function')

        posterior_mean_x, _ = self.gp_predict(X_objective)
        plt.plot(X_objective.flat, posterior_mean_x.flat, 'rx', label='f_map')

        plt.ylabel('f(x)')
        plt.xlabel('x')
        plt.gca().fill_between(self.Xs_scaled.flatten(), (mean - std_dev).flatten(), (mean + std_dev).flatten(),
                               color="#b0e0ff", label='Posterior Variance')

        plt.legend()
        plt.tight_layout()
        plt.savefig("Images/"+str(itr_count)+"-"+title+".pdf", bbox_inches='tight')
        plt.savefig("Images/"+str(itr_count)+"-"+title+".png", bbox_inches='tight')
        plt.close()


