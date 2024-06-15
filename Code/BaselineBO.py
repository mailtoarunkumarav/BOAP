import numpy as np
from BaselineAcq import BaseAcquisition
from GPModule import GaussianProcess
import openpyxl
from termcolor import colored


class BayesOpt:

    def __init__(self, utils):
        self.utils = utils

    def generate_suggestion(self, iteration_count):

        print("Generating suggestion for Baseline BO")

        if self.utils.exp_params.num_features == 1:
            x_min = self.utils.exp_params.bounds[:, 0]
            x_max = self.utils.exp_params.bounds[:, 1]

            Xs = np.linspace(x_min, x_max, self.utils.exp_params.num_testpoints).reshape(-1, 1)

            Xs_scaled = (Xs - x_min) / (x_max - x_min)
            ys = self.utils.conduct_experiment(Xs)

        else:

            random_points = []
            Xs = []
            # Generate specified (number of unseen data points) random numbers for each dimension
            for dim in np.arange(self.utils.exp_params.num_features):
                random_data_point_each_dim = np.linspace(self.utils.exp_params.bounds[dim][0],
                                                         self.utils.exp_params.bounds[dim][1],
                                                         self.utils.exp_params.num_testpoints).reshape(1,
                                                         self.utils.exp_params.num_testpoints)
                random_points.append(random_data_point_each_dim)
            random_points = np.vstack(random_points)

            # Generated values are to be reshaped into input points in the form of x1=[x11, x12, x13].T
            for sample_num in np.arange(self.utils.exp_params.num_testpoints):
                array = []
                for dim_count in np.arange(self.utils.exp_params.num_features):
                    array.append(random_points[dim_count, sample_num])
                Xs.append(array)
            Xs = np.vstack(Xs)

            x_min = self.utils.exp_params.bounds[:, 0]
            x_max = self.utils.exp_params.bounds[:, 1]

            Xs_scaled = np.divide((Xs - x_min), (x_max - x_min))
            ys = self.utils.conduct_experiment(Xs)

        X_scaled, y_np = self.utils.load_observations_GP(self.utils.exp_params.baseline_obs_file,
                                                         self.utils.exp_params.baseline_sheetname)

        return_y_stddev = True
        delta_prob = 0.1
        base_gp_obj = GaussianProcess(X_scaled, y_np, "conventional_gp",
                                      self.utils.exp_params.restarts_llk, self.utils.exp_params.lengthscale_bounds,
                                      self.utils.exp_params.lengthscale, self.utils.exp_params.signal_variance,
                                      self.utils.exp_params.sigma_noise, self.utils.exp_params.num_features,
                                      None, None, None, return_y_stddev, Xs, ys, Xs_scaled)

        base_gp_obj.fit_data(X_scaled, y_np)
        base_gp_obj.estimateparams_refit(X_scaled, y_np)


        expertAcq = BaseAcquisition(self.utils.exp_params.restarts_acq, self.utils.exp_params.acq_type, delta_prob,
                                    self.utils.exp_params.num_features)
        x_opt, acq_func_values = expertAcq.max_acq_func(base_gp_obj, Xs_scaled, iteration_count)

        x_opt_numpy = x_opt * (x_max - x_min) + x_min

        print(colored("Baseline BO generated suggestion #{} and saved:".format(iteration_count), "blue"))
        base_gp_obj.plot_1d_gp(iteration_count, "Base-BO", X_scaled, y_np, self.utils.exp_params.num_testpoints)

        xnew = list(np.around(x_opt_numpy,decimals=3))
        ynew = list(np.around(self.utils.conduct_experiment(np.array([x_opt_numpy])),decimals=3))
        xnew_ynew = xnew + ynew
        xnew_ynew.insert(0, "Std-BO")
        xnew_ynew.insert(0, str(iteration_count + self.utils.exp_params.num_init_obs))
        workbook = openpyxl.load_workbook(filename=self.utils.exp_params.baseline_obs_file)
        sheet = workbook[self.utils.exp_params.baseline_sheetname]
        sheet.append(xnew_ynew)
        workbook.save(self.utils.exp_params.baseline_obs_file)
