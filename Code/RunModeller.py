# # Packages import
import sys
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from pathlib import Path
from termcolor import colored
import torch
from GPModule import GaussianProcess
from ExpParams import ExpParams
from UtilsCollector import UtilsCollector
from BaselineBO import BayesOpt
import openpyxl

device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

import warnings
warnings.filterwarnings("ignore")

class PropModeller:
    def __init__(self, exp_params, utils):
        self.exp_params = exp_params
        self.utils = utils

    def wrapper(self, run):

        # # Initialising helper objects
        self.utils = utils
        baseline_bo = BayesOpt(utils)

        # # Generate initial random observations
        self.utils.generate_initial_observations()

        if Path(self.exp_params.obs_file).is_file():
            obsfile_mod_time = os.path.getmtime(exp_params.obs_file)
            if sys.argv[1] == "Synthetic":
                print(colored("▪▪▪▪▪▪▪▪  Setting up the environment for Real-world experiments  ▪▪▪▪▪▪▪▪", "green"))
        else:
            print(colored("Could you please ensure the mandatory files for starting the experiments??", "red"))
            exit(0)

        itr_count = 1
        armlist=[]
        bool_start = True
        while itr_count <= self.exp_params.num_iters:
            # # Poll for changes to the observations file. [Can be enhanced to check if the number of rows are same]
            new_mod_time = os.path.getmtime(self.exp_params.obs_file)
            if np.round(new_mod_time, 3) != np.round(obsfile_mod_time, 3) or bool_start or sys.argv[1] == "Synthetic":
                print(colored("\n\n▪▪▪▪▪▪▪▪ Running algorithms at Iteration #{} ▪▪▪▪▪▪▪▪".format(itr_count),"green"))
                obsfile_mod_time = new_mod_time
                return_y_stddev = True

                # Generate Suggestion for Baseline in the current iteration
                baseline_bo.generate_suggestion(itr_count)

                if self.utils.exp_params.num_features == 1:

                    x_min = self.exp_params.bounds[:, 0]
                    x_max = self.exp_params.bounds[:, 1]
                    Xs = np.linspace(x_min, x_max, self.utils.exp_params.num_testpoints).reshape(-1, 1)
                    ys = self.utils.conduct_experiment(Xs)
                    Xs_scaled = (Xs - x_min) / (x_max - x_min)

                else:
                    # Generate test points in the given bound
                    random_points = []
                    Xs = []
                    # Generate specified (number of unseen data points) random numbers for each dimension
                    for dim in np.arange(self.exp_params.num_features):
                        random_data_point_each_dim = np.linspace(self.exp_params.bounds[dim][0],
                                                                 self.exp_params.bounds[dim][1],
                                                                 self.exp_params.num_testpoints). \
                            reshape(1, self.exp_params.num_testpoints)
                        random_points.append(random_data_point_each_dim)
                    random_points = np.vstack(random_points)

                    # Generated values are to be reshaped into input points in the form of x1=[x11, x12, x13].T
                    for sample_num in np.arange(self.exp_params.num_testpoints):
                        array = []
                        for dim_count in np.arange(self.exp_params.num_features):
                            array.append(random_points[dim_count, sample_num])
                        Xs.append(array)
                    Xs = np.vstack(Xs)

                    x_min = self.utils.exp_params.bounds[:, 0]
                    x_max = self.utils.exp_params.bounds[:, 1]

                    Xs_scaled = np.divide((Xs - x_min), (x_max - x_min))
                    ys = self.utils.conduct_experiment(Xs)

                X_exp_obj, y_exp_obj = self.utils.load_observations_GP(self.exp_params.obs_file, "Objective")
                gp_exp_obj = GaussianProcess(X_exp_obj, y_exp_obj, "conventional_gp",
                                             self.exp_params.restarts_llk, self.exp_params.lengthscale_bounds,
                                             self.exp_params.lengthscale, self.exp_params.signal_variance,
                                             self.exp_params.sigma_noise, self.exp_params.num_features,
                                             self.exp_params.nr_max_iters, self.exp_params.nr_eta,
                                             self.exp_params.nr_tol, return_y_stddev, Xs, ys, Xs_scaled)
                gp_exp_obj.fit_data(X_exp_obj, y_exp_obj)
                gp_exp_obj.plot_1d_gp(itr_count,  str(run)+"expobj-bef", X_exp_obj, y_exp_obj, self.exp_params.num_testpoints)

                X_hum_obj, y_hum_obj = self.utils.load_observations_GP(self.exp_params.obs_file, "Objective")

                _, pref_prop1 = self.utils.load_observations_GP(self.exp_params.obs_file, "Property1")
                gp_prop1 = GaussianProcess(X_hum_obj, pref_prop1, "preference_gp",
                                           self.exp_params.restarts_llk, self.exp_params.lengthscale_bounds,
                                           self.exp_params.lengthscale, self.exp_params.signal_variance,
                                           self.exp_params.sigma_noise, self.exp_params.num_features,
                                           self.exp_params.nr_max_iters, self.exp_params.nr_eta, self.exp_params.nr_tol,
                                           return_y_stddev, Xs, ys, Xs_scaled)

                # Plotting posterior for 1D objective function
                gp_prop1.fit_data(X_hum_obj, pref_prop1)
                gp_prop1.plot_1d_gp(itr_count, str(run)+"prop1-bef", X_hum_obj, pref_prop1, self.exp_params.num_testpoints)

                _, pref_prop2 = self.utils.load_observations_GP(self.exp_params.obs_file, "Property2")
                gp_prop2 = GaussianProcess(X_hum_obj, pref_prop2, "preference_gp",
                                           self.exp_params.restarts_llk, self.exp_params.lengthscale_bounds,
                                           self.exp_params.lengthscale, self.exp_params.signal_variance,
                                           self.exp_params.sigma_noise, self.exp_params.num_features,
                                           self.exp_params.nr_max_iters, self.exp_params.nr_eta, self.exp_params.nr_tol,
                                           return_y_stddev, Xs, ys, Xs_scaled)

                # Plotting posterior for 1D objective function
                gp_prop2.fit_data(X_hum_obj, pref_prop2)
                gp_prop2.plot_1d_gp(itr_count,  str(run)+"prop2-bef", X_hum_obj, pref_prop2, self.exp_params.num_testpoints)

                X_aux_obj = self.utils.generate_auxillary_inputs(X_hum_obj, gp_prop1, gp_prop2)
                gp_hum_obj = GaussianProcess(X_aux_obj, y_hum_obj, "conventional_gp",
                                             self.exp_params.restarts_llk, self.exp_params.aux_lengthscale_bounds,
                                             self.exp_params.aux_lengthscale, self.exp_params.signal_variance,
                                             self.exp_params.sigma_noise,
                                             self.exp_params.num_features + self.exp_params.num_properties,
                                             self.exp_params.nr_max_iters, self.exp_params.nr_eta, self.exp_params.nr_tol,
                                             return_y_stddev,  Xs, ys, Xs_scaled)

                gp_hum_obj.aux_gp = True
                # Plotting posterior for 1D objective function
                gp_hum_obj.fit_data(X_aux_obj, y_hum_obj)
                gp_hum_obj.plot_1d_gp(itr_count,  str(run)+"hobj-bef", X_hum_obj, y_hum_obj, self.exp_params.num_testpoints)

                if self.exp_params.params_estimation:
                    print("Estimating parameters for Experiment-Objective GP")
                    gp_exp_obj.estimateparams_refit(X_exp_obj, y_exp_obj)
                    print("Estimating parameters for GP modelling property 1")
                    gp_prop1.estimateparams_refit(X_hum_obj, pref_prop1)
                    print("Estimating parameters for GP modelling property 2")
                    gp_prop2.estimateparams_refit(X_hum_obj, pref_prop2)
                    print("Estimating parameters for Human-Objective GP")
                    gp_hum_obj.estimateparams_refit(X_aux_obj, y_hum_obj)

                    # gp_exp_obj.plot_1d_gp(itr_count, "expobj-af", X_exp_obj, y_exp_obj, self.exp_params.num_testpoints)
                    # gp_prop1.plot_1d_gp(itr_count, "prop1-af", X_exp_obj, pref_prop1, self.exp_params.num_testpoints)
                    # gp_prop2.plot_1d_gp(itr_count, "prop2-af", X_exp_obj, pref_prop2, self.exp_params.num_testpoints)
                    # gp_hum_obj.plot_1d_gp(itr_count, "hobj-af", X_exp_obj, y_hum_obj, self.exp_params.num_testpoints)

                # Calculate reward for each arm using Thompson sampling
                r_exp_arm, x_new_scaled = self.utils.calculate_arm_reward_GO("exp", gp_exp_obj)
                r_human_arm, x_new_scaled = self.utils.calculate_arm_reward_GO("human", gp_hum_obj)

                print("ExpArm_Rewards: ", r_exp_arm, "\tHumArmRewards:", r_human_arm)
                if r_human_arm > r_exp_arm:
                    print("Pulling human arm for suggesting the next candidate: ", x_new_scaled)
                    arm_pulled = "Human"
                    reward_val = str(np.around(r_human_arm, decimals=3))+">"+str(np.around(r_exp_arm, decimals=3))
                else:
                    print("Pulling experiments arm for suggesting the next candidate: ", x_new_scaled)

                    arm_pulled = "Experiment"
                    reward_val = str(np.around(r_exp_arm, decimals=3))+">"+str(np.around(r_human_arm, decimals=3))

                print("Evaluating the objective function at this point")
                x_new = x_new_scaled * (x_max - x_min) + x_min
                print("Xnew suggested (scaled): ", x_new_scaled, " \toriginal:", x_new)
                y_new = self.utils.conduct_experiment(np.array([x_new]))
                print("f(", x_new, ")=", y_new)

                if arm_pulled == "Human":
                    # Generate preference of the newly suggested point
                    self.utils.update_preferences(x_new, y_new)

                arm_sheet = self.exp_params.obj_sheet
                print("Writing files...")
                xnew_ynew = list(np.around(x_new, decimals=3)) + list(np.around(y_new, decimals=3))
                xnew_ynew.insert(0, arm_pulled)
                xnew_ynew.insert(0, str(itr_count-1 + self.exp_params.num_init_obs))
                workbook = openpyxl.load_workbook(filename=self.exp_params.obs_file)
                sheet = workbook[arm_sheet]
                sheet.append(xnew_ynew)

                order_sheet = workbook[self.exp_params.arms_sheetname]
                pulled_arm_details = list([reward_val])
                pulled_arm_details.insert(0, arm_pulled)
                pulled_arm_details.insert(0, str(itr_count - 1 + self.exp_params.num_init_obs))
                order_sheet.append(pulled_arm_details)

                workbook.save(self.exp_params.obs_file)

                print(colored("##### Iteration# {} Complete #####".format(itr_count), "green"))
                plt.close()
                # plt.show()
                itr_count += 1
            time.sleep(1)

        return

        # print("plotting regret")
        # _, y_obs = self.utils.load_observations_GP(self.exp_params.obs_file, "ExperimentObjective")
        # max_val = self.exp_params.func_max
        # plt.figure("Regret")
        # tmp = [max_val - max(y_obs[0:4])]
        # for i in range(1, len(y_obs)):
        #     if (max_val - y_obs[i]) < tmp[-1]:
        #         tmp.append(max_val - y_obs[i])
        #     else:
        #         tmp.append(tmp[-1])
        # plt.plot(np.arange(0, len(tmp)), tmp)
        # plt.xlabel(" Batch #")
        # plt.ylabel("Regret")
        # plt.show()

if __name__ == '__main__':
    sys.argv.append("Synthetic")
    exp_type = sys.argv[1]

    runs = 10
    for run in range(runs):
        np.random.seed(run)
        print(colored("\n\n#### Starting run {} ####".format(run+1), "blue"))
        exp_params = ExpParams(exp_type)
        utils = UtilsCollector(exp_params)
        propsmodel_obj = PropModeller(exp_params, utils)
        propsmodel_obj.wrapper(run+1)
        os.rename(exp_params.directory_path+"BaselineData.xlsx",exp_params.directory_path+"BaselineData_"
                  +str(run+1)+".xlsx")
        os.rename(exp_params.directory_path + "PreferenceData.xlsx", exp_params.directory_path + "PreferenceData_"
                  + str(run + 1) + ".xlsx")
        print(colored("\n\n#### Run {} completed ####".format(run+1), "blue"))


