from scipy.stats import norm
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

class BaseAcquisition():

    def __init__(self, number_of_restarts, acq_type, delta_prob, num_features):
        self.number_of_restarts = number_of_restarts
        self.acq_type = acq_type
        self.delta_prob = delta_prob
        self.num_features = num_features

    def ucb_acq_func(self, mean, std_dev, iteration_count):
        with np.errstate(divide='ignore') or np.errstate(invalid='ignore'):
            b = 1
            a = 1
            r = 1
            beta = 2 * np.log((iteration_count ** 2) * (2 * (np.pi ** 2)) * (1 / (3 * self.delta_prob))) + \
                   (2 * self.num_features) * np.log((iteration_count ** 2) * self.num_features * b * r *
                                                    (np.sqrt(
                                                        np.log(4 * self.num_features * a * (1 / self.delta_prob)))))
            result = mean + np.sqrt(beta) * std_dev
            return result

    def upper_confidence_bound_util(self, x, gp_obj, iteration_count):
        with np.errstate(divide='ignore') or np.errstate(invalid='ignore'):
            # Use Gaussian Process to predict the values for mean and variance at given x
            mean, variance = gp_obj.gp_predict(np.matrix(x))
            std_dev = np.sqrt(variance)
            result = self.ucb_acq_func(mean, std_dev, iteration_count)
            # Since scipy.minimize function is used to find the minima and so converting it to maxima by * with -1
            return -1 * result

    # Method to maximize the ACQ function as specified the user
    def max_acq_func(self, gp_obj, Xs, iteration_count):

            # Initialize the xmax value and the function values to zeroes
            x_max_value = np.zeros(gp_obj.num_features)
            fmax = - 1 * float("inf")

            # Temporary data structures to store xmax's function values of each run of finding max using scipy.minimize
            tempmax_x = []
            tempfvals = []

            # Data structure to create the starting points for the scipy.minimize method
            random_points = []
            starting_points = []
            # Depending on the number of dimensions and bounds, generate random multiple starting points to find maxima
            for dim in np.arange(gp_obj.num_features):
                random_data_point_each_dim = np.random.uniform(0, 1,
                                                               self.number_of_restarts)\
                    .reshape(1, self.number_of_restarts)
                random_points.append(random_data_point_each_dim)

            # Vertically stack the arrays of randomly generated starting points as a matrix
            random_points = np.vstack(random_points)

            # Reformat the generated random starting points in the form [x1 x2].T for the specified number of restarts
            for sample_num in np.arange(self.number_of_restarts):
                array = []
                for dim_count in np.arange(gp_obj.num_features):
                    array.append(random_points[dim_count, sample_num])
                starting_points.append(array)
            starting_points = np.vstack(starting_points)

            if self.acq_type == "UCB":
                # print("ACQ Function : UCB ")

                # Obtain the maxima of the ACQ function by starting the optimization at different start points
                for starting_point in starting_points:

                    # Find the maxima in the bounds specified for the UCB ACQ function
                    max_x = opt.minimize(lambda x: self.upper_confidence_bound_util(x, gp_obj, iteration_count),
                                         starting_point,
                                         method='L-BFGS-B',
                                         tol=0.001,
                                         bounds=[[0,1] for i in range(gp_obj.num_features)])

                    # Use gaussian process to predict mean and variances for the maximum point identified
                    mean, variance = gp_obj.gp_predict(np.matrix(max_x['x']))
                    std_dev = np.sqrt(variance)
                    fvalue = self.ucb_acq_func(mean, std_dev, iteration_count)

                    # Store the maxima of ACQ function and the corresponding value at the maxima, required for debugging
                    tempmax_x.append(max_x['x'])
                    tempfvals.append(fvalue)

                    # Compare the values in the current run to find the best value overall and store accordingly
                    if (fvalue > fmax):
                        #print("New best Fval: ", fvalue, " found at: ", max_x['x'])
                        x_max_value = max_x['x']
                        fmax = fvalue

                print("UCB Best is ", fmax, "at ", x_max_value)

                # Calculate the ACQ function values at each of the unseen data points to plot the ACQ function
                with np.errstate(invalid='ignore'):
                    mean, variance = gp_obj.gp_predict(Xs)
                    std_dev = np.sqrt(variance)
                    acq_func_values = self.ucb_acq_func(mean, std_dev, iteration_count)

            return x_max_value, acq_func_values
