import numpy as np

# # # # Primary experimental parameters # # # #
# # Objective function parameters
NUMBER_OF_PROPERTIES = 2

# Function Specific
# TRUE_FUNC = "Benchmark"
# TRUE_FUNC = "Ackley1D"
# TRUE_FUNC = "Matyas2D"
# TRUE_FUNC = "Griewank3D"
TRUE_FUNC = "Griewank5D"
# TRUE_FUNC = "Rosenbrock3D"

if TRUE_FUNC == "Benchmark":
    FUNC_MAX = 1.4
    EACH_FEAT_BOUND = [0, 10]
    PARAMS_ESTIMATION = True
    EACH_FEAT_LENGTHSCALE = 0.2
    MIN_LENGTHSCALE = 0.1
    MAX_LENGTHSCALE = 0.9
    NUMBER_OF_FEATURES = 1

elif TRUE_FUNC == "Ackley1D":
    FUNC_MAX = 0
    EACH_FEAT_BOUND = [-32.768, 32.768]
    PARAMS_ESTIMATION = True
    EACH_FEAT_LENGTHSCALE = 0.2
    MIN_LENGTHSCALE = 0.1
    MAX_LENGTHSCALE = 0.9
    NUMBER_OF_FEATURES = 1

elif TRUE_FUNC == "Matyas2D":
    FUNC_MAX = 0
    EACH_FEAT_BOUND = [-10, 10]
    PARAMS_ESTIMATION = True
    EACH_FEAT_LENGTHSCALE = 0.2
    MIN_LENGTHSCALE = 0.1
    MAX_LENGTHSCALE = 0.9
    NUMBER_OF_FEATURES = 2

elif TRUE_FUNC == "Griewank3D":
    FUNC_MAX = 0
    EACH_FEAT_BOUND = [-600, 600]
    PARAMS_ESTIMATION = True
    EACH_FEAT_LENGTHSCALE = 0.2
    MIN_LENGTHSCALE = 0.1
    MAX_LENGTHSCALE = 0.9
    NUMBER_OF_FEATURES = 3

elif TRUE_FUNC == "Griewank5D":
    FUNC_MAX = 0
    EACH_FEAT_BOUND = [-600, 600]
    PARAMS_ESTIMATION = True
    EACH_FEAT_LENGTHSCALE = 0.2
    MIN_LENGTHSCALE = 0.1
    MAX_LENGTHSCALE = 0.9
    NUMBER_OF_FEATURES = 5

elif TRUE_FUNC == "Rosenbrock3D":
    FUNC_MAX = 0
    EACH_FEAT_BOUND = [-2.048, 2.048]
    PARAMS_ESTIMATION = True
    EACH_FEAT_LENGTHSCALE = 0.2
    MIN_LENGTHSCALE = 0.1
    MAX_LENGTHSCALE = 0.9
    NUMBER_OF_FEATURES = 3


# # Algorithmic parameters
NUM_ITERS = NUMBER_OF_FEATURES * 10 + 5
NUM_RESTARTS_ACQ = 50
NUM_RESTARTS_LLK = 50
ACQ_TYPE = "UCB"
DELTA_PROB = 0.1

# Gaussian process specific parameters
# True SYnthetic Function
SIGNAL_VARIANCE = 1
SIGMA_NOISE = 1e-3

# I/O files related parameters
DIRECTORY_PATH = "DataFiles/"
OBS_FILE = DIRECTORY_PATH + "PreferenceData.xlsx"
SUGGESTIONS_PATH = "DataFiles/Suggestions/"
SUGGESTIONS_FILE = SUGGESTIONS_PATH + "Suggestions.xlsx"
OBJECTIVE_SHEETNAME = "Objective"
ARMS_SHEETNAME = "ArmsPullOrder"

BASELINE_OBS_FILE = DIRECTORY_PATH + "BaselineData.xlsx"
BASELINE_SHEETNAME = "Baseline"

# Newton-Raphson descent parameters
NR_MAX_ITERS = 1000
NR_ETA = 0.01
NR_TOLERANCE = 1e-5
NUMBER_OF_TESTPOINTS = 500

# # # # End of primary experimental parameters # # # #

# # Experimental parameters derived from primary experimental parameters (See above)
NUM_INIT_OBS = NUMBER_OF_FEATURES + 3
LENGTHSCALE = [EACH_FEAT_LENGTHSCALE for i in range(NUMBER_OF_FEATURES)]
AUX_LENGTHSCALE = [EACH_FEAT_LENGTHSCALE for i in range(NUMBER_OF_FEATURES+NUMBER_OF_PROPERTIES)]
PROP_SHEET_NAMES = ["Property"+str(k+1) for k in range(NUMBER_OF_PROPERTIES)]
INPUT_VARIABLES = ["x"+str(l+1) for l in range(NUMBER_OF_FEATURES)]
OUTPUT_VARIABLE = ["y"]
PREFERENCE_VARIABLE = ["Preference"]
BOUNDS = np.array([EACH_FEAT_BOUND for j in range(NUMBER_OF_FEATURES)])
ALL_SHEETS = [OBJECTIVE_SHEETNAME] + PROP_SHEET_NAMES
LENGTHSCALE_BOUNDS = [[MIN_LENGTHSCALE, MAX_LENGTHSCALE] for i in range(NUMBER_OF_FEATURES)]
AUX_LENGTHSCALE_BOUNDS = [[MIN_LENGTHSCALE, MAX_LENGTHSCALE] for i in range(NUMBER_OF_FEATURES+NUMBER_OF_PROPERTIES)]


class ExpParams:
    # # Constructor for experimental parameters object
    def __init__(self, exp_type, num_properties=NUMBER_OF_PROPERTIES, num_features=NUMBER_OF_FEATURES,
                 num_init_obs=NUM_INIT_OBS, num_iters=NUM_ITERS, restarts_acq=NUM_RESTARTS_ACQ, acq_type=ACQ_TYPE,
                 restarts_llk=NUM_RESTARTS_LLK, min_lengthscale=MIN_LENGTHSCALE,
                 max_lengthscale=MAX_LENGTHSCALE, lengthscale_bounds=LENGTHSCALE_BOUNDS, lengthscale=LENGTHSCALE,
                 aux_lengthscale=AUX_LENGTHSCALE, aux_lengthscale_bounds=AUX_LENGTHSCALE_BOUNDS,
                 signal_variance=SIGNAL_VARIANCE, sigma_noise=SIGMA_NOISE, bounds=BOUNDS, directory_path=DIRECTORY_PATH,
                 obs_file=OBS_FILE, suggestions_path=SUGGESTIONS_PATH, suggestions_file=SUGGESTIONS_FILE,
                 props_sheetname=PROP_SHEET_NAMES, nr_max_iters=NR_MAX_ITERS, nr_eta=NR_ETA, nr_tol=NR_TOLERANCE,
                 num_testpoints=NUMBER_OF_TESTPOINTS, params_estimation=PARAMS_ESTIMATION, delta_prob=DELTA_PROB,
                 true_func=TRUE_FUNC, input_variables=INPUT_VARIABLES, obj_sheet=OBJECTIVE_SHEETNAME,
                 output_variable=OUTPUT_VARIABLE,
                 pref_variable = PREFERENCE_VARIABLE, func_max=FUNC_MAX, arms_sheetname=ARMS_SHEETNAME,
                 baseline_obs_file=BASELINE_OBS_FILE, baseline_sheetname=BASELINE_SHEETNAME):

        self.exp_type = exp_type
        self.num_properties = num_properties
        self.num_features = num_features
        self.num_init_obs = num_init_obs
        self.num_iters = num_iters
        self.restarts_acq = restarts_acq
        self.acq_type = acq_type
        self.restarts_llk = restarts_llk
        self.min_lengthscale = min_lengthscale
        self.max_lengthscale = max_lengthscale
        self.lengthscale_bounds = lengthscale_bounds
        self.lengthscale = lengthscale
        self.aux_lengthscale = aux_lengthscale
        self.aux_lengthscale_bounds = aux_lengthscale_bounds
        self.signal_variance = signal_variance
        self.sigma_noise = sigma_noise
        self.bounds = bounds
        self.directory_path = directory_path
        self.obs_file = obs_file
        self.suggestions_path = suggestions_path
        self.suggestions_file = suggestions_file
        self.props_sheetname = props_sheetname
        self.nr_max_iters = nr_max_iters
        self.nr_eta = nr_eta
        self.nr_tol = nr_tol
        self.num_testpoints = num_testpoints
        self.params_estimation = params_estimation
        self.delta_prob = delta_prob
        self.true_func = true_func
        self.input_variables = input_variables
        self.obj_sheet = obj_sheet
        self.output_variable = output_variable
        self.pref_variable = pref_variable
        self.func_max = func_max
        self.arms_sheetname = arms_sheetname
        self.baseline_obs_file = baseline_obs_file
        self.baseline_sheetname = baseline_sheetname
