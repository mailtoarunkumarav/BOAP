Experimental Details of BOAP

We have conducted all our experiments in a server with the following specifications.
  • RAM: 16 GB
  • Processor: Intel (R) Xeon(R) W-2133 CPU @ 3.60GHz
  • GPU: NVIDIA Quadro P 1000 4GB + 8GB Memory
  • Operating System: Ubuntu 18.04 LTS Bionic

We recommend installing the following dependencies for the smooth execution of the modules.
  • Python - 3.6
  • scipy - 1.2.1
  • numpy - 1.16.2
  • pandas - 0.24.2
  • scikit-learn - 0.21.2


For running the experiments, please follow the steps mentioned below.
  1. Navigate to directory named “Code”, containing the source files required for conducting the experiments.
  2. For Synthetic experiments:
      (a) Navigate to “Synthetic” folder
          $ cd Synthetic
      (b) Modify experimental parameters in ExpParams.py
      (c) Specify the experimental parameters
          $ python RunModeller.py -f <function_name>
          For example: $ python RunModeller.py -f Griewank5D
