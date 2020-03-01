The code for this assignment can be found at this GitHub URL:
https://github.com/changrybirds/ml-2020spring/tree/master/a2-randomized-optimization

This assignment folder contains all the code necessary to run the experiments.

The top folder in this repository contains a requirements.txt file that lists the exact
packages needed to run the code.

The main packages necessary to run the code in this repo are:
    - numpy
    - pandas
    - sklearn
    - matplotlib
    - mlrose-hiive

Each optimization problem has its own script:
    - continuous_peaks_exp.py
    - flip_flop_exp.py
    - knapsack_exp.py

The neural network weight optimization problem also has its own script:
    - run_opt_nn.py

Running each of the above scripts in a terminal or a python interpreter will run the corresponding
experiment for that problem using each of the 4 randomized optimization algorithms, and will generate
the necessary charts.

All charts are generated and saved in a `graphs/` subdirectory.
Be sure to create a director named `graphs` in the same folder before running the code.
