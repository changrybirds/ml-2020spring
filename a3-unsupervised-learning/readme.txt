The code for this assignment can be found at this GitHub URL:
https://github.com/changrybirds/ml-2020spring/tree/master/a3-unsupervised-learning

This assignment folder contains all the code necessary to run the experiments.

This assignment folder contains a requirements.txt file that lists the exact
packages needed to run the code.

The main external packages necessary to run the code in this repo are:
    - numpy
    - pandas
    - sklearn
    - matplotlib
    - scipy

Included scripts runnable from command line:
    - `clustering.py` : runs k-means and expectation maximization algorithms (part 1)
    - `dim_reduction.py` : runs PCA, ICA, RP, and DT (feature importances)
        also re-runs clustering experiments with reduced-dimensionality X's
        (parts 2 and 3)
    - `nn_dim_reduction.py` : runs NN problems (parts 4 and 5)

Running each of the above scripts in a terminal or a python interpreter will run the corresponding
parts labeled, and will generate all necessary charts.

All charts are generated and saved in a `graphs/` subdirectory. CSVs are saved in a `tmp` directory.
Be sure to create directories named `graphs` and `tmp` in the same folder before running the code.
