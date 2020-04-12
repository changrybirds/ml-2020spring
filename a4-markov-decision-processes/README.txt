The code for this assignment can be found at this GitHub URL:
https://github.com/changrybirds/ml-2020spring/tree/master/a4-markov-decision-processes

This assignment folder contains all the code necessary to run the experiments.

This assignment folder contains a `requirements.txt` file that lists the exact
packages needed to run the code.

The main external packages necessary to run the code in this repo are:
    - numpy
    - pandas
    - matplotlib
    - scipy
    - mdptoolbox-hiive

Note that there are a couple of minor edits made to the mdp class definition in mdptoolbox-hiive.
The updated file - whose contents can be copy and pasted to the `mdp.py` file - can be found in this repo as `updated_mdp.py`.

Included scripts runnable from command line:
    - `frozen_lake_exp.py` : runs the frozen lake experiment using VI, PI, and Q learning
    - `forest_mgmt_exp.py` : runs the forest management problem experiment using VI, PI, and Q learning

Running each of the above scripts in a terminal or a python interpreter will run the corresponding
parts labeled, and will generate all necessary charts.

All charts are generated and saved in a `graphs/` subdirectory. CSVs are saved in a `tmp` directory.
Be sure to create directories named `graphs` and `tmp` in the same folder before running the code.
