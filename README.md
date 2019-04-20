trajectory-search
==============================

transport trajectory search

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data containing candidate trajectory ID and rtree files.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks for visualizing trajectories and compute performance statistics. 
    │   ├── pruning_power.py
    │   └── trajectory_viz.py
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to process raw data
    │   │   └── make_trajectory.py
    │   │
    │   ├── features       <- Scripts to build rtree
    │   │   ├── build_rtree.py
    │   │   └── build_bbox.py
    │   │
    │   ├── models         <- Scripts to search rtrees and then use EDR to compute top-k
    │   │   │                 trajectories
    │   │   │              <- Script to sequential scan all trajectories and find top-k
    │   │   ├── predict_model.py
    │   │   ├── search_rtree.py
    │   │   └── build_truth.py
    │   │
    │   └── statistics     <- Scripts to compute top k accuracy of result
    │       └── topkAccuracy.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
