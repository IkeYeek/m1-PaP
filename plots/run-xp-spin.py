#!/usr/bin/env python3
from expTools import *

# Recommended plot :
# ./plots/easyplot.py -if spin.csv -v omp -- col=schedule

options = {
    "--kernel": ["spin"],
    "--variant": ["omp"],
    "--iterations": [20],
    "--size": [1024],
    "--tile-size": [8, 32, 128],
}
# Pour renseigner l'option '-of' il faut donner le chemin depuis le répertoire easypap
options["-of"] = ["spin.csv"]

ompenv = {
    "OMP_NUM_THREADS": list(range(4, os.cpu_count() + 1, 4)),
    "OMP_SCHEDULE": ["static", "dynamic"],
}
# Lancement des experiences
execute("./run ", ompenv, options, nbruns=3, verbose=False, easyPath=".")

# Version séquentielle
options["--variant"] = ["seq"]
del options["--tile-size"]
ompenv = {"OMP_NUM_THREADS": [1]}
execute("./run", ompenv, options, nbruns=1, verbose=False, easyPath=".")

print("Recommended plot:")
print(" ./plots/easyplot.py -if spin.csv -v omp -- col=schedule")
