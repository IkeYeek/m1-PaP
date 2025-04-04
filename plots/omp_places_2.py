#!/usr/bin/env python3

from graphTools import *
from expTools import *
import os

# Dictionnaire avec les options de compilations d'apres commande
options = {}
options["-k"] = ["life"]
options["-i"] = [5]
options["-v"] = ["lazy_ompfor"]
options["-s"] = [512, 1024, 2048, 4096]
options["-tw"] = [32]
options["-th"] = [32]
options["-wt"] = ["opt"]
options["-a"] = ["moultdiehard130", "clown", "random"]
options["--label"] = ["with-ft"]
options["-ft"] = [""]

# Pour renseigner l'option '-of' il faut donner le chemin depuis le fichier easypap
options["-of"] = ["./data/perf/etape-2/omp_places_2.csv"]


# Dictionnaire avec les options OMP
ompenv = {}
ompenv["OMP_NUM_THREADS"] = [30]
ompenv["OMP_SCHEDULE"] = ["static,8"]

options["--label"] = ["cores+close"]
ompenv["OMP_PLACES"] = ["cores"]
ompenv["OMP_PROC_BIND"] = ["close"]

nbruns = 1
# Lancement des experiences
execute('./run ', ompenv, options, nbruns, verbose=True, easyPath=".")

options["--label"] = ["sockets+close"]
ompenv["OMP_PLACES"] = ["sockets"]
execute('./run ', ompenv, options, nbruns, verbose=True, easyPath=".")

options["--label"] = ["sockets+spreadmaster"]
ompenv["OMP_PROC_BIND"] = ["spread,master"]
execute('./run', ompenv, options, nbruns, verbose=True, easyPath=".")
# Lancement de la version seq avec le nombre de thread impose a 1
options["-v"] = ["seq"]
ompenv["OMP_NUM_THREADS"] = [1]
execute('./run', ompenv, options, nbruns, verbose=True, easyPath=".")
print(" plots/easyplot.py -if ./data/perf/life.csv -v omp_tiled -- col=schedule row=label")
