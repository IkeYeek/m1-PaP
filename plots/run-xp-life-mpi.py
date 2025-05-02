#!/usr/bin/env python3
from expTools import *

mpi_configs = ['"-np 1"', '"-np 2"', '"-np 4"', '"-np 8"']

easypapOptions = {
    "-k": ["life"],                 
    "-i": [5],                       
    "-v": ["mpi_omp"],               
    "-a": ["moultdiehard130"],                          
    "-s": [512],                     
    "--label": ["square"],           
    "-of": ["life_mpi.csv"],    
    "-mpi" : mpi_configs    
}



ompICV = {
    "OMP_SCHEDULE": ["dynamic",],
    "OMP_PLACES": ["threads",],
    "OMP_NUM_THREADS": [1, 2, 4, 8, 10, 16],  
}
nbruns = 1  

execute("./run", ompICV, easypapOptions, nbruns, verbose=False, easyPath=".")

easypapOptions["-v"] = ["seq"]

seqOmpICV = {"OMP_NUM_THREADS": [1]}
execute("./run", seqOmpICV, easypapOptions, nbruns, verbose=False, easyPath=".")

print("Experiments complete. Recommended plots:")
print("plots/easyplot.py -if life_mpi.csv -v mpi_omp -- col=schedule row=label")
