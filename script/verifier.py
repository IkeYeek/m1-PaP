"""
CAREFUL: this piece of code was fully vibe-coded using Claude, I do not either guarantee that it works or that it is coherent, it seems to work though
"""

import os
from copy import copy
from itertools import *
import subprocess
import glob


def verify(
    base_options, compare_options, keep_gpu_mode=False, verbose=True, easyPath="."
):
    """
    Compare SHA256 hashes between different variants of the same simulation.

    Args:
        base_options (dict): Dictionary with options for base variants
        compare_options (dict): Dictionary with options for variants to compare
        keep_gpu_mode (bool): If True, preserve -g option when comparing variants
        verbose (bool): Whether to print detailed execution information
        easyPath (str): Path to the easypap directory

    Returns:
        dict: Dictionary of configurations with different hashes
    """
    path = os.getcwd()
    os.chdir(easyPath)

    # Ensure all required options exist
    required_options = ["-k", "-i", "-v", "-s", "-a"]
    for opt in required_options:
        if opt not in base_options:
            print(f"Error: Missing required option {opt} in base_options")
            os.chdir(path)
            return None
        if opt not in compare_options:
            print(f"Error: Missing required option {opt} in compare_options")
            os.chdir(path)
            return None

    # Add SHA256 option to both configs
    base_options["-sh"] = [""]
    compare_options["-sh"] = [""]

    # Dictionary to store hash results
    hash_results = {}
    # Dictionary to store configurations with different hashes
    different_configs = {}

    # Run base variants
    base_env = {"OMP_NUM_THREADS": ["1"]}  # Default environment
    if verbose:
        print("Running base variants...")
    execute("./run", base_env, base_options, 1, verbose=verbose, easyPath=easyPath)

    # Collect hash files for base variants
    kernel = base_options["-k"][0]  # Assume single kernel for simplicity
    for variant in base_options["-v"]:
        for size in base_options["-s"]:
            for arg in base_options["-a"]:
                for iter_val in base_options["-i"]:
                    # Determine workitem type if specified
                    wt_list = base_options.get("-wt", ["default"])
                    for wt in wt_list:
                        # Find the hash file
                        hash_files = glob.glob(
                            f"data/hash/{kernel}-{variant}-{wt}-dim-{size}-iter-{iter_val}-arg-{arg}.sha256"
                        )

                        if not hash_files:
                            print(
                                f"Warning: No hash file found for {kernel}-{variant}-{wt}"
                            )
                            continue

                        # Read the hash
                        with open(hash_files[0], "r") as f:
                            hash_value = f.read().strip()

                        # Check if this variant uses GPU
                        is_gpu = "-g" in base_options

                        # Store hash
                        config_key = f"{kernel}-{variant}-{wt}-{size}-{arg}"
                        hash_results[config_key] = {
                            "hash": hash_value,
                            "is_gpu": is_gpu,
                        }

    # Handle GPU flag for compare variants if needed
    if "-g" in base_options and not keep_gpu_mode:
        # If we don't want to keep GPU mode, remove the -g flag
        if "-g" in compare_options:
            del compare_options["-g"]
    elif "-g" not in base_options and "-g" in compare_options and not keep_gpu_mode:
        # If base doesn't use GPU but compare does, and we don't want to keep it
        del compare_options["-g"]

    # Run compare variants
    compare_env = {"OMP_NUM_THREADS": ["1"]}  # Default environment
    if verbose:
        print("Running compare variants...")
    execute(
        "./run", compare_env, compare_options, 1, verbose=verbose, easyPath=easyPath
    )

    # Collect hash files for compare variants
    kernel = compare_options["-k"][0]  # Assume single kernel for simplicity
    for variant in compare_options["-v"]:
        for size in compare_options["-s"]:
            for arg in compare_options["-a"]:
                for iter_val in compare_options["-i"]:
                    # Determine workitem type if specified
                    wt_list = compare_options.get("-wt", ["default"])
                    for wt in wt_list:
                        # Find the hash file
                        hash_files = glob.glob(
                            f"data/hash/{kernel}-{variant}-{wt}-dim-{size}-iter-{iter_val}-arg-{arg}.sha256"
                        )

                        if not hash_files:
                            print(
                                f"Warning: No hash file found for {kernel}-{variant}-{wt}"
                            )
                            continue

                        # Read the hash
                        with open(hash_files[0], "r") as f:
                            hash_value = f.read().strip()

                        # Check if this variant uses GPU
                        is_gpu = "-g" in compare_options

                        # Store hash
                        config_key = f"{kernel}-{variant}-{wt}-{size}-{arg}"
                        hash_results[config_key] = {
                            "hash": hash_value,
                            "is_gpu": is_gpu,
                        }

    # Compare hashes between base variants and compare variants
    for base_variant in base_options["-v"]:
        base_wt_list = base_options.get("-wt", ["default"])
        for base_wt in base_wt_list:
            for compare_variant in compare_options["-v"]:
                compare_wt_list = compare_options.get("-wt", ["default"])
                for compare_wt in compare_wt_list:
                    for size in base_options["-s"]:
                        if size not in compare_options["-s"]:
                            continue
                        for arg in base_options["-a"]:
                            if arg not in compare_options["-a"]:
                                continue
                            for iter_val in base_options["-i"]:
                                if iter_val not in compare_options["-i"]:
                                    continue

                                base_key = (
                                    f"{kernel}-{base_variant}-{base_wt}-{size}-{arg}"
                                )
                                compare_key = f"{kernel}-{compare_variant}-{compare_wt}-{size}-{arg}"

                                if (
                                    base_key in hash_results
                                    and compare_key in hash_results
                                ):
                                    if (
                                        hash_results[base_key]["hash"]
                                        != hash_results[compare_key]["hash"]
                                    ):
                                        diff_key = f"{base_key} vs {compare_key}"
                                        different_configs[diff_key] = {
                                            "base_hash": hash_results[base_key]["hash"],
                                            "compare_hash": hash_results[compare_key][
                                                "hash"
                                            ],
                                        }
                                        if verbose:
                                            print(f"Different hashes found: {diff_key}")

    os.chdir(path)

    # Print summary
    if verbose:
        print("\nVerification Results:")
        if different_configs:
            print(
                f"Found {len(different_configs)} configurations with different hashes:"
            )
            for config, hashes in different_configs.items():
                print(f"  - {config}")
                print(f"    Base hash: {hashes['base_hash']}")
                print(f"    Comp hash: {hashes['compare_hash']}")
        else:
            print("All configurations produced identical hashes.")

    return different_configs


# Import the execute function from expTools
from itertools import *
import subprocess


def iterateur_option(dicopt, sep=" "):
    options = []
    for opt, listval in dicopt.items():
        optlist = []
        for val in listval:
            optlist += [opt + sep + str(val)]
        options += [optlist]
    for value in product(*options):
        yield " ".join(value)


def execute(commande, ompenv, option, nbruns=1, verbose=True, easyPath="."):
    path = os.getcwd()
    os.chdir(easyPath)
    for i in range(nbruns):
        for omp in iterateur_option(ompenv, "="):
            for opt in iterateur_option(option):
                if verbose:
                    print(omp + " " + commande + " -n " + opt)
                if (
                    subprocess.call([omp + " " + commande + " -n " + opt], shell=True)
                    == 1
                ):
                    os.chdir(path)
                    return "Error on the command used"
    if not (verbose):
        print("Experiences done")
    os.chdir(path)


if __name__ == "__main__":
    base_gpu_options = {}
    base_gpu_options["-k"] = ["life"]
    base_gpu_options["-wt"] = ["avx2", "avx512"]
    base_gpu_options["-i"] = [10]
    base_gpu_options["-v"] = ["mpi_omp"]
    base_gpu_options["-mpi"] = ['"-np 4"']
    base_gpu_options["-s"] = [8192]
    base_gpu_options["-a"] = ["random"]

    compare_gpu_options = copy(base_gpu_options)
    compare_gpu_options["-wt"] = ["opt"]

    gpu_results = verify(
        base_gpu_options, compare_gpu_options, keep_gpu_mode=True, verbose=True
    )
