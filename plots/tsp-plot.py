#!/usr/bin/env python3
from graphTools import *
import sys

assert sys.version_info >= (3, 8)

import re


derived_attr = {}

# computed_attr["nb_villes"] = lambda df: re.search(r'(\d+)-', df["arg"]).group(1) if re.search(r'(\d+)-', df["arg"]) else None
# computed_attr["grain"] = lambda df: re.search(r'-(\d+)', df["arg"]).group(1) if re.search(r'-(\d+)', df["arg"]) else None


derived_attr["nb_villes"] = lambda df: df["arg"].apply(
    lambda x: (re.search(r"(\d+)-", x).group(1) if re.search(r"(\d+)-", x) else None)
)
derived_attr["grain"] = lambda df: df["arg"].apply(
    lambda x: (re.search(r"-(\d+)", x).group(1) if re.search(r"-(\d+)", x) else None)
)


args = parseArguments(sys.argv, derived_attr=derived_attr)


df = getDataFrame(args)


# args.x = "nb ville"
# args.heaty = "grain"
# see customizing-with-matplotlibrc-files
# https: // matplotlib.org/tutorials/introductory/customizing.html

sns.set(
    style="darkgrid",
    rc={"text.usetex": False, "legend.handletextpad": 0, "figure.titlesize": "medium"},
)

# plt.style.use(['dark_background'])


# Selection des lignes :
# df = df[(-df.threads.isin([8])) & (df.kernel.isin(['mandel']))].reset_index(drop = True)

df["variant"] = df["variant"].replace(["ompcol2", "ompcol3", "ompcol4"], "collapse")


# Creation du graphe :
fig = easyPlotDataFrame(df=df, args=args)

savePlot(fig)
