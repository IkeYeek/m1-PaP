#!/usr/bin/env python3
from graphTools import *
import sys

assert sys.version_info >= (3, 8)


computed_attr = {}

# wasted time in stalls :  only valid for sequential executions
computed_attr["watt"] = lambda df: (df["µ-joule"] / df["time"])
computed_attr["joule"] = lambda df: (df["µ-joule"] / 1000000)
computed_attr["M-pixel-per-watt"] = lambda df: (  df["size"] * df["size"] * df["iterations"] / df["watt"] / 1000000)

args = parseArguments(sys.argv, computed_attr)

df = getDataFrame(args)

# see customizing-with-matplotlibrc-files
# https: // matplotlib.org/tutorials/introductory/customizing.html

sns.set(
    style="darkgrid",
    rc={"text.usetex": False, "legend.handletextpad": 0, "figure.titlesize": "medium"},
)

# plt.style.use(['dark_background'])


# Selection des lignes :
# df = df[(-df.threads.isin([8])) & (df.kernel.isin(['mandel']))].reset_index(drop = True)


# Creation du graphe :
fig = easyPlotDataFrame(df=df, args=args)

savePlot(fig)
