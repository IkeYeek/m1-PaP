#!/usr/bin/env python3
from graphTools import *
import sys

assert sys.version_info >= (3, 8)


computed_attr = {}

# wasted time in stalls :  only valid for sequential executions
computed_attr["stall"] = lambda df: (
    df["total_stalls"] * df["time"] / (df["total_cycles"] * 1000)
)

computed_attr["stall_ratio"] = lambda df: (df["total_stalls"] / (df["total_cycles"]))

computed_attr["frequency"] = lambda df: (
    (df["total_cycles"]) / (df["time"] * df["threads"])
)

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
