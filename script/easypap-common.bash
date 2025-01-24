#!/usr/bin/env bash

if [[ -z ${EASYPAPDIR+x} ]]; then
    echo "Error: EASYPAPDIR variable is not defined (easypap-common.bash)" >&2
fi

TRACEDIR=${TRACEDIR:-${EASYPAPDIR}/data/traces}
CUR_TRACEFILE=ezv_trace_current.evt
PREV_TRACEFILE=ezv_trace_previous.evt

HASHDIR=${EASYPAPDIR}/data/hash
DUMPDIR=${EASYPAPDIR}/data/dump
PERFDIR=${EASYPAPDIR}/data/perf

SIMU=${EASYPAPDIR}/bin/easypap
VIEW=${EASYPAPDIR}/lib/ezm/view/bin/easyview

KERNEL_PREFIX=("" "mipp_" "cuda_")
