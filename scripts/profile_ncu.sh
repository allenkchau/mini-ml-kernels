#!/usr/bin/env bash

KERNEL_EXE=$1   # e.g., build/kernels/relu/bench_relu
OUTDIR=profiles/$(basename $KERNEL_EXE)

mkdir -p $OUTDIR

ncu --set full --target-processes all \
    --metrics sm__sass_thread_inst_executed_op_dfma_pred_on.sum \
    -o $OUTDIR/report \
    $KERNEL_EXE
