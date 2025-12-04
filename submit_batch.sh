#!/bin/bash

for lr in 1e-3 5e-4 1e-4 5e-5 1e-5
do
    echo "Submitting job: LR=$lr, Base"
    sbatch -J train_base_$lr job.sh $lr --base

    echo "Submitting job: LR=$lr, EDM"
    sbatch -J train_edm_$lr job.sh $lr --edm

    echo "Submitting job: LR=$lr, V-PARAM"
    sbatch -J train_vparam_$lr job.sh $lr --vparam

    echo "Submitting job: LR=$lr, MIN-SNR"
    sbatch -J train_snr_$lr job.sh $lr --snr

    # echo "Submitting job: LR=$lr, Progressive Difficulty"
    # sbatch -J train_pd_$lr job.sh $lr --pd

    echo "Submitting job: LR=$lr, Adaptive Sampling"
    sbatch -J train_adap_s_$lr job.sh $lr --adap_s

    echo "Submitting job: LR=$lr, STF Smoothing"
    sbatch -J train_stf_$lr job.sh $lr --stf
done