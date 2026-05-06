#!/usr/bin/env bash

#SBATCH --job-name=build_torch
#SBATCH --output=build_torch.out
#SBATCH --error=build_torch.err
#SBATCH --cpus-per-task=32
#SBATCH --mem=80G

export SINGULARITY_TMPDIR=$HOME/.singularity/tmp
export SINGULARITY_CACHEDIR=$HOME/.singularity/cache
mkdir -p $SINGULARITY_CACHEDIR $SINGULARITY_TMPDIR

#product.def
# The path to the definition file
input_def="llm.def"

# The resulting container image
output_sif="llm.sif"

singularity build --fakeroot $output_sif $input_def
