#!/bin/bash
#
# Sample script for running k-NN calculation.
#
# Requires having run ./setup.sh first.

EMBEDDINGS=data/sample_embeddings.bin
NEIGHBORS=data/sample_embeddings.neighbors

if [ ! -e dependencies/pyconfig.sh ]; then
    echo "Please run ./setup.sh before running this script!"
    exit
else
    source dependencies/pyconfig.sh
fi

${PY} -m nn_saver \
    ${EMBEDDINGS} \
    -o ${NEIGHBORS} \
    -k 25 \
    -t 4 \
    --vocab ${NEIGHBORS}.vocab \
    -l ${NEIGHBORS}.log
