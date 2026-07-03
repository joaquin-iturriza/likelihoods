#!/bin/bash
cd /eos/user/j/joiturri/jitu/amplitude_DSI
source amplitudes_env/bin/activate
cd /eos/user/j/joiturri/likelihoods
singularity pull pdflatex.sif docker://astrotrop/pdflatex
singularity shell pdflatex.sif
python /eos/user/j/joiturri/likelihoods/run.py 