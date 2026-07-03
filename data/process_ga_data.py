#!/usr/bin/env python3.10

import os
import glob
import numpy as np

for name in ['3g2a', '4g2a']:
    # unpack the data
    os.system(f'tar -xvzf {name}-events-njet+sherpa_new.tar.xz')

    # find all .data files
    data_files = glob.glob('*.data')
    
    # merge files
    with open(f'{name}.data', 'w') as outfile:
        for fname in data_files:
            with open(fname) as infile:
                outfile.write(infile.read())
                
    # remove the original files
    for fname in data_files:
        os.remove(fname)
        
    # convert data to .npy
    data = np.loadtxt(f'{name}.data')
    match name:
        case '3g2a':
            outname = 'aag'
        case '4g2a':
            outname = 'aagg'
    np.save(f'{outname}.npy', data)
    
    # remove the .data file
    os.remove(f'{name}.data')
    