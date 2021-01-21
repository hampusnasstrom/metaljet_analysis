import os
import warnings

import fabio
import numpy as np
import pandas as pd
from os import path

import pyFAI

from HelpFunctions import progress


def integrate_data(directory: str, poni: str, flat: str = None, mask: str = None) -> pd.DataFrame:
    """
    Function for loading all integrated patterns in a directory

    :param directory: Path to directory from which to load the patterns
    :type directory: str
    :param poni: Path to poni file
    :type poni: str
    :param flat: Optional path to flatfield image
    :type flat: str
    :param mask: Optional path to mask image
    :type mask: str
    :return: A pandas data frame with scattering vector as columns and time as index
    :rtype: pandas.DataFrame
    """
    ai = pyFAI.load(poni)
    try:
        mask_file = fabio.open(mask)
        mask = mask_file.data
    except FileNotFoundError:
        mask = None
    try:
        flatfield_file = fabio.open(flat)
        flatfield = flatfield_file.data
        flatfield[flatfield > 1000] = 1
    except FileNotFoundError:
        flatfield = None

    log = pd.read_csv(path.join(directory, 'log_all.txt'), delimiter='\t', skiprows=8)
    n_files = log['measurement '].values[-1]
    dates = log['Date '].values
    times = log['Time '].values
    data = []
    time = []
    q = np.loadtxt(path.join(directory, 'image_%05d_integrated.dat' % 1))[:, 0]
    for idx, measurement in enumerate(log['measurement '].values):
        progress(idx, n_files, status='reading files')
        try:
            data.append(np.loadtxt(path.join(directory, 'image_%05d_integrated.dat' % measurement))[:, 1])
            time.append(pd.to_datetime(dates[idx] + ' ' + times[idx]))
        except OSError:
            message = 'Image %d has no integrated data' % measurement
            warnings.warn(message)
    progress(1, 1, status='done')
    df = pd.DataFrame(data=data, index=time, columns=q)
    return df
