import os
import warnings
import sys

import fabio
import numpy as np
import pandas as pd
from os import path
import matplotlib.pyplot as plt

import pyFAI

from matplotlib import use
from HelpFunctions import progress, extend_mesh

use('Qt5Agg')


def integrate_data(directory: str, poni: str, flat: str = None, mask: str = None) -> pd.DataFrame:
    """
    Function for integrating all images in a directory

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
    frame = fabio.open(path.join(directory, 'image_%05d.tif' % 1)).data
    q = ai.integrate1d(data=frame, npt=2048, unit="q_nm^-1")[0]
    for idx, measurement in enumerate(log['measurement '].values):
        progress(idx, n_files, status='reading files')
        frame = fabio.open(path.join(directory, 'image_%05d.tif' % measurement)).data
        data.append(ai.integrate1d(data=frame,
                                   mask=mask,
                                   npt=2048,
                                   flat=flatfield,
                                   unit="q_nm^-1"
                                   )[1])
        time.append(pd.to_datetime(dates[idx] + ' ' + times[idx]))
    progress(1, 1, status='done')
    df = pd.DataFrame(data=data, index=time, columns=q)
    return df


if __name__ == "__main__":
    args = {'directory': sys.argv[1]}
    name = os.path.split(args['directory'])[1]
    args['poni'] = os.path.join(args['directory'], name+'.poni')
    # args['flat'] = os.path.join(args['directory'], name + '_flat.tif')
    args['flat'] = 'None'
    args['mask'] = os.path.join(args['directory'], name + '_mask.edf')
    patterns = integrate_data(**args)

    fig, ax = plt.subplots()
    xrd_datetimes = patterns.index.values
    t_mesh = extend_mesh(xrd_datetimes - xrd_datetimes[0]).astype(float) * 1e-9
    q_mesh = extend_mesh(patterns.columns.values.astype(float))
    ax.pcolormesh(t_mesh, q_mesh, patterns.values.T, cmap='magma')
    ax.set_xlabel('Time / s')
    ax.set_ylabel(r'Scattering vector $q$ / nm$^{-1}$')
