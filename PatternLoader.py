import sys

import numpy as np
import pandas as pd
from glob import glob
from os import path
import re
import warnings
import matplotlib.pyplot as plt

from matplotlib import use
from HelpFunctions import progress, extend_mesh

use('Qt5Agg')


def load_patterns(directory: str) -> pd.DataFrame:
    """
    Function for loading all integrated patterns in a directory

    :param directory: Path to directory from which to load the patterns
    :type directory: str
    :return: A pandas data frame with scattering vector as columns and time as index
    :rtype: pandas.DataFrame
    """
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


def load_pvd_log(log_path: str):
    pvd_data = pd.read_csv(log_path,
                           delimiter='\t',
                           skiprows=7)
    with open(log_path, 'r') as file:
        content = file.readlines()
    date = re.findall(r': (.+?)\n', content[2])
    pvd_data['datetime'] = pd.to_datetime(date[0] + ' ' + pvd_data['Time'])
    pvd_data.set_index('datetime', inplace=True)
    return pvd_data


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit('ERROR: Not enough input parameters.')
    elif len(sys.argv) > 3:
        sys.exit('ERROR: Too many input parameters.')
    else:
        run_path = path.join(sys.argv[1], sys.argv[2])
        print(run_path)
        patterns = load_patterns(run_path)
        patterns.to_csv(path.join(run_path, sys.argv[2] + '_integrated.csv'))

        fig, axs = plt.subplots(2, 1, sharex=True)
        xrd_datetimes = patterns.index.values
        t = extend_mesh(xrd_datetimes - xrd_datetimes[0]).astype(float) * 1e-9
        q = extend_mesh(patterns.columns.values.astype(float))
        axs[1].pcolormesh(t, q, patterns.values.T)
        axs[1].set_xlabel('Time / s')
        axs[1].set_ylabel(r'Scattering vector $q$ / nm$^{-1}$')

        pvd_log = load_pvd_log(path.join(run_path, sys.argv[2] + '.csv'))
        pvd_datetimes = pvd_log.index.values
        t_pvd = (pvd_datetimes - xrd_datetimes[0]).astype(float) * 1e-9
        axs[0].plot(t_pvd, pvd_log['LinkamStage PV'])
        axs[0].set_ylabel(r'Hotplate temperature / °C')
        axs[1].set_xlim([t[0], t[-1]])
        fig.tight_layout()
        fig.savefig(path.join(run_path, sys.argv[2] + '_integrated.png'), dpi=300)
