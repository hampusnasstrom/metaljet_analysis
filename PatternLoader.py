import sys

import numpy as np
import pandas as pd
from glob import glob
from os import path


def progress(count: int, total: int, status='') -> None:
    """
    Progress bar for sys

    :param count: Current count
    :type count: int
    :param total: Total counts
    :type total: int
    :param status: Optional status string to display
    :type status: str
    :return: None
    """
    bar_len = 30
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('\r[%s] %4.1f%s ...%s' % (bar, percents, '%', status))
    sys.stdout.flush()


def load_patterns(directory: str) -> pd.DataFrame:
    """
    Function for loading all integrated patterns in a directory

    :param directory: Path to directory from which to load the patterns
    :type directory: str
    :return: A pandas data frame with scattering vector as columns and time as index
    :rtype: pandas.DataFrame
    """
    files = glob(path.join(directory, '*_integrated.dat'))
    print(path.join(directory, '*_integrated.dat'))
    n_files = len(files)
    if n_files == 0:
        raise ValueError('Directory: %s, does not contain any XRD patterns.' % directory)
    else:
        q = np.loadtxt(files[0])[:, 0]
        data = np.zeros((n_files, len(q)))
        time = np.zeros(n_files)
        for idx, file in enumerate(files):
            progress(idx, n_files, status='reading files')
            data[idx, :] = np.loadtxt(file)[:, 1]
            time[idx] = idx  # TODO: Get time from log file
        progress(idx + 1, n_files, status='done')
        df = pd.DataFrame(data=data, index=time, columns=q)
        return df


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('ERROR: Not enough input parameters.')
    elif len(sys.argv) > 3:
        sys.exit('ERROR: Too many input parameters.')
    else:
        patterns = load_patterns(sys.argv[1])
        if len(sys.argv) >= 3:
            patterns.to_csv(sys.argv[2])
