import sys

import numpy as np
import pandas as pd
from glob import glob
from os import path


def load_patterns(diectory: str) -> pd.DataFrame:
    """
    Function for loading all integrated patterns in a directory

    :param diectory: Path to directory from which to load the patterns
    :type diectory: str
    :return: A pandas data frame with scattering vector as columns and time as index
    :rtype: pandas.DataFrame
    """
    files = glob(path.join(diectory, '*_integrated.dat'))
    if len(files) == 0:
        raise ValueError('Directory: %s, does not contain any XRD patterns.' % diectory)
    else:
        q = np.loadtxt(files[0])[:, 0]
        data = np.zeros((len(files), len(q)))
        time = np.zeros(len(files))
        for idx, file in enumerate(files):
            data[idx, :] = np.loadtxt(file)[:, 1]
            time[idx] = idx  # TODO: Get time from log file
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
