import sys
import pandas as pd
import numpy as np
import os

from datetime import datetime

from HelpFunctions import progress


def read_data(folder_path: str, header=True, date=datetime.now()) -> pd.DataFrame:
    files = [file_name for file_name in os.listdir(folder_path) if file_name[-4:] == '.txt']
    if len(files) == 0:
        sys.stdout.write('No .txt files in folder\n')
        sys.exit(-1)
    else:
        intensities = []
        date_times = []
        df = None
        for idx, file in enumerate(files):
            progress(idx + 1, len(files), 'reading files')
            file_path = os.path.join(folder_path, file)
            if header:
                # Get date:
                with open(file_path, 'r') as file_handle:
                    file_handle.readline()
                    file_handle.readline()
                    date_data = file_handle.readline()
                    date = datetime.strptime(date_data[6:], '%a %b %d %H:%M:%S CET %Y\n')
                skip_rows = 13
            else:
                skip_rows = 2
            df = pd.read_csv(file_path,
                             delim_whitespace=True,
                             decimal=",",
                             skiprows=skip_rows,
                             header=None,
                             names=['wavelength', 'intensity'])
            intensities.append(df['intensity'].values)
            date_times.append(datetime.combine(date.date(), datetime.strptime(file[-16:-4], '%H-%M-%S-%f').time()))
        intensities = np.array(intensities)
        times = np.array(date_times, dtype='datetime64')
        # times = (tmp - tmp[0]) / np.timedelta64(1, 's')
        wavelengths = df['wavelength'].values
        return pd.DataFrame(intensities, columns=wavelengths, index=times)


def subtract_dark(data: pd.DataFrame, dark: pd.DataFrame) -> pd.DataFrame:
    dark_average = np.mean(dark.values, axis=0)
    corrected = data - dark_average
    return corrected


def calculate_relative(data: pd.DataFrame, dark: pd.DataFrame, bright: pd.DataFrame) -> pd.DataFrame:
    bright_average = np.mean(bright.values, axis=0)
    dark_average = np.mean(dark.values, axis=0)
    relative = (data - dark_average) / (bright_average - dark_average)
    return relative


if __name__ == "__main__":
    sys.stdout.write('\nReading data\n')
    output_df = read_data(folder_path=sys.argv[1])
    if len(sys.argv) == 3:  # Subtract dark
        sys.stdout.write('\nReading dark\n')
        dark_data = read_data(folder_path=sys.argv[2])
        output_df = subtract_dark(output_df, dark=dark_data)
    elif len(sys.argv) > 3:  # Calculate relative
        sys.stdout.write('\nReading dark\n')
        dark_data = read_data(folder_path=sys.argv[2])
        sys.stdout.write('\nReading bright\n')
        bright_data = read_data(folder_path=sys.argv[3])
        output_df = calculate_relative(output_df, dark=dark_data, bright=bright_data)
    if len(sys.argv) == 7:  # Calculate nir relative
        sys.stdout.write('\nReading NIR data\n')
        nir_output_df = read_data(folder_path=sys.argv[4],
                                  header=False, date=pd.Timestamp(output_df.index.values[0]).to_pydatetime())
        sys.stdout.write('\nReading nir dark\n')
        nir_dark_data = read_data(folder_path=sys.argv[5], header=False)
        sys.stdout.write('\nReading nir bright\n')
        nir_bright_data = read_data(folder_path=sys.argv[6], header=False)
        nir_output_df = calculate_relative(nir_output_df, dark=nir_dark_data, bright=nir_bright_data)
        sys.stdout.write('\nSaving nir data...')
        nir_output_df.to_csv(os.path.join(sys.argv[1], 'nir_merged_data.csv'))
        sys.stdout.write(' done!\n')
    sys.stdout.write('\nSaving data...')
    output_df.to_csv(os.path.join(sys.argv[1], 'merged_data.csv'))
    sys.stdout.write(' done!\n')
