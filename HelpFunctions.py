import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.optimize import curve_fit
from scipy.sparse.linalg import spsolve
from matplotlib import use

use('Qt5Agg')


def extend_mesh(x: np.ndarray) -> np.ndarray:
    """
    Function for extending mesh from center points to edge points

    :param x: Mesh to extend
    :type x: numpy.ndarray
    :return: The extended mesh
    :rtype: numpy.ndarray
    """
    x_delta = (np.diff(x)) / 2
    x_extended = x[:-1] + x_delta
    x_extended = np.insert(x_extended, 0, x[0] - x_delta[0])
    x_extended = np.append(x_extended, x[-1] + x_delta[-1])
    return x_extended


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


def baseline_als(y, lam, p, niter=10):
    """
    Baseline fit according to:
    https://stackoverflow.com/questions/29156532/python-baseline-correction-library

    :param y:
    :param lam:
    :param p:
    :param niter:
    :return:
    """
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + np.multiply(lam, D.dot(D.transpose()))
        z = spsolve(Z, np.multiply(w, y))
        w = np.multiply(p, (y > z)) + np.multiply((1 - p), (y < z))
    return z


def two_theta_conversion(two_theta, old_energy, new_energy):
    return 2 * 180 * np.arcsin(np.sin(np.pi * two_theta / 2 / 180) * old_energy / new_energy) / np.pi


def q_to_two_theta(q, energy):
    return 2 * 180 * np.arcsin(q * energy_to_wavelength(energy) / 4 / np.pi) / np.pi


def two_theta_to_q(two_theta, energy):
    return 4 * np.pi * np.sin(np.pi * two_theta / 180 / 2) / energy_to_wavelength(energy)


def energy_to_wavelength(energy):
    h = 4.135667696e-15  # eV * s
    c = 299792458.0e9  # nm / s
    if isinstance(energy, str):
        if energy == 'Ga':
            energy = 9.2517e3
        elif energy == 'Cu':
            energy = 8.0478e3
        else:
            raise NotImplementedError
    elif not isinstance(energy, (int, float, np.ndarray)):
        raise ValueError
    return h * c / energy


def wavelength_to_energy(wavelength):
    h = 4.135667696e-15  # eV * s
    c = 299792458.0e9  # nm / s
    if isinstance(wavelength, str):
        if wavelength == 'Ga':
            energy = 9.2517e3
        elif wavelength == 'Cu':
            energy = 8.0478e3
        else:
            raise NotImplementedError
    elif not isinstance(wavelength, (int, float, np.ndarray)):
        raise ValueError
    else:
        energy = h * c / wavelength
    return energy


def lattice_from_hkl(hkl: np.array, q: np.array):
    if hkl.shape != (3, 3):
        raise ValueError('Shape of hkl matrix needs to be 3x3')
    elif len(q) != 3:
        raise ValueError('q needs to be length 3')
    q = q.reshape((3, 1))
    y = np.power(q / 2 / np.pi, 2)
    return 1 / np.sqrt(np.matmul(np.linalg.inv(np.power(hkl, 2)), y))


def q_from_lattice(hkl: np.array, lattice: np.array):
    return 2 * np.pi * np.sqrt(np.sum(np.power(hkl.reshape(3, ), 2) / np.power(lattice.reshape(3, ), 2)))


def peak_fit(two_theta, intensities, guess=None, fwhm=0.1, plot_result=False):
    intensities = intensities.reshape(two_theta.shape)
    energy_ka1 = 9.2517e3
    energy_ka2 = 9.2248e3
    ratio = 0.51206
    f = double_peak_function(energy_ka1, energy_ka2, ratio, fwhm)
    if guess is None:
        guess = two_theta_to_q(two_theta[np.argmax(intensities)], energy_ka1)
    popt, pcov = curve_fit(f=f, xdata=two_theta, ydata=intensities, p0=[np.max(intensities), guess, 0, 0])

    if plot_result:
        fig, ax = plt.subplots()
        ax.plot(two_theta, intensities, 'o', alpha=0.5, color='C0', ms=3)
        ax.plot(two_theta, f(two_theta, *popt), color='C3', linewidth=1)
        two_theta1 = q_to_two_theta(popt[1], energy_ka1)
        zero1 = poly(two_theta1, popt[2], popt[3])
        two_theta2 = q_to_two_theta(popt[1], energy_ka2)
        zero2 = poly(two_theta2, popt[2], popt[3])
        ax.plot([two_theta1, two_theta1], [zero1, zero1 + popt[0]], color='C1')
        ax.plot([two_theta2, two_theta2], [zero2, zero2 + popt[0] * ratio], color='C1')

    return popt


def double_peak_function(energy_1, energy_2, ratio, fwhm):
    sigma = fwhm / 2 * np.sqrt(2 * np.log(2))

    def double_peak(x, *p):
        gaussian = gauss(x,
                         p[0], q_to_two_theta(p[1], energy_1), sigma,
                         p[0] * ratio, q_to_two_theta(p[1], energy_2), sigma)
        background = poly(x, p[2], p[3])
        return gaussian + background

    return double_peak


def track_peaks(two_theta, intensities, peaks, lattice_guess):
    energy_ka1 = 9.2517e3
    energy_ka2 = 9.2248e3
    fwhm = 0.1
    two_theta = two_theta.reshape(-1, )
    intensities = intensities.reshape(-1, len(two_theta))
    n_patterns = intensities.shape[0]
    fits = []
    for n, peak in enumerate(peaks):
        fit = []
        guess = q_from_lattice(np.array(peak), lattice_guess)
        for idx in range(n_patterns):
            progress(n * n_patterns + idx + 1, len(peaks) * n_patterns,
                     status='Tracking peak %d/%d' % (n + 1, len(peaks)))
            if q_to_two_theta(guess, energy_ka1) > two_theta[0] and q_to_two_theta(guess, energy_ka2) < two_theta[-1]:
                two_theta_high = q_to_two_theta(guess, energy_ka2) + (2 * fwhm)
                two_theta_low = q_to_two_theta(guess, energy_ka1) - (3 * fwhm)
                roi = np.where((two_theta < two_theta_high) * (two_theta > two_theta_low))
                try:
                    single_fit = peak_fit(two_theta[roi], intensities[idx, roi], guess=guess, fwhm=fwhm,
                                          plot_result=False)
                    high = q_to_two_theta(single_fit[1], energy_ka2)
                    low = q_to_two_theta(single_fit[1], energy_ka1)
                    if low > two_theta[roi][0] and high < two_theta[roi][-1]:
                        fit.append(single_fit)
                    else:
                        fit.append((np.nan, np.nan, np.nan, np.nan))
                    # print('Guessed %f, fitted %f' % (guess, fit[-1]))
                except RuntimeError:
                    fit.append((np.nan, np.nan, np.nan, np.nan))
            else:
                fit.append((np.nan, np.nan, np.nan, np.nan))
            guess = fit[-1][1]
        fits.append(fit)
    return fits


def gauss(x, *p):
    """Sum of gaussians:
        p = [Intenisty at peak, peak position, peak variance]
    """
    y = 0
    args = len(p)
    if args % 3:
        return 0
    else:
        n = int(args / 3)
        for k in range(n):
            y += p[3 * k] * np.exp(-np.power((x - p[3 * k + 1]), 2.) / (2. * p[3 * k + 2] ** 2.))
        return y


def poly(x, *p):
    y = 0
    for n in range(len(p)):
        y += p[n] * np.power(x, n)
    return y


def gaussian_max(x_data, data):
    intensity = []
    position = []
    fwhm = []
    for line in range(data.shape[0]):
        popt, pcov = curve_fit(lambda x, *p: gauss(x, p[0], p[1], p[2]) + poly(x, p[3], p[4]),
                               xdata=x_data, ydata=data[line, :], p0=[np.max(data[line, :]),
                                                                      x_data[np.argmax(data[line, :])], 25, 0, 0])
        intensity.append(popt[0])
        position.append(popt[1])
        fwhm.append(popt[2] * 2 * np.sqrt(2 * np.log(2)))
    return np.array(intensity), np.array(position), np.array(fwhm)


def find_closest(array, target):
    """
    Taken from comment by Bi Rico on
    https://stackoverflow.com/questions/8914491/finding-the-nearest-value-and-return-the-index-of-array-in-python

    :param array:
    :type array:
    :param target:
    :type target:
    :return:
    :rtype:
    """
    idx = array.searchsorted(target)
    idx = np.clip(idx, 1, len(array)-1)
    left = array[idx-1]
    right = array[idx]
    idx -= target - left < right - target
    return idx
