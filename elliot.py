import numpy as np

from typing import Union

from scipy.fftpack import hilbert
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit, minimize, basinhopping
from scipy.signal import fftconvolve
from scipy.special import wofz, polygamma, digamma

from numpy.core.multiarray import ndarray

h_bar = 6.582119569e-16  # eV*s
h = 4.135667696e-15  # eV*s
c = 299792458.0  # m/s
k = 8.617333262145e-5


def alpha_exciton(energy, exciton_energy, bandgap_energy, gaussian_hwhm, lorentzian_hwhm, scale_factor, excitons=5):
    # Create row vector for summation index n
    n = np.arange(1, excitons + 1, dtype=float).reshape(1, -1)
    # Create matrix of Voigt profiles with specified HWHMs
    profiles = voigt(energy.reshape(-1, 1) - (bandgap_energy - exciton_energy / np.power(n, 2)),
                     gaussian_hwhm, lorentzian_hwhm)
    scale = 1 / ((np.power(n[:, :-1], 3) * bandgap_energy) - (n[:, :-1] * exciton_energy))
    scale = np.append(scale, [[-1 * polygamma(2, excitons) / (2 * bandgap_energy)]]) * 2 * scale_factor * np.power(
        exciton_energy, 3 / 2)
    scaled_profiles = scale * profiles
    return np.sum(scaled_profiles, axis=1)


def alpha_exciton_lorentzian(energy, exciton_energy, bandgap_energy, hwhm, scale_factor, excitons=5):
    # Create row vector for summation index n
    n = np.arange(1, excitons + 1, dtype=float).reshape(1, -1)
    # Create matrix of Voigt profiles with specified HWHMs
    profiles = lorentzian(energy.reshape(-1, 1) - (bandgap_energy - exciton_energy / np.power(n, 2)), hwhm)
    scale = 1 / ((np.power(n[:, :-1], 3) * bandgap_energy) - (n[:, :-1] * exciton_energy))
    scale = np.append(scale, [[-1 * polygamma(2, excitons) / (2 * bandgap_energy)]]) * 2 * scale_factor * np.power(
        exciton_energy, 3 / 2)
    scaled_profiles = scale * profiles
    return np.sum(scaled_profiles, axis=1)


def alpha_continuum_lorentzian(energy, exciton_energy, bandgap_energy, hwhm, scale_factor):
    roi = np.where(energy > bandgap_energy)
    above_bandgap = 1 / (
            energy[roi] * (1 - np.exp(-2 * np.pi * np.sqrt(exciton_energy / (energy[roi] - bandgap_energy)))))
    below_bandgap = np.zeros(len(energy) - len(energy[roi]))
    total = scale_factor * np.sqrt(exciton_energy) * np.concatenate([below_bandgap, above_bandgap])
    mid = int(len(energy) / 2)
    broadening = lorentzian(energy - energy[mid], hwhm)
    return np.convolve(broadening, total, mode='same') * (energy[1] - energy[0])


def alpha_exciton_scaled(energy, exciton_energy, bandgap_energy, gaussian_hwhm, lorentzian_hwhm, scale_factor,
                         excitons=5, continuum_gaussian_hwhm=None, continuum_lorentzian_hwhm=None):
    # Create row vector for summation index n
    n = np.arange(1, excitons + 1, dtype=float).reshape(1, -1)
    # Scale the HWHMs according to doi: 10.1103/PhysRevB.22.6162
    if continuum_gaussian_hwhm is not None:
        exciton_gaussian_hwhm = gaussian_hwhm
        gaussian_hwhm = continuum_gaussian_hwhm - (continuum_gaussian_hwhm - exciton_gaussian_hwhm) / np.power(n, 2)
    if continuum_lorentzian_hwhm is not None:
        exciton_lorentzian_hwhm = lorentzian_hwhm
        lorentzian_hwhm = continuum_lorentzian_hwhm - (continuum_lorentzian_hwhm - exciton_lorentzian_hwhm) / np.power(
            n, 2)
    # Create matrix of Voigt profiles with specified HWHMs
    profiles = voigt(energy.reshape(-1, 1) - (bandgap_energy - exciton_energy / np.power(n, 2)),
                     gaussian_hwhm, lorentzian_hwhm)
    scale = 1 / ((np.power(n[:, :-1], 3) * bandgap_energy) - (n[:, :-1] * exciton_energy))
    scale = np.append(scale, [[-1 * polygamma(2, excitons) / (2 * bandgap_energy)]]) * 2 * scale_factor * np.power(
        exciton_energy, 3 / 2)
    scaled_profiles = scale * profiles
    return np.sum(scaled_profiles, axis=1)


def alpha_continuum(energy, exciton_energy, bandgap_energy, gaussian_hwhm, lorentzian_hwhm, scale_factor):
    roi = np.where(energy > bandgap_energy)
    above_bandgap = 1 / (
            energy[roi] * (1 - np.exp(-2 * np.pi * np.sqrt(exciton_energy / (energy[roi] - bandgap_energy)))))
    below_bandgap = np.zeros(len(energy) - len(energy[roi]))
    total = scale_factor * np.sqrt(exciton_energy) * np.concatenate([below_bandgap, above_bandgap])
    mid = int(len(energy) / 2)
    broadening = voigt(energy - energy[mid], gaussian_hwhm, lorentzian_hwhm)
    return np.convolve(broadening, total, mode='same') * (energy[1] - energy[0])


def alpha_exciton_koch(energy, exciton_energy, bandgap_energy, gaussian_hwhm, lorentzian_hwhm, scale_factor,
                       excitons=5):
    # Create row vector for summation index n
    n = np.arange(1, excitons + 1, dtype=float).reshape(1, -1)
    # Create matrix of Voigt profiles with specified HWHMs
    profiles = voigt(energy.reshape(-1, 1) - (bandgap_energy - exciton_energy / np.power(n, 2)),
                     gaussian_hwhm, lorentzian_hwhm)
    # scale = 1 / np.power(n[:, :-1], 3)
    scale = (bandgap_energy - (np.power(n[:, :-1], 2) * exciton_energy)) / np.power(n[:, :-1], 3)
    scale = (np.append(scale, [[-1 * polygamma(2, excitons) / (2 * bandgap_energy)]]) * 4 * scale_factor * np.pi)
    scaled_profiles = scale * profiles
    return np.sum(scaled_profiles, axis=1)


def alpha_exciton_scaled_koch(energy, exciton_energy, bandgap_energy, gaussian_hwhm, lorentzian_hwhm, scale_factor,
                              excitons=5, continuum_gaussian_hwhm=None, continuum_lorentzian_hwhm=None):
    # Create row vector for summation index n
    n = np.arange(1, excitons + 1, dtype=float).reshape(1, -1)
    # Scale the HWHMs according to doi: 10.1103/PhysRevB.22.6162
    if continuum_gaussian_hwhm is not None:
        exciton_gaussian_hwhm = gaussian_hwhm
        gaussian_hwhm = continuum_gaussian_hwhm - (continuum_gaussian_hwhm - exciton_gaussian_hwhm) / np.power(n, 2)
    if continuum_lorentzian_hwhm is not None:
        exciton_lorentzian_hwhm = lorentzian_hwhm
        lorentzian_hwhm = continuum_lorentzian_hwhm - (continuum_lorentzian_hwhm - exciton_lorentzian_hwhm) / np.power(
            n, 2)
    # Create matrix of Voigt profiles with specified HWHMs
    profiles = voigt(energy.reshape(-1, 1) - (bandgap_energy - exciton_energy / np.power(n, 2)),
                     gaussian_hwhm, lorentzian_hwhm)
    scale = (bandgap_energy - (np.power(n[:, :-1], 2) * exciton_energy)) / np.power(n[:, :-1], 3)
    scale = (np.append(scale, [[-1 * polygamma(2, excitons) / (2 * bandgap_energy)]]) * 4 * scale_factor * np.pi)
    scaled_profiles = scale * profiles
    return np.sum(scaled_profiles, axis=1)


def alpha_continuum_koch(energy, exciton_energy, bandgap_energy, gaussian_hwhm, lorentzian_hwhm, scale_factor):
    roi = np.where(energy > bandgap_energy)
    # above_bandgap = (1 /
    #                  (1 - np.exp(-2 * np.pi * np.sqrt(exciton_energy / (energy[roi] - bandgap_energy)))))
    above_bandgap = (energy[roi] /
                     (1 - np.exp(-2 * np.pi * np.sqrt(exciton_energy / (energy[roi] - bandgap_energy)))))
    below_bandgap = np.zeros(len(energy) - len(energy[roi]))
    total = scale_factor * np.pi * np.concatenate([below_bandgap, above_bandgap]) / exciton_energy
    mid = int(len(energy) / 2)
    broadening = voigt(energy - energy[mid], gaussian_hwhm, lorentzian_hwhm)
    return np.convolve(broadening, total, mode='same') * (energy[1] - energy[0])


def alpha_sum_koch(energy, exciton_energy, bandgap_energy, exciton_gaussian_hwhm, exciton_lorentzian_hwhm,
                   continuum_gaussian_hwhm, continuum_lorentzian_hwhm, scale_factor, excitons=5):
    convolution_energy = np.linspace(0.1, 10, 5001)
    continuum_interpolation = interp1d(convolution_energy,
                                       alpha_continuum_koch(convolution_energy, exciton_energy, bandgap_energy,
                                                            continuum_gaussian_hwhm, continuum_lorentzian_hwhm,
                                                            100 * scale_factor),
                                       kind='cubic')
    return continuum_interpolation(energy) + alpha_exciton_scaled_koch(energy, exciton_energy, bandgap_energy,
                                                                       exciton_gaussian_hwhm, exciton_lorentzian_hwhm,
                                                                       100 * scale_factor, excitons=excitons,
                                                                       continuum_gaussian_hwhm=continuum_gaussian_hwhm,
                                                                       continuum_lorentzian_hwhm=continuum_lorentzian_hwhm)


def alpha_sum(energy, exciton_energy, bandgap_energy, exciton_gaussian_hwhm, exciton_lorentzian_hwhm,
              continuum_gaussian_hwhm, continuum_lorentzian_hwhm, scale_factor, excitons=5):
    convolution_energy = np.linspace(0.1, 10, 5001)
    continuum_interpolation = interp1d(convolution_energy,
                                       alpha_continuum(convolution_energy, exciton_energy, bandgap_energy,
                                                       continuum_gaussian_hwhm, continuum_lorentzian_hwhm,
                                                       100 * scale_factor),
                                       kind='cubic')
    return continuum_interpolation(energy) + alpha_exciton_scaled(energy, exciton_energy, bandgap_energy,
                                                                  exciton_gaussian_hwhm, exciton_lorentzian_hwhm,
                                                                  100 * scale_factor, excitons=excitons,
                                                                  continuum_gaussian_hwhm=continuum_gaussian_hwhm,
                                                                  continuum_lorentzian_hwhm=continuum_lorentzian_hwhm)


def transmission(wavelength, exciton_energy, bandgap_energy, exciton_gaussian_hwhm, exciton_lorentzian_hwhm,
                 continuum_gaussian_hwhm, continuum_lorentzian_hwhm, scale_factor, thickness, n_inf,
                 excitons=5):
    convolution_energy = np.linspace(0.1, 10, 5001)
    alpha = (alpha_continuum(convolution_energy, exciton_energy, bandgap_energy,
                             continuum_gaussian_hwhm, continuum_lorentzian_hwhm,
                             1e6 * scale_factor)
             + alpha_exciton_scaled(convolution_energy, exciton_energy, bandgap_energy,
                                    exciton_gaussian_hwhm, exciton_lorentzian_hwhm,
                                    1e6 * scale_factor, excitons=excitons,
                                    continuum_gaussian_hwhm=continuum_gaussian_hwhm,
                                    continuum_lorentzian_hwhm=continuum_lorentzian_hwhm))
    k_theo = 1e2 * alpha * h_bar * c / convolution_energy / 2
    n_theo = interp1d(convolution_energy, n_inf + hilbert(k_theo), kind='cubic')
    k_theo = interp1d(convolution_energy, k_theo, kind='cubic')
    energy = 1e6 * h * c / wavelength
    n = n_theo(energy) + 1j * k_theo(energy)
    return tmm_transmission(np.array([np.ones(n.shape), n, 1.45 * np.ones(n.shape)]), wavelength, thickness)


def tmm_transmission(n, wavelengths, d):
    n_sum = n[:-1] + n[1:]
    r = (n[1:] - n[:-1]) / n_sum
    t = 2 * n[:-1] / n_sum
    delta = d * 2 * np.pi * n[1:-1] / wavelengths

    return np.power(np.abs(t[0] * t[1] / (np.exp(-1j * delta[0]) + r[0] * r[1] * np.exp(1j * delta[0]))), 2) * n[
        -1].real


def get_alpha_fit_function(exciton_energy=None, bandgap_energy=None, exciton_gaussian_hwhm=None,
                           exciton_lorentzian_hwhm=None, continuum_gaussian_hwhm=None, continuum_lorentzian_hwhm=None,
                           scale_factor=None, excitons=10):
    def alpha_fit(x, *p):
        next_p = 0
        parameters = [exciton_energy, bandgap_energy, exciton_gaussian_hwhm, exciton_lorentzian_hwhm,
                      continuum_gaussian_hwhm, continuum_lorentzian_hwhm, scale_factor]
        p_used = []
        for parameter in parameters:
            if parameter is None:
                p_used.append(p[next_p])
                next_p += 1
            else:
                p_used.append(parameter)
        return alpha_sum(x, *p_used, excitons=excitons)

    return alpha_fit


def alpha_minimization_fit(x, y, parameter_guess, ax):
    def upon_minimization(minimized_params, f, accept):
        if accept:
            print(minimized_params)
            print(f)
            ax.plot(x, alpha_sum(x, *minimized_params, excitons=10), label='f=%.5f' % f)

    def fun(p):
        return np.sum(np.power(y - alpha_sum(x, *p, excitons=10), 2))

    bounds = [(0.00001, np.inf)] * len(parameter_guess)
    p_opt = basinhopping(fun, parameter_guess, stepsize=0.05, minimizer_kwargs={"bounds": bounds}, disp=True,
                         callback=upon_minimization)
    return p_opt


def gaussian(x: Union[int, float, ndarray], hwhm: float) -> Union[int, float, ndarray]:
    """
    Calculate the Gaussian profile at value(s) x
    :param x: Value(s) for which to evaluate the profile
    :type x: Union[int, float, ndarray]
    :param hwhm: Half width at half maximum of the profile
    :type hwhm: float
    :return: Evaluation of the profile at the x values
    :rtype: Union[int, float, ndarray]
    """
    return np.sqrt(np.log(2) / np.pi) / hwhm * np.exp(-1 * np.power((x / hwhm), 2) * np.log(2))


def lorentzian(x: Union[int, float, ndarray], hwhm: float) -> Union[int, float, ndarray]:
    """
    Calculate the Lorentzian profile at value(s) x
    :param x: Value(s) for which to evaluate the profile
    :type x: Union[int, float, ndarray]
    :param hwhm: Half width at half maximum of the profile
    :type hwhm: float
    :return: Evaluation of the profile at the x values
    :rtype: Union[int, float, ndarray]
    """
    return hwhm / np.pi / (np.power(x, 2) + np.power(hwhm, 2))


def voigt(x: Union[int, float, ndarray], gaussian_hwhm: float, lorentzian_hwhm: float) -> Union[int, float, ndarray]:
    """
    Calculate the Voigt profile at value(s) x.
    Source: https://scipython.com/book/chapter-8-scipy/examples/the-voigt-profile/
    :param x: Value(s) for which to evaluate the profile
    :type x: Union[int, float, ndarray]
    :param gaussian_hwhm: Half width at half maximum of the Gaussian profile
    :type gaussian_hwhm: float
    :param lorentzian_hwhm: Half width at half maximum of the Gaussian profile
    :type lorentzian_hwhm: float
    :return: Evaluation of the profile at the x values
    :rtype: Union[int, float, ndarray]
    """
    sigma = gaussian_hwhm / np.sqrt(2 * np.log(2))

    return np.real(wofz((x + 1j * lorentzian_hwhm) / sigma / np.sqrt(2))) / sigma / np.sqrt(2 * np.pi)


def log_normal(x: ndarray, sigma: float, mean: float):
    """
    Calculate a log normal distribution with specified standard deviation and mean. The distribution is assumed to be
    zero for negative values.
    :param x: Array of strictly increasing x values
    :type x: ndarray
    :param sigma: Standard deviation of the distribution
    :type sigma: float
    :param mean: Mean of the distribution
    :type mean: float
    :return: Evaluation of the profile at the x values
    :rtype: ndarray
    """
    x = x + np.exp(mean)
    roi = np.where(x > 0)
    below = np.zeros(len(x) - len(x[roi]))
    above = (np.exp(-1 * np.power(np.log(x[roi]) - mean, 2) / (2 * np.power(sigma, 2)))
             / (x[roi] * sigma * np.sqrt(2 * np.pi)))
    return np.concatenate([below, above])


def log_normal_normal(x, gaussian_hwhm, log_normal_sigma, log_normal_mean):
    g = np.convolve(log_normal(x, log_normal_sigma, log_normal_mean), gaussian(x, gaussian_hwhm),
                    mode='same') * (x[1] - x[0])
    return g


def tanguy(energy, exciton_energy, bandgap_energy, broadening, scale_factor):
    def eta(z):
        return np.sqrt(exciton_energy / (bandgap_energy - z))

    def g(z):
        eta_val = eta(z)
        return 2 * np.log(eta_val) - 2 * np.pi / np.tan(np.pi * eta_val) - 2 * digamma(eta_val) - 1 / eta_val

    arg = energy + 1j * broadening
    return scale_factor * np.sqrt(exciton_energy) * (g(arg) + g(-arg) - 2 * g(0)) / np.power(arg, 2)


def refractive_index(epsilon):
    return np.sqrt((epsilon.real + np.sqrt(np.power(epsilon.real, 2) + np.power(epsilon.imag, 2))))


def absorption_coefficient(energy, epsilon):
    return energy * epsilon.imag / (refractive_index(epsilon) * h_bar * c * 1e2)


def abs_coef_fit_function(x, *p):
    evaluation_points = 1001
    convolution_energy = np.linspace(min(0.1, min(x) - p[2]), max(x) + p[2], evaluation_points)
    delta_energy = np.abs(convolution_energy[1] - convolution_energy[0])
    energy_relative = convolution_energy - convolution_energy[int(evaluation_points / 2)]
    f_broad = np.convolve(gaussian(energy_relative, p[3]), log_normal(energy_relative, p[2], p[4]),
                          mode='same') * delta_energy
    alpha = interp1d(convolution_energy,
                     absorption_coefficient(convolution_energy,
                                            1.3 ** 2 + np.convolve(f_broad,
                                                                   tanguy(convolution_energy, p[0], p[1], 0.001, p[5]),
                                                                   mode='same') * delta_energy),
                     kind='cubic')
    return alpha(x)


def transmittance_fit_function(energy, exciton_energy, bandgap_energy, lognorm_sigma, norm_sigma, lognorm_mu,
                               scale_factor,
                               n_bgr, thickness):
    evaluation_points = 1001
    convolution_energy = np.linspace(min(0.1, min(energy) - lognorm_sigma),
                                     max(energy) + lognorm_sigma, evaluation_points)
    delta_energy = np.abs(convolution_energy[1] - convolution_energy[0])
    energy_relative = convolution_energy - convolution_energy[int(evaluation_points / 2)]
    f_broad = np.convolve(gaussian(energy_relative, norm_sigma), log_normal(energy_relative, lognorm_sigma, lognorm_mu),
                          mode='same') * delta_energy
    epsilon = np.power(n_bgr, 2) + np.convolve(f_broad, tanguy(convolution_energy, exciton_energy, bandgap_energy,
                                                               0.001, scale_factor),
                                               mode='same') * delta_energy
    n = refractive_index(epsilon)
    n_real = interp1d(convolution_energy, n, kind='cubic')
    n_imag = interp1d(convolution_energy, epsilon.imag / (2 * n), kind='cubic')
    n = n_real(energy) + 1j * n_imag(energy)
    return (tmm_transmission(np.array([np.ones(n.shape), n, 1.45 * np.ones(n.shape)]), 1e6 * h * c / energy, thickness),
            tmm_reflectance(np.array([np.ones(n.shape), n, 1.45 * np.ones(n.shape)]), 1e6 * h * c / energy, thickness))


def tmm_reflectance(n, wavelengths, d):
    n_sum = n[:-1] + n[1:]
    r = (n[1:] - n[:-1]) / n_sum
    t = 2 * n[:-1] / n_sum
    delta = d * 2 * np.pi * n[1:-1] / wavelengths

    return np.power(np.abs(((r[0] * np.exp(-1j * delta[0]) + r[1] * np.exp(1j * delta[0])) /
                            (np.exp(-1j * delta[0]) + r[0] * r[1] * np.exp(1j * delta[0])))), 2)


def tanguy_tmm_transmittance(energy, convolution_energy, energy_relative, delta_energy, exciton_energy, bandgap_energy,
                             gaussian_hwhm, lorentzian_hwhm, scale_factor, n_bgr, thickness, roughness):
    epsilon = np.power(n_bgr, 2) + fftconvolve(tanguy(convolution_energy, exciton_energy, bandgap_energy,
                                                      lorentzian_hwhm, scale_factor),
                                               gaussian(energy_relative, gaussian_hwhm),
                                               mode='same') * delta_energy
    n = refractive_index(epsilon)
    n_real = interp1d(convolution_energy, n, kind='cubic')
    n_imag = interp1d(convolution_energy, epsilon.imag / (2 * n), kind='cubic')
    n = n_real(energy) + 1j * n_imag(energy)
    wavelength = 1e6 * h * c / energy  # um
    transmittance = tmm_transmission(np.array([np.ones(n.shape), n, 1.45 * np.ones(n.shape)]), wavelength,
                                     thickness) * np.exp(-1 * np.power(2 * np.pi * roughness / wavelength, 2))
    return transmittance


def get_transmittance_fit_function(gaussian_hwhm_max, lorentzian_hwhm_max, lorentzian_hwhm_min, energy_max, energy_min,
                                   convolution_error=1e-4, exciton_energy=None, bandgap_energy=None, gaussian_hwhm=None,
                                   lorentzian_hwhm=None, scale_factor=None, n_bgr=None, thickness=None, roughness=None):
    evaluation_points = int(5 * (4 - 0.1) / (2 * lorentzian_hwhm_min) // 2 * 2 + 1)
    convolution_energy = np.linspace(0.1, 4, evaluation_points)
    energy_relative = convolution_energy - convolution_energy[int(evaluation_points / 2)]
    delta_energy = np.abs(convolution_energy[1] - convolution_energy[0])

    def transmittance_fit(x, *p):
        next_p = 0
        parameters = [exciton_energy, bandgap_energy, gaussian_hwhm, lorentzian_hwhm, scale_factor, n_bgr, thickness,
                      roughness]
        p_used = []
        for parameter in parameters:
            if parameter is None:
                p_used.append(p[next_p])
                next_p += 1
            else:
                p_used.append(parameter)
        return tanguy_tmm_transmittance(x, convolution_energy, energy_relative, delta_energy, *p_used)

    return transmittance_fit


if __name__ == "__main__":
    test_args = {
        'energy': np.linspace(0.1, 10, 5001),
        'exciton_energy': 0.0205,
        'bandgap_energy': 1.656,
        'gaussian_hwhm': 0.012,
        'lorentzian_hwhm': 0.00,
        'scale_factor': 2 * 26.4
    }
    # test_args['energy'] = np.linspace(test_args['bandgap_energy'] - 2 * test_args['exciton_energy'],
    #                                   test_args['bandgap_energy'] + 2 * test_args['exciton_energy'], 1000)
    # test_args['energy'] = np.linspace(test_args['bandgap_energy'] - 0.2,
    #                                   test_args['bandgap_energy'] + 0.2, 1000)

    import matplotlib.pyplot as plt
    from matplotlib import use

    use('Qt5Agg')
    fig, ax = plt.subplots()
    energy_val = np.linspace(1, 2, 500)
    epsilon = tanguy(energy_val, 0.004, 1.42, 0.006, 1)
    ax.plot(energy_val, epsilon.real)
    ax.plot(energy_val, epsilon.imag)
    fig2, ax2 = plt.subplots()
    ax2.plot(energy_val, refractive_index(epsilon))
    ax2.plot(energy_val, absorption_coefficient(energy_val, epsilon))

    # Ns = np.geomspace(3, 1000, 20).astype(int)
    # widths = [0.1, 0.01, 0.001]
    # for width in widths:
    #     test_args['gaussian_hwhm'] = width
    #     test_args['lorentzian_hwhm'] = width
    #     gt = np.sum(alpha_exciton(**test_args, excitons=10000), axis=1)
    #     error = [np.max(np.abs(alpha_exciton(**test_args, excitons=N)-gt)) for N in Ns]
    #     ax.plot(Ns, error, '.-', label='HWHM$=%d$ meV' % (width * 1e3))
    # ax.set_yscale('log')
    # ax.set_xscale('log')
    # ax.set_xlabel('Exciton positions, $N$')
    # ax.set_ylabel('|Maximum error|')
    # ax.legend()
    # ax.set_title('Error for %d meV binding energy' % (test_args['exciton_energy'] * 1e3))
    #

    # N = 10
    # cont_hwhm = 0.02
    # alpha_exc = alpha_exciton_scaled_koch(**test_args, excitons=N, continuum_gaussian_hwhm=cont_hwhm)
    # test_args['gaussian_hwhm'] = cont_hwhm
    # alpha_cont = alpha_continuum_koch(**test_args)
    # ax.plot(test_args['energy'], alpha_exc, label=r'$\alpha_\mathrm{X}$')
    # ax.plot(test_args['energy'], alpha_cont, label=r'$\alpha_\mathrm{C}$')
    # ax.plot(test_args['energy'], (alpha_exc + alpha_cont), label=r'Sum')
    # ax.set_xlim(1.55, 1.75)
    # ax.legend()
    #
    test_x = [1.7594059405940594, 1.7544554455445545, 1.7496699669966997, 1.7457095709570958, 1.741089108910891,
              1.7374587458745874, 1.7333333333333334, 1.7295379537953797, 1.7262376237623762, 1.722937293729373,
              1.7174917491749175, 1.7125412541254126, 1.7066006600660066, 1.697029702970297, 1.6897689768976898,
              1.686138613861386, 1.6836633663366336, 1.6815181518151816, 1.6790429042904291, 1.6757425742574257,
              1.6732673267326734, 1.6711221122112212, 1.6686468646864687, 1.666006600660066, 1.6643564356435645,
              1.6622112211221123, 1.660891089108911, 1.6594059405940595, 1.657920792079208, 1.6566006600660066,
              1.6552805280528053, 1.6541254125412541, 1.6531353135313531, 1.6521452145214521, 1.6506600660066006,
              1.64983498349835, 1.6486798679867987, 1.6476897689768977, 1.6463696369636964, 1.6455445544554457,
              1.6442244224422442, 1.6433993399339935, 1.641914191419142, 1.640924092409241, 1.63993399339934,
              1.6382838283828383, 1.6367986798679868, 1.6348184818481848, 1.6338283828382838, 1.6325082508250826,
              1.6313531353135313, 1.6301980198019803, 1.6288778877887788, 1.6275577557755776, 1.6265676567656766,
              1.6252475247524754, 1.6240924092409241, 1.6227722772277229, 1.6216171617161717, 1.6206270627062707,
              1.6193069306930694, 1.617821782178218, 1.6169966996699672, 1.615841584158416, 1.6145214521452147,
              1.6133663366336635, 1.6120462046204622, 1.6105610561056107, 1.6090759075907592, 1.6074257425742575,
              1.6056105610561058, 1.6037953795379538, 1.601980198019802, 1.5983498349834984, 1.5952145214521454,
              1.592739273927393, 1.5902640264026404, 1.5871287128712872, 1.5843234323432345, 1.5821782178217823,
              1.5795379537953795, 1.575907590759076, 1.5717821782178218]
    test_y = [2.7230483271375463, 2.7063197026022303, 2.6895910780669143, 2.678438661710037, 2.6617100371747213,
              2.650557620817844, 2.6394052044609664, 2.633828996282528, 2.6226765799256504, 2.611524163568773,
              2.600371747211896, 2.5892193308550184, 2.578066914498141, 2.578066914498141, 2.58364312267658,
              2.5892193308550184, 2.5947955390334574, 2.6059479553903344, 2.611524163568773, 2.633828996282528,
              2.6561338289962824, 2.678438661710037, 2.7063197026022303, 2.745353159851301, 2.7788104089219328,
              2.8178438661710037, 2.856877323420074, 2.895910780669145, 2.9460966542750926, 2.9851301115241635,
              3.029739776951673, 3.074349442379182, 3.12453531598513, 3.174721189591078, 3.2304832713754648,
              3.2750929368029738, 3.336431226765799, 3.3977695167286246, 3.470260223048327, 3.5260223048327135,
              3.5985130111524164, 3.6598513011152414, 3.7379182156133828, 3.8104089219330852, 3.877323420074349,
              3.927509293680297, 3.960966542750929, 3.949814126394052, 3.882899628252788, 3.7936802973977692,
              3.6486988847583643, 3.4479553903345725, 3.2193308550185873, 2.9516728624535316, 2.66728624535316,
              2.3550185873605947, 2.0650557620817844, 1.7695167286245352, 1.5018587360594795, 1.2509293680297398,
              1.033457249070632, 0.8159851301115242, 0.6654275092936803, 0.5315985130111525, 0.4256505576208178,
              0.33085501858736066, 0.26951672862453535, 0.20260223048327142, 0.1579925650557621, 0.11338289962825288,
              0.08550185873605953, 0.06319702602230493, 0.04646840148698894, 0.03531598513011158, 0.03531598513011158,
              0.024163568773234223, 0.024163568773234223, 0.0185873605947956, 0.024163568773234223, 0.0185873605947956,
              0.0185873605947956, 0.0185873605947956, 0.029739776951672958]

    # wavelengths = 1e6 * h * c / np.array(test_x)
    fig3, ax3 = plt.subplots(figsize=[6, 5])
    E_eval = np.linspace(test_x[-1], test_x[0], 1001)
    ax3.plot(np.array(test_x), 1e4 * np.array(test_y), 'o', alpha=0.5, label='Davies et al.')
    alpha = absorption_coefficient(E_eval, 2.1 ** 2 + tanguy(E_eval, 0.022, 1.658, 0.001, 3))
    f_broad = np.convolve(gaussian(E_eval - E_eval[int(len(E_eval) / 2)], 0.007),
                          log_normal(E_eval - E_eval[int(len(E_eval) / 2)], 0.6, -4), mode='same') * np.abs(
        E_eval[1] - E_eval[0])
    # ax3.plot(E_eval, alpha)
    # ax3.plot(E_eval, np.convolve(f_broad, alpha, mode='same') * np.abs(E_eval[1] - E_eval[0]))

    p_opt, p_cov = curve_fit(abs_coef_fit_function, np.array(test_x), 1e4 * np.array(test_y), p0=[0.022, 1.658, 0.6, 0.007, -4, 3])
    ax3.plot(E_eval, absorption_coefficient(E_eval, 2.1 ** 2 + tanguy(E_eval, p_opt[0], p_opt[1], 0.001, p_opt[5])),
             label=r'Tanguy w. $\Gamma=1$ meV')
    ax3.plot(E_eval, abs_coef_fit_function(E_eval, *p_opt), label='Tanguy broadened w.\nNormal and Lognormal')
    ax3.legend()
    ax3.set_ylim([-0.3e4, 5e4])
    ax3.set_ylabel(r'Absorption coefficient, $\alpha$ / cm$^{-1}$')
    ax3.set_xlabel(r'Photon energy, $\hbar\omega$ / eV')
    ax3.ticklabel_format(style='sci', scilimits=(0, 0), useOffset=False, useMathText=True)
    fig3.savefig(r'D:\Profile\oah\my_files\phd\presentations\210309_excitons\davies_fit.png', dpi=300)

    #
    # from HelpFunctions import find_closest
    #
    # energy_eval = np.linspace(0.1, 10, 5001)
    # N = 10
    # koch = alpha_exciton_koch(**test_args, excitons=N) + alpha_continuum_koch(**test_args)
    # davies = alpha_exciton(**test_args, excitons=N) + alpha_continuum(**test_args)
    # fig2, ax2 = plt.subplots()
    # scale_pos = find_closest(energy_eval, test_args['bandgap_energy'] - test_args['exciton_energy'])
    # ax2.plot(energy_eval, koch / koch[scale_pos])
    # ax2.plot(energy_eval, davies / davies[scale_pos])
    #
    # # 'energy': np.array(test_x),
    # alpha_sum_args = {
    #     'exciton_energy': 0.0205,
    #     'bandgap_energy': 1.656,
    #     'exciton_gaussian_hwhm': 0.012,
    #     'exciton_lorentzian_hwhm': 0.00,
    #     'continuum_gaussian_hwhm': 0.02,
    #     'continuum_lorentzian_hwhm': 0.00,
    #     'scale_factor': 0.264,
    #     'excitons': 10
    # }
    # fig3, ax3 = plt.subplots()
    # ax3.plot(wavelengths, transmission(wavelengths, **alpha_sum_args, n_inf=2.3, thickness=0.5))

    # ax.plot(test_x, alpha_sum(**alpha_sum_args), 'x-')

    # alpha_fit_function = get_alpha_fit_function()
    # p0 = [0.0205, 1.656, 0.012, 0.001, 0.02, 0.001, 26.4]
    # p0 = [0.01, 1.65, 0.01, 0.01, 0.02, 0.02, 0.3]
    # # popt, pcov = curve_fit(alpha_fit_function, np.array(test_x), np.array(test_y), p0=p0, bounds=(0, np.inf))
    # p_opt_2 = alpha_minimization_fit(np.array(test_x), np.array(test_y), np.array(p0), ax=ax)
    # fig2, ax2 = plt.subplots()
    # popt = p_opt_2.x
    # ax2.plot(test_x, test_y, 'o', alpha=0.5, label='Experiment')
    # ax2.plot(test_x, alpha_exciton_scaled(energy=np.array(test_x), exciton_energy=popt[0], bandgap_energy=popt[1],
    #                                       gaussian_hwhm=popt[2], lorentzian_hwhm=popt[3], scale_factor=100 * popt[6],
    #                                       excitons=10, continuum_gaussian_hwhm=popt[4],
    #                                       continuum_lorentzian_hwhm=popt[5]),
    #          label=r'$\alpha_\mathrm{X}$')
    # ax2.plot(np.linspace(0.1, 10, 5001), alpha_continuum(energy=np.linspace(0.1, 10, 5001), exciton_energy=popt[0],
    #                                                      bandgap_energy=popt[1], gaussian_hwhm=popt[4],
    #                                                      lorentzian_hwhm=popt[5], scale_factor=100 * popt[6]),
    #          label=r'$\alpha_\mathrm{C}$')
    # ax2.plot(test_x, alpha_sum(np.array(test_x), *popt, excitons=10), label=r'Sum')
    # # ax2.plot(test_x, alpha_sum(np.array(test_x), *p_opt_2.x, excitons=10), label=r'Sum')
    # print(p_opt_2)
    # ax2.set_xlim(test_x[-1], test_x[0])
    # ax2.legend()

    # fig2, ax2 = plt.subplots()
    # ax2.plot(test_args['energy'], log_normal_normal(test_args['energy'], 0.01, 0.1, 1))
    # import timeit
    # x_values = np.linspace(-10, 10, 1000)
    # print(timeit.timeit("voigt(x_values,0.5,0.5)", globals=globals(), number=1000))
    # print(timeit.timeit("0.5*gaussian(x_values,0.5)+0.5*lorentzian(x_values,0.5)", globals=globals(), number=1000))
