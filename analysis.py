import sys
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import hilbert
from scipy.optimize import curve_fit

from scipy.stats import binned_statistic
from HelpFunctions import extend_mesh, find_closest, gauss, peak_fit, q_to_two_theta, lattice_from_hkl, track_peaks, \
    wavelength_to_energy
from PatternLoader import load_pvd_log
from matplotlib import use
from matplotlib.colors import LogNorm

from elliot import abs_coef_fit_function, transmittance_fit_function, refractive_index, gaussian, tanguy, \
    tmm_transmission, alpha_exciton, alpha_continuum, alpha_exciton_lorentzian, alpha_continuum_lorentzian, \
    alpha_continuum_koch, alpha_exciton_koch, tmm_reflectance, absorption_coefficient, get_transmittance_fit_function, \
    tanguy_tmm_transmittance

if __name__ == "__main__":
    use('Qt5Agg')
    folder_path = sys.argv[1]
    name = os.path.split(folder_path)[1]

    # Import integrated patterns
    patterns = pd.read_csv(os.path.join(folder_path, name + '_reintegrated.csv'), index_col=0)
    t0 = np.datetime64(patterns.index.values[0])
    t_patterns = (patterns.index.values.astype(np.datetime64) - t0) / np.timedelta64(1, 's')
    q = patterns.columns.values.astype(float)

    # Import pvd log
    pvd_log = load_pvd_log(os.path.join(folder_path, name + '.csv'))
    t_temperature = (pvd_log.index.values.astype(np.datetime64) - t0) / np.timedelta64(1, 's')

    # Import transmission
    trans_uv_vis = pd.read_csv(os.path.join(folder_path, name + '_transmission.csv'), index_col=0)
    t_trans = (trans_uv_vis.index.values.astype(np.datetime64) - t0) / np.timedelta64(1, 's')
    trans_nir = pd.read_csv(os.path.join(folder_path, name + '_nir_transmission.csv'), index_col=0)
    t_trans_nir = (trans_nir.index.values.astype(np.datetime64) - t0) / np.timedelta64(1, 's')
    trans_nir_interp = trans_nir.values[find_closest(t_trans_nir, t_trans), :]

    wave_uv_vis = trans_uv_vis.columns.values.astype(float)
    wave_nir = trans_nir.columns.values.astype(float)

    nir_ref_spec = np.mean(trans_nir_interp[:find_closest(t_trans, 150), :], axis=0)
    nir_diff_spec = trans_nir_interp / nir_ref_spec
    nir_ref_spec_post = np.mean(trans_nir_interp[find_closest(t_trans, 5699):, :], axis=0)
    nir_diff_spec_post = trans_nir_interp / nir_ref_spec_post
    corr_pre = np.mean(
        nir_diff_spec[:find_closest(t_trans, 1661), find_closest(wave_nir, 1650):find_closest(wave_nir, 1700)], axis=1)
    corr_post = np.mean(
        nir_diff_spec_post[find_closest(t_trans, 1661):, find_closest(wave_nir, 1650):find_closest(wave_nir, 1700)],
        axis=1)
    correction = np.concatenate([np.ones(find_closest(t_trans, 1661)) * corr_pre,
                                 np.ones(len(t_trans) - find_closest(t_trans, 1661)) * corr_post])
    trans_nir_corrected = (trans_nir_interp.T / correction).T
    bins = np.arange(969.5, 1080.5, 2)
    binned_uv_vis_overlap = binned_statistic(wave_uv_vis, trans_uv_vis.values, statistic='mean', bins=bins)
    binned_nir_overlap = binned_statistic(wave_nir, trans_nir_corrected, statistic='mean', bins=bins)
    correction_uv_vis = np.mean(binned_uv_vis_overlap[0] / binned_nir_overlap[0], axis=1)
    trans_uv_vis_corrected = (trans_uv_vis.values.T / correction_uv_vis).T

    roi_uv_vis = np.where((465 <= wave_uv_vis) * (wave_uv_vis < 1100))
    wave = np.concatenate([wave_uv_vis[roi_uv_vis], wave_nir])
    trans = np.concatenate([trans_uv_vis_corrected[:, roi_uv_vis[0]], trans_nir_corrected], axis=1)

    trans_binned, bin_edges, binnumber = binned_statistic(wave, trans, statistic='mean', bins=623)
    # roi_wave = np.where((465 <= wave_uv_vis) * (wave_uv_vis < 1050))
    # trans_binned, bin_edges, binnumber = binned_statistic(wave[roi_wave], trans_uv_vis.values[:, roi_wave[0]],
    #                                                       statistic='mean', bins=200)
    bin_width = (bin_edges[1] - bin_edges[0])
    bin_centers = bin_edges[1:] - bin_width / 2
    difference = np.diff(trans_binned, axis=1) / np.diff(wavelength_to_energy(bin_centers)) * -1
    wave_diff = bin_centers[1:] - bin_width / 2
    energy_diff = wavelength_to_energy(wave_diff)

    diff_max = []
    p0 = [0.1, 640, 10]
    p0 = [10, 1.93, 0.03]
    for idx in range(difference.shape[0]):
        p_opt, p_cov = curve_fit(gauss, energy_diff, difference[idx], p0=p0)
        p0 = p_opt
        diff_max.append(p_opt[1])
    diff_max = np.array(diff_max)

    # Plot data
    fig, axs = plt.subplots(2, 2, sharex='all', figsize=[12, 5])

    # trans_max = np.max(trans_binned, axis=1)
    # trans_norm = trans_max[0] * trans_binned.T / trans_max
    axs[0, 1].pcolormesh(extend_mesh(t_trans), bin_edges, trans_binned.T, cmap='magma')
    axs[0, 1].set_ylabel('Wavelength / nm')
    axs[0, 1].set_title('White Light Transmission')

    axs[0, 0].plot(t_temperature, pvd_log['LinkamStage PV'])
    axs[0, 0].set_ylabel(r'Hotplate temperature / °C')
    axs[0, 0].set_title('Heating profile')

    axs[1, 0].pcolormesh(extend_mesh(t_patterns), extend_mesh(q), patterns.values.T,
                         norm=LogNorm(vmin=0.2), cmap='magma')
    axs[1, 0].set_ylabel(r'Scattering vector $q$ / nm$^{-1}$')
    axs[1, 0].set_xlim(np.min(t_patterns), np.max(t_patterns))
    axs[1, 0].set_title('Azimuthally Integrated GIWAXS')

    axs[1, 1].plot(t_trans, diff_max)
    axs[1, 1].set_ylim([636, 650])
    axs[1, 1].set_ylabel(r'$\mathrm{arg\,max}_\lambda\left[\frac{\mathrm{d}T}{\mathrm{d}\lambda}\right]$')
    axs[1, 1].set_title('Absorption Onset')
    axs[1, 0].set_xlabel('Process time / s')
    axs[1, 1].set_xlabel('Process time / s')

    # fig.savefig(r'D:\Profile\oah\my_files\phd\presentations\210309_excitons\metaljet_results.png', dpi=300)

    fig, axs = plt.subplots(1, 3, sharex='all', figsize=[12, 5])
    axs[0].plot(t_temperature, pvd_log['LinkamStage PV'])
    axs[0].set_ylabel(r'Hotplate temperature / °C')
    axs[0].set_title('Heating profile')
    axs[1].pcolormesh(extend_mesh(t_patterns), extend_mesh(q), patterns.values.T,
                      norm=LogNorm(vmin=0.2), cmap='magma')
    axs[1].set_ylabel(r'Scattering vector $q$ / nm$^{-1}$')
    axs[1].set_xlim(np.min(t_patterns), np.max(t_patterns))
    axs[1].set_title('Azimuthally Integrated GIWAXS')
    axs[2].pcolormesh(extend_mesh(t_trans), bin_edges, trans_binned.T, cmap='magma')
    axs[2].set_ylabel('Wavelength / nm')
    axs[2].set_title('White Light Transmission')
    axs[1].set_xlabel('Process time / s')
    fig.tight_layout()

    fig.savefig(r'D:\Profile\oah\my_files\phd\presentations\210322_hypercells_meeting\metaljet_results.png', dpi=300)

    # Elliot analysis
    fig2, axs2 = plt.subplots(1, 2, figsize=[12, 5])
    # axs2[0].plot(bin_centers, trans_binned[find_closest(t_trans, 2750)], 'o', alpha=0.5,
    #              label=r'T = %d $^\circ$C' % pvd_log['LinkamStage PV'][find_closest(t_temperature, 2750)])
    # axs2[0].plot(bin_centers, trans_binned[find_closest(t_trans, 1750)], 'o', alpha=0.5,
    #              label=r'T = %d $^\circ$C' % pvd_log['LinkamStage PV'][find_closest(t_temperature, 1750)])
    axs2[0].plot(bin_centers, trans_binned[find_closest(t_trans, 470)], 'o', alpha=0.5,
                 label=r'T = %d $^\circ$C' % pvd_log['LinkamStage PV'][find_closest(t_temperature, 470)])
    axs2[0].set_ylabel('Transmittance')
    axs2[0].set_xlabel(r'Photon wavelength / $\lambda$')
    axs2[0].legend()

    # axs2[1].plot(1240 / bin_centers, np.log(0.86 / trans_binned[find_closest(t_trans, 2750)])/5.65e-5, 'o', alpha=0.5,
    #              label=r'T = %d $^\circ$C' % pvd_log['LinkamStage PV'][find_closest(t_temperature, 2750)])
    # axs2[1].plot(1240 / bin_centers, np.log(0.86 / trans_binned[find_closest(t_trans, 1750)])/5.65e-5, 'o', alpha=0.5,
    #              label=r'T = %d $^\circ$C' % pvd_log['LinkamStage PV'][find_closest(t_temperature, 1750)])
    axs2[1].plot(1240 / bin_centers, np.log(0.86 / trans_binned[find_closest(t_trans, 470)]) / 5.65e-5, 'o', alpha=0.5,
                 label=r'T = %d $^\circ$C' % pvd_log['LinkamStage PV'][find_closest(t_temperature, 470)])
    axs2[1].set_ylabel(r'$\ln{(0.86/\mathrm{Transmittance})}/565\cdot10^{-7}$ / cm$^{-1}$')
    axs2[1].ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useOffset=False, useMathText=True)
    axs2[1].set_xlabel(r'Photon energy / eV')
    axs2[1].legend()
    axs2[1].set_xlim([1.85, 2.6])
    # fig2.savefig(r'D:\Profile\oah\my_files\phd\presentations\210309_excitons\metaljet_results_2.png', dpi=300)

    roi = np.where(((1240 / bin_centers) > 1.9) * ((1240 / bin_centers) < 2.15))
    p0 = np.array([2.43735966e-02, 1.96198987e+00, 5.92719437e-01, 2.18947260e-02, -5.00664357e+00, 2 * 2.52213562])
    p0 = np.array([2.45e-02, 1.96198987e+00, 5.92719437e-01, 2.18947260e-02, -5.00664357e+00, 3.3])
    p0 = np.array([2.5e-02, 1.98e+00, 5.92719437e-01, 2.18947260e-02, -5.00664357e+00, 3.3])
    # p_opt, p_cov = curve_fit(abs_coef_fit_function, (1240 / bin_centers)[roi],
    #                          (np.log(0.8 / trans_binned[find_closest(t_trans, 2750)])/5.6e-5)[roi], p0=p0)
    axs2[1].plot(1240 / bin_centers, abs_coef_fit_function(1240 / bin_centers, *p0), label='Tanguy guess')
    T, R = transmittance_fit_function(1240 / bin_centers, *p0, 1.3, 0.565)
    axs2[0].plot(bin_centers, T, color='C2', label='T from TMM')
    axs2[0].plot(bin_centers, R, '--', color='C2', label='R from TMM')
    axs2[0].legend()
    axs2[1].legend()
    # fig2.savefig(r'D:\Profile\oah\my_files\phd\presentations\210309_excitons\metaljet_results_fit_guess.png', dpi=300)

    fig, ax = plt.subplots(figsize=[6, 5])
    ax.set_ylabel(r'Transmittance, $T$')
    ax.set_xlabel(r'Photon wavelength, $\lambda$ / nm')
    ax.plot(bin_centers, trans_binned[find_closest(t_trans, 0)],
            'o', alpha=0.5, label=r'T = %d $^\circ$C' % pvd_log['LinkamStage PV'][find_closest(t_temperature, 0)])
    ax.legend(loc='lower right')
    fig.savefig(r'D:\Profile\oah\my_files\phd\presentations\210322_hypercells_meeting\transmittance.png', dpi=300)

    fig, ax = plt.subplots(figsize=[6, 5])
    ax.set_ylabel(r'$\ln{(0.86/\mathrm{Transmittance})}/565\cdot10^{-7}$ / cm$^{-1}$')
    ax.set_xlabel(r'Photon energy, $\hbar\omega$ / eV')
    ax.set_xlim([1.8, 2.05])
    ax.set_ylim([-1e4, 1.1e5])
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useOffset=False, useMathText=True)
    ax.plot(1240 / bin_centers, np.power(np.log(0.86 / trans_binned[find_closest(t_trans, 470)]), 1) / 5.65e-5,
            'o', alpha=0.5, label=r'T = %d $^\circ$C' % pvd_log['LinkamStage PV'][find_closest(t_temperature, 470)])
    ax.plot(1240 / bin_centers, np.power(np.log(0.86 / trans_binned[find_closest(t_trans, 0)]), 1) / 5.65e-5,
            'o', alpha=0.5, label=r'T = %d $^\circ$C' % pvd_log['LinkamStage PV'][find_closest(t_temperature, 0)])
    ax.legend(loc='upper left')
    # fig.savefig(r'D:\Profile\oah\my_files\phd\presentations\210322_hypercells_meeting\tauc_1.png', dpi=300)
    ax.plot(1240 / bin_centers, np.power(np.log(0.86 / trans_binned[find_closest(t_trans, 1750)]), 1) / 5.65e-5,
            'o', alpha=0.5, label=r'T = %d $^\circ$C' % pvd_log['LinkamStage PV'][find_closest(t_temperature, 1750)])
    ax.legend(loc='upper left')
    # fig.savefig(r'D:\Profile\oah\my_files\phd\presentations\210322_hypercells_meeting\tauc_2.png', dpi=300)
    ax.set_xlim([1.8, 2.6])
    ax.set_ylim([-1e3, 1e5])
    fig.savefig(r'D:\Profile\oah\my_files\phd\presentations\210322_hypercells_meeting\pseudo_abs.png', dpi=300)
    # fig.savefig(r'D:\Profile\oah\my_files\phd\presentations\210322_hypercells_meeting\tauc_3.png', dpi=300)

    ax.legend(loc='upper left')
    fig.savefig(r'D:\Profile\oah\my_files\phd\presentations\210322_hypercells_meeting\tauc_4.png', dpi=300)

    # Lattice analysis
    rois = [np.where((16.8 < q) * (q < 17.6)), np.where((19 < q) * (q < 22)), np.where((22 < q) * (q < 24))]
    # rois = [np.where((19 < q) * (q < 22)), np.where((22 < q) * (q < 24)), np.where((24 < q) * (q < 26))]
    two_theta = q_to_two_theta(q, "Ga")
    popts = []
    for roi in rois:
        popts.append(peak_fit(two_theta[roi], patterns.values[0, roi], fwhm=0.18, plot_result=False))

    lattice = lattice_from_hkl(hkl=np.array([[1, 2, 1], [0, 0, 4], [1, 3, 0]]), q=np.array([popt[1] for popt in popts]))
    # lattice = lattice_from_hkl(hkl=np.array([[0, 0, 4], [1, 3, 0], [2, 0, 4]]), q=np.array([popt[1] for popt in popts]))
    print(lattice.reshape(-1, ) * np.array([1 / np.sqrt(2), 1 / np.sqrt(2), 1 / 2]) * 10)
    cutoff = find_closest(t_patterns, 1360)
    peaks = [[1, 2, 1], [0, 0, 4], [1, 3, 0], [1, 3, 2], [2, 2, 4], [1, 4, 1], [1, 3, 4]]
    result = track_peaks(two_theta=two_theta, intensities=patterns.values[:cutoff, :],
                         peaks=peaks, lattice_guess=lattice)
    fig3, ax3 = plt.subplots()
    ax3.pcolormesh(extend_mesh(t_patterns[:cutoff]), extend_mesh(q), patterns.values[:cutoff, :].T,
                   norm=LogNorm(vmin=0.2), cmap='magma')
    for peak in result:
        ax3.plot(t_patterns[:cutoff], [popt[1] for popt in peak], color='w')

    lattice_params = []
    peak_index = [1, 2, 3]
    peaks[2] = [1, 1, 4]
    peaks[3] = [0, 2, 4]
    for i in range(len(result[0])):
        lattice_params.append(lattice_from_hkl(hkl=np.array([peaks[peak_index[0]],
                                                             peaks[peak_index[1]],
                                                             peaks[peak_index[2]]]),
                                               q=np.array([result[peak_index[0]][i][1],
                                                           result[peak_index[1]][i][1],
                                                           result[peak_index[2]][i][1]])))
    lattice_params = np.array(lattice_params)
    scaled_params = lattice_params.reshape(-1, 3) * np.array([1 / np.sqrt(2), 1 / np.sqrt(2), 1 / 2]) * 1e3

    fig5, ax5 = plt.subplots()
    ax5.pcolormesh(extend_mesh(t_patterns[cutoff:]), extend_mesh(q), patterns.values[cutoff:, :].T,
                   norm=LogNorm(vmin=0.2), cmap='magma')

    rois = [np.where((20.5 < q) * (q < 21.5)), np.where((23 < q) * (q < 24)), np.where((25.5 < q) * (q < 26.5))]
    # rois = [np.where((19 < q) * (q < 22)), np.where((22 < q) * (q < 24)), np.where((24 < q) * (q < 26))]
    two_theta = q_to_two_theta(q, "Ga")
    popts = []
    for roi in rois:
        popts.append(peak_fit(two_theta[roi], patterns.values[cutoff + 3, roi], fwhm=0.1, plot_result=False))

    lattice = lattice_from_hkl(hkl=np.array([[2, 2, 0], [3, 1, 0], [3, 1, 2]]), q=np.array([popt[1] for popt in popts]))
    peaks2 = np.array([[2, 2, 0], [3, 1, 0], [3, 1, 2]])
    result2 = track_peaks(two_theta=two_theta, intensities=patterns.values[cutoff:, :],
                          peaks=peaks2, lattice_guess=lattice)
    for peak in result2:
        ax5.plot(t_patterns[cutoff:], [popt[1] for popt in peak], color='w')

    lattice_params2 = []
    peak_index = [0, 1, 2]
    for i in range(len(result2[0])):
        lattice_params2.append(lattice_from_hkl(hkl=np.array([peaks2[peak_index[0]],
                                                              peaks2[peak_index[1]],
                                                              peaks2[peak_index[2]]]),
                                                q=np.array([result2[peak_index[0]][i][1],
                                                            result2[peak_index[1]][i][1],
                                                            result2[peak_index[2]][i][1]])))
    lattice_params2 = np.array(lattice_params2)
    scaled_params2 = lattice_params2.reshape(-1, 3) * np.array([1 / np.sqrt(2), 1 / np.sqrt(2), 1 / 2]) * 1e3

    fig4, [ax4, ax4t] = plt.subplots(2, 1, sharex='all', figsize=[12, 6])
    ax4.plot(t_patterns[:cutoff], scaled_params[:, 0], 'o', alpha=0.5, color='C0', label='a'),
    ax4.plot(t_patterns[:cutoff], scaled_params[:, 1], 'o', alpha=0.5, color='C1', label='b'),
    ax4.plot(t_patterns[:cutoff], scaled_params[:, 2], 'o', alpha=0.5, color='C2', label='c'),
    ax4.plot(t_patterns[:cutoff], np.power(np.product(scaled_params, axis=1), 1 / 3), 'o', alpha=0.5,
             label='Geometric Mean', color='C3')

    ax4.plot(t_patterns[cutoff:], scaled_params2[:, 0], 'o', alpha=0.5, color='C0', label='a')
    ax4.plot(t_patterns[cutoff:], scaled_params2[:, 1], 'o', alpha=0.5, color='C1', label='b')
    ax4.plot(t_patterns[cutoff:], scaled_params2[:, 2], 'o', alpha=0.5, color='C2', label='c')
    ax4.plot(t_patterns[cutoff:], np.power(np.product(scaled_params2, axis=1), 1 / 3), 'o', alpha=0.5,
             label='Geo. Avrg.', color='C3')
    ax4.set_ylabel('Lattice parameter / pm')

    ax4t.plot(t_trans, diff_max, 'o', alpha=0.5, color='C2')
    ax4t.set_ylabel(r'$\hbar\cdot\mathrm{arg\,max}_{\omega}\left[\frac{\mathrm{d}T}{\mathrm{d}\omega}\right]$ / eV',
                    color='C2')
    ax4t.tick_params(axis='y', labelcolor='C2')
    ax4t.set_ylim([1.9, 1.95])

    ax4t2 = ax4t.twinx()
    ax4t2.plot(t_temperature, pvd_log['LinkamStage PV'] + 273.15, color='C3')
    ax4t2.set_ylabel(r'Hotplate temperature / K', color='C3')
    ax4t2.set_ylim([0, 600])
    ax4t2.tick_params(axis='y', labelcolor='C3')
    ax4t2.set_xlim(t_patterns[0], t_patterns[-1])
    ax4t.set_xlabel('Process time / s')
    fig4.tight_layout()
    fig4.subplots_adjust(hspace=0)
    handles, labels = ax4.get_legend_handles_labels()
    ax4.legend(handles[:4], labels[:4], loc="lower left", bbox_to_anchor=(0.62, 0.1))
    fig4.show()

    fig4.savefig(r'D:\Profile\oah\my_files\phd\presentations\210322_hypercells_meeting\metaljet_results_2.png', dpi=300)

    # More Elliot
    h_bar = 6.582119569e-16  # eV*s
    h = 4.135667696e-15  # eV*s
    c = 299792458.0  # m/s
    k = 8.617333262145e-5

    fig, ax = plt.subplots()
    fig2, [ax2right, ax2] = plt.subplots(1, 2, figsize=[12, 5])
    points = 20001
    E = np.linspace(0.1, 10, points)
    energy_relative = E - E[int(points / 2)]
    dE = E[1] - E[0]

    epsilon = 1.345 ** 2 + np.convolve(gaussian(energy_relative, 2.7e-2),
                                       tanguy(E, 29e-3, 2005e-3, 160 * k / 4, 3.25), mode='same') * dE
    # epsilon = 1.345 ** 2 + tanguy(E, 29e-3, 2005e-3, 160 * k / 10, 3.25)
    n = refractive_index(epsilon)
    n = n + 1j * (epsilon.imag / (2 * n))
    wave = 1e9 * h * c / E
    T = tmm_transmission(np.array([np.ones(n.shape), n, 1.45 * np.ones(n.shape)]), 1e-3 * wave, 0.553)
    R = tmm_reflectance(np.array([np.ones(n.shape), n, 1.45 * np.ones(n.shape)]), 1e-3 * wave, 0.553)

    args = {"exciton_energy": 29e-3,
            "bandgap_energy": 2005e-3,
            "gaussian_hwhm": 2.7e-2,
            "lorentzian_hwhm": 160 * k / 4,
            "scale_factor": 0.8 * 1e6 * 3.25 / (2 * np.pi)}
    alpha = alpha_exciton(E, **args, excitons=10) + alpha_continuum(E, **args)
    k_theo = 1e2 * alpha * h_bar * c / E / 2
    n_theo = 1 + 1.345 + hilbert(k_theo)

    args = {"exciton_energy": 29e-3,
            "bandgap_energy": 2005e-3,
            "hwhm": 160 * k / 4,
            "scale_factor": 0.8 * 1e6 * 3.25 / (2 * np.pi)}
    alpha = alpha_exciton_lorentzian(E, **args, excitons=10) + alpha_continuum_lorentzian(E, **args)
    k_theo_lor = 1e2 * alpha * h_bar * c / E / 2
    n_theo_lor = 1 + 1.345 + hilbert(k_theo_lor)

    args = {"exciton_energy": 18e-3,
            "bandgap_energy": 2003e-3,
            "gaussian_hwhm": 3.8e-2,
            "lorentzian_hwhm": 160 * k / 4,
            "scale_factor": 1.8 * 1e2 * 3.25 / (2 * np.pi)}
    alpha = alpha_exciton_koch(E, **args, excitons=10) + alpha_continuum_koch(E, **args)
    k_theo_koch = 1e2 * alpha * h_bar * c / E / 2
    n_theo_koch = 0.88 + 1.345 + hilbert(k_theo_koch)

    ax.clear()
    ax.plot(E, n.real)
    ax.plot(E, n.imag)
    # ax.plot(E, k_theo)
    # ax.plot(E, n_theo)
    # ax.plot(E, k_theo_lor)
    # ax.plot(E, n_theo_lor)
    ax.plot(E, k_theo_koch)
    ax.plot(E, n_theo_koch)
    # ax.set_xlim(1.85, 2.1)

    n_koch = n_theo_koch + 1j * k_theo_koch
    T_koch = tmm_transmission(np.array([np.ones(n.shape), n_koch, 1.45 * np.ones(n.shape)]), 1e-3 * wave, 0.553)

    ax2.clear()
    ax2right.clear()
    ax2.plot(bin_centers, trans_binned[find_closest(t_trans, 470)], 'o', alpha=0.5,
             label=r'Measured Transmittance at T = %d $^\circ$C' % pvd_log['LinkamStage PV'][
                 find_closest(t_temperature, 470)])
    ax2.plot(wave, T * np.exp(-1 * np.power(2 * np.pi * 33 / wave, 2)), color='C0', label='Calculated Transmittance')
    ax2.plot(wave, R * np.exp(-1 * np.power(2 * np.pi * 33 / wave, 2)), '--', color='C0',
             label='Calculated Reflectance')
    ax2.set_xlim([467, 1712])
    ax2.legend()
    ax2.set_xlabel(r'Photon wavelength, $\lambda$ / nm')
    ax2.set_ylabel(r'Transmittance and Reflectance')
    ax2right.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useOffset=False, useMathText=True)
    ax2right.plot(E, absorption_coefficient(E, epsilon), label=r'Modelled $\alpha$')
    ax2right.plot([2005e-3, 2005e-3], [0, 4.5e4], '--', label=r'$E_\mathrm{g}=%.3f$ eV' % 2005e-3)
    ax2right.plot([2005e-3 - 29e-3, 2005e-3 - 29e-3], [0, 4.5e4], '--', label=r'$E_0=%d$ meV' % 29)
    ax2right.set_ylabel(r'Absorption Coefficient, $\alpha$ / cm$^{-1}$')
    ax2right.set_xlabel(r'Photon energy, $\hbar\omega$ / eV')
    ax2right.set_xlim([1.85, 2.2])
    ax2right.legend(loc='center right')
    # fig2.savefig(r'D:\Profile\oah\my_files\phd\presentations\210322_hypercells_meeting\model.png', dpi=300)

    # ax2.plot(wave, T_koch * np.exp(-1 * np.power(2 * np.pi * 33 / wave, 2)))
    # ax2.set_xlim([467, 1712])

    ax2.plot(bin_centers, trans_binned[find_closest(t_trans, 0)], 'o', alpha=0.5,
             label=r'T = %d $^\circ$C' % pvd_log['LinkamStage PV'][find_closest(t_temperature, 0)], color='C1')
    epsilon2 = 1.345 ** 2 + np.convolve(gaussian(energy_relative, 3.5e-2),
                                        tanguy(E, 21e-3, 2005e-3, 293 * k / 4, 3.95), mode='same') * dE
    n2 = refractive_index(epsilon2)
    n2 = n2 + 1j * (epsilon2.imag / (2 * n2))
    wave = 1e9 * h * c / E
    T2 = tmm_transmission(np.array([np.ones(n2.shape), n2, 1.45 * np.ones(n2.shape)]), 1e-3 * wave, 0.550)
    ax2.plot(wave, T2 * np.exp(-1 * np.power(2 * np.pi * 20 / wave, 2)), color='C1')
    points = int(5 * (4 - 0.1) / (2 * (160 * k / 4)) // 2 * 2 + 1)
    E = np.linspace(0.1, 4, points)
    energy_relative = E - E[int(points / 2)]
    dE = E[1] - E[0]
    ax2.plot(1e9 * h * c / E, tanguy_tmm_transmittance(E, E, energy_relative, dE, 21e-3, 2005e-3,
                                                       3.5e-2, 293 * k / 4, 3.95, 1.345, 0.550, 20e-3),
             color='C2')
    ax2.set_xlim([467, 1712])

    fit_function = get_transmittance_fit_function(1, 1, (160 * k / 4), 1, 1)
                                                  # lorentzian_hwhm=293 * k / 4,
                                                  # n_bgr=1.345,
                                                  # thickness=0.550,
                                                  # roughness=20e-3)
    roi = np.where((bin_centers < 825) * (bin_centers > 575))
    p0 = [21e-3, 2005e-3, 3.5e-2, 293 * k / 4, 3.95, 1.345, 0.550, 20e-3]
    p_opts = []
    for i in range(find_closest(t_trans, 0), find_closest(t_trans, 470)):
        p_opt, p_cov = curve_fit(fit_function, 1e9 * h * c / bin_centers[roi], trans_binned[i][roi], p0=p0)
        p_opts.append(p_opt)
        p0 = p_opt
        np.savetxt('p_opts.txt', np.array(p_opts))
    ax2.plot(bin_centers, fit_function(1e9 * h * c / bin_centers, *p_opts[-1]))
    p_opts = np.array(p_opts)