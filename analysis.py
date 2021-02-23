import sys
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import binned_statistic
from HelpFunctions import extend_mesh, find_closest
from PatternLoader import load_pvd_log
from matplotlib import use
from matplotlib.colors import LogNorm

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
    correction_uv_vis = np.mean(binned_uv_vis_overlap[0]/binned_nir_overlap[0], axis=1)
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
    difference = np.diff(trans_binned, axis=1)
    wave_diff = bin_centers[1:] - bin_width / 2

    # Plot data
    fig, axs = plt.subplots(2, 2, sharex=True)

    # trans_max = np.max(trans_binned, axis=1)
    # trans_norm = trans_max[0] * trans_binned.T / trans_max
    axs[0, 1].pcolormesh(extend_mesh(t_trans), bin_edges, trans_binned.T, cmap='magma')
    axs[0, 1].set_ylabel('Wavelength / nm')

    axs[0, 0].plot(t_temperature, pvd_log['LinkamStage PV'])
    axs[0, 0].set_ylabel(r'Hotplate temperature / °C')

    axs[1, 0].pcolormesh(extend_mesh(t_patterns), extend_mesh(q), patterns.values.T,
                         norm=LogNorm(vmin=0.2), cmap='magma')
    axs[1, 0].set_ylabel(r'Scattering vector $q$ / nm$^{-1}$')
    axs[1, 0].set_xlim(np.min(t_patterns), np.max(t_patterns))

    axs[1, 0].set_xlabel('Process time / s')
    axs[1, 1].set_xlabel('Process time / s')
