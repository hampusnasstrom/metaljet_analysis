import pyFAI
import fabio
import numpy as np
import matplotlib.pyplot as plt
from os import path

from HelpFunctions import baseline_als

n_images = 10
chis = np.arange(13, 3, -1)
measurements = ['ee-nyfs_20200625_Igal_B7_pos_2_chiscan', 'ee-nyfs_20200625_Igal_B8_pos_2_chiscan']
titles = ['Sample B7', 'Sample B8']
short_names = ['B7', 'B8']


fig, axs = plt.subplots(2, 1, sharex=True)
ai = pyFAI.load(r'\\ul-nas\metaljet_data\ee-nyfs_20200626_LaB6_chiscan\ee-nyfs_20200626_LaB6_chi8.poni')
integrated_data = []
for measurement_idx, measurement in enumerate(measurements):
    ax = axs[measurement_idx]
    for idx, chi in enumerate(chis):
        with fabio.open(r'\\ul-nas\metaljet_data\ee-nyfs_20200626_LaB6_chiscan\mask_chi%d.edf' % chi) as mask_file:
            mask = mask_file.data
        with fabio.open(r'\\ul-nas\metaljet_data\ee-nyfs_20200625_Igal_ITO\image_%05d.tif' % (idx+1)) as flat_file:
            flat = flat_file.data
        with fabio.open(path.join(r'\\ul-nas\metaljet_data', measurement, 'image_%05d.tif' % (idx+1))) as data_file:
            data = data_file.data
        res = ai.integrate1d(data=data, npt=3000, unit="q_nm^-1", flat=flat, mask=mask)
        tmp_intensity = res[1][np.logical_not(np.isinf(res[1]))]
        tmp_q = res[0][np.logical_not(np.isinf(res[1]))]
        intensity = tmp_intensity[np.logical_not(np.isnan(tmp_intensity))]
        q = tmp_q[np.logical_not(np.isnan(tmp_intensity))]
        bgr = baseline_als(intensity, 1e6, 0.01)
        intensity = intensity-bgr
        integrated_data.append((q, intensity))
        ax.plot(q[20:-200], intensity[20:-200], label=r'$\chi=%d$' % chi)
        np.savetxt(r'D:\colabs\levine\%s_chi%d_integrated.csv' % (short_names[measurement_idx], chi),
                   np.array([q, intensity]).T,
                   comments='# q units: nm^-1\n# incidence angle: 4 deg\n# sample: %s\n\n' % short_names[measurement_idx],
                   header='q,intensity',
                   delimiter=',')
    ax.set_ylabel('Azimuthally integrated intensity')
    ax.set_title(titles[measurement_idx])

axs[1].set_xlabel(r'Scattering vector, $q$ / nm$^{-1}$')
plt.legend()
