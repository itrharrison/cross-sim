import numpy as np
from astropy import units
import pyccl as ccl
import sacc

def setup_gamma_tracers(cosmo, dndz, A_IA):
    
    tracer_list = []

    if A_IA is not None:
        A_IA = (dndz[:,0], A_IA * np.ones(len(dndz[:,0])))
    
    for ibin in np.arange(1,dndz.shape[-1]):
        
        tracer_list.append(ccl.WeakLensingTracer(cosmo,
                                                dndz=(dndz[:,0], dndz[:,ibin]),
                                                ia_bias=A_IA
                                                )
                          )
    
    return tracer_list 

def calculate_cl(cosmo, galaxy_tracers, cmb_tracer, dndz, ngal, sigma_e, binning):
   
    n_maps = len(galaxy_tracers) + 1

    w_bins = binning['Well'].weight.T
    
    cls = np.zeros([n_maps, n_maps, binning['n_ell']])

    cls[0, 0, :] = np.dot(w_bins, ccl.angular_cl(cosmo, cmb_tracer, cmb_tracer, binning['ells_theory'] + Nell_so_k_binned))

    for ibin in np.arange(1,dndz.shape[-1]):

        # Nell_gals_bin = np.ones(binning['n_ell']) / (ngal[ibin-1].value * (60 * 180 / np.pi)**2)
        Nell_gamma_bin = np.ones(len(binning['ells_theory'])) * sigma_e[ibin - 1]**2. / (ngal[ ibin - 1] * (60 * 180 / np.pi)**2)

        cls[0, ibin, :] = np.dot(w_bins, ccl.angular_cl(cosmo, galaxy_tracers[ibin-1], cmb_tracer, binning['ells_theory']))
        cls[ibin, 0, :] = cls[0, ibin, :]

        for jbin in np.arange(1,dndz.shape[-1]):
            if ibin==jbin:
                cls[ibin, ibin, :] = np.dot(w_bins, ccl.angular_cl(cosmo, galaxy_tracers[ibin-1], galaxy_tracers[ibin-1], binning['ells_theory']) + Nell_gamma_bin)
            else:
                cls[ibin, jbin, :] = np.dot(w_bins, ccl.angular_cl(cosmo, galaxy_tracers[ibin-1], galaxy_tracers[jbin-1], binning['ells_theory']))
                cls[jbin, ibin, :] = np.dot(w_bins, ccl.angular_cl(cosmo, galaxy_tracers[jbin-1], galaxy_tracers[ibin-1], binning['ells_theory']))

    return cls

def calculate_knox_covariance(cls, binning, fsky):
    
    n_maps = cls.shape[0]
    n_cross = (n_maps * (n_maps + 1)) // 2
    covar = np.zeros([n_cross, n_ell, n_cross, binning['n_ell']])

    id_i = 0
    for i1 in range(n_maps):
        for i2 in range(i1, n_maps):
            id_j = 0
            for j1 in range(n_maps):
                for j2 in range(j1, n_maps):
                    cl_i1j1 = cls[i1, j1, :]
                    cl_i1j2 = cls[i1, j2, :]
                    cl_i2j1 = cls[i2, j1, :]
                    cl_i2j2 = cls[i2, j2, :]
                    # Knox formula
                    cov = (cl_i1j1 * cl_i2j2 + cl_i1j2 * cl_i2j1) / (binning['delta_ell'] * fsky * (2 * binning['ells'] + 1))
                    covar[id_i, :, id_j, :] = np.diag(cov)
                    id_j += 1
            id_i += 1

    covar = covar.reshape([n_cross * n_ell, n_cross * n_ell])
    
    return covar

cosmo = ccl.Cosmology(Omega_c=0.25,
                      Omega_b=0.05,
                      h=0.677,
                      m_nu=0.06,
                      n_s=0.965,
                      A_s=2.11e-9,
                      Omega_k=0.0,
                      Neff=3.046)

ell_min = 100
ell_max = 1000
n_ell = 20
delta_ell = (ell_max - ell_min) // n_ell

ells = ell_min + (np.arange(n_ell) + 0.5) * delta_ell

ells_win = np.arange(ell_min, ell_max + 1)
wins = np.zeros([n_ell, len(ells_win)])

for i in range(n_ell):
    wins[i, i * delta_ell : (i + 1) * delta_ell] = 1.0
    
Well = sacc.BandpowerWindow(ells_win, wins.T)

binning = {'ell_max': ell_max,
           'n_ell': n_ell,
           'delta_ell': delta_ell,
           'ells_theory': ells_win,
           'ells': ells,
           'Well': Well}

dndz_skao = np.loadtxt('./data/nz_skao_med_deep.txt')

Asky_skao = 5000 * units.deg * units.deg
fsky_skao = Asky_skao.value / 41253

ngal_skao = np.array([0.675, 0.675, 0.675, 0.675])
sigma_e_skao = np.array([0.3, 0.3, 0.3, 0.3])

zstar = 1086
fsky_so = 0.051

tracer_so_k = ccl.CMBLensingTracer(cosmo, z_source=zstar)

# Approximation to SO LAT beam
fwhm_so_k = 2. * units.arcmin
sigma_so_k = (fwhm_so_k.to(units.rad).value / 2.355)
ell_beam = np.arange(3000)
beam_so_k = np.exp(-ell_beam * (ell_beam + 1) * sigma_so_k**2)

Nell_so = np.loadtxt('./data/nlkk_v3_1_0deproj0_SENS1_fsky0p4_it_lT30-3000_lP30-5000.dat', delimiter=' ')

ell_soNell, Nell_so_k = Nell_so[:,0], Nell_so[:,7]

# Bin Nells to the same as data
bindx = np.digitize(ell_soNell, ells_win, right=True)
Nell_so_k_binned = [np.mean(Nell_so_k[bindx == i]) for i in range(0, len(ells_win))]

skao_tracers = setup_gamma_tracers(cosmo, dndz_skao, A_IA=None)

skao_cls = calculate_cl(cosmo, skao_tracers, tracer_so_k, dndz_skao, ngal_skao, sigma_e_skao, binning)

skao_cov = calculate_knox_covariance(skao_cls, binning, fsky_skao)

s = sacc.Sacc()

for ibin in np.arange(1,dndz_skao.shape[-1]):

    s.add_tracer('NZ', 'gs_skao_bin{}'.format(ibin),
                quantity='galaxy_shear',
                spin=2,
                z=dndz_skao[:,0],
                nz=dndz_skao[:,ibin],
                metadata={'ngal': ngal_skao[ibin-1], 'sigma_e': sigma_e_skao[ibin-1]})

s.add_tracer('Map', 'ck_so',
            quantity='cmb_convergence',
            spin=0,
            ell=ell_beam,
            beam=beam_so_k)

for ibin in np.arange(1,dndz_skao.shape[-1]):
    for jbin in np.arange(1,dndz_skao.shape[-1]):
        if ibin<=jbin:
            s.add_ell_cl('galaxy_shear_cl_ee',
                         'gs_skao_bin{}'.format(ibin),
                         'gs_skao_bin{}'.format(jbin),
                         ells, skao_cls[ibin, jbin, :],
                         window=Well)

    s.add_ell_cl('cmbGalaxy_convergenceShear_cl_e',
                 'gs_skao_bin{}'.format(ibin),
                 'ck_so',
                 ells, skao_cls[0, ibin, :],
                 window=Well)

s.add_ell_cl('cmb_convergence_cl',
             'ck_so',
             'ck_so',
             ells, skao_cls[0, 0, :],
             window=Well)

s.add_covariance(skao_cov)

s.remove_selection(data_type='cmb_convergence_cl')
s.remove_selection(data_type='galaxy_shear_cl_ee')

s.save_fits('./data/gs_skao-ck_so.sim.sacc.fits', overwrite=True)

from matplotlib import pyplot as plt

plt.figure()
for t1, t2 in s.get_tracer_combinations():
    l, cl, cov = s.get_ell_cl(None, t1, t2, return_cov=True)
    err = np.sqrt(np.diag(cov))
    plt.errorbar(l, cl, err, label='%s - %s' % (t1, t2))
plt.loglog()
plt.legend(ncol=2)
plt.xlabel(r'$\ell$',fontsize=16)
plt.ylabel(r'$C_\ell$',fontsize=16)
plt.savefig('./plots/skao_x_so.png', dpi=300, bbox_inches='tight')