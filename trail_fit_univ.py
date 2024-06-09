#! /usr/bin/env python3

##### universal trail fitting package for solar system moving objects (ver.221025)
# 
# initialize object: 
#   TF = Trail_Fit(imgarr, XY0, pscale=0.39, boxsize=20)
#
# start trail fitting procedure
#   TF.fit_trail(profile, algorithm, iteration=1, L=0., PA=0., BG='', alpha=1., beta=1., sigma=1., FWHM=1.5, dXY=None)
#      profile='G' or 'M' (Gaussian/Moffat)
#      iteration=int (number of iteration during the trail fit)
#      algorithm='DE' or 'lsq' (differential evolution/least square minimization)
#        https://pablormier.github.io/2017/09/05/a-tutorial-on-differential-evolution-with-python/
#      some initial gausses: L/PA/BG/alpha/beta/sigma/FWHM/dXY
# 
# other functions: 
#   TF.FWHM()
#     return FWHM from trail fitting
#   TF.mkmod(arr_in, XYc)
#     remake the trail fitting model with different image dimension (arr_in) and star center (XYc)
#     use TF.mod for the trail fitting model in origin boxsize
#   TF.apphot(ap, gain, FWHM=None, SNR_threshold=3.)
#     aperture photometry based on the trail fitting result (star center, FWHM, and trail length)
#     assign photometric aperture ap=[inner, mid, outer] in unit of FWHM
#     the FWHM can be given manually if you don't want to apply the FWHM from trail fitting result
#   TF.mkplot(title_str='', output_basename='', ap=False, mp=False)
#     plot the trail fitting result (and overlaying with photometric aperture if enable)
#     title_str: the title which will shown on the top of the output plot
#     ap: enable it if you want to overlay the photometric aperture
#     mp: set True if the trail fitting task is using multiprocessing method
#
#####

import os, sys
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt
from scipy.stats import sigmaclip
from matplotlib.gridspec import GridSpec
from astropy.visualization import ZScaleInterval
from scipy.optimize import curve_fit, least_squares, differential_evolution



# define PSF model
def G_profile(r, par):	# Gaussian profile
  a, sigma = par
  return a * np.exp(-(r)**2 / (2 * sigma**2))

def M_profile(r, par):	# Moffat profile
  a, alpha, beta = par
  return a * (1+(r/alpha)**2) ** (-beta)

# define R-grid
def mkRgrid(imgarr, x0, y0, L=1., PA=0.):
  # PA0 = North, clockwise
  imgshape = np.shape(imgarr)
  X, Y = np.meshgrid(np.arange(imgshape[1]), np.arange(imgshape[0]))
  Xrot = np.fabs( (X-x0)*np.cos(np.radians(PA)) + (Y-y0)*np.sin(np.radians(PA)))
  Yrot = np.fabs(-(X-x0)*np.sin(np.radians(PA)) + (Y-y0)*np.cos(np.radians(PA))) - L/2.
  Yrot[Yrot<0.] = 0.
  return np.linalg.norm((Xrot, Yrot), axis=0)

# clip subframe for model fitting
def mk_subframe(fullarr, XY0, subsize):
  XY0, arrdim = np.array(XY0), np.shape(fullarr)
  subarr = np.full((subsize, subsize), np.nan)
  # zero point XY related to the full frame (XY0_init must be larger than [0, 0])
  XY0_int = np.array([max(0, XY0[0]-(subsize-1)/2), max(0, XY0[1]-(subsize-1)/2)]).astype(int)
  # cutoff subframe from the full frame
  subarr = fullarr[XY0_int[0]:XY0_int[0]+subsize, XY0_int[1]:XY0_int[1]+subsize]
  # convert XY0 in the subframe
  subXY0 = XY0-XY0_int
  return subarr, XY0_int, subXY0

# plot image array (overlayed with a given coordinate)
def plt_arr(arr, XY=''):
  zsc = ZScaleInterval()
  Zmin, Zmax = zsc.get_limits(arr)
  plt.imshow(arr, vmin=Zmin, vmax=Zmax, cmap='gray')
  if XY != '':
    plt.scatter(XY[0], XY[1], color='red', marker='x')
  plt.show()
  plt.close()


# main function here
class Trail_Fit:
  def __init__(self, imgarr, XY0, pscale=0.39, boxsize=20):
  # imgarr: full frame array for trail fit
  # input XY0: [X0, Y0] in ds9/MaxIm DL coordinate system
    self.XY0 = np.array(XY0)[::-1]	# remind the X-Y inversion between ds9/MaxIm DL and ndarray
    if type(imgarr) != np.ndarray or len(XY0) < 2:
      print ('Invided input data, skip the fitting procedure')
      return None
    self.fullarr = imgarr
    self.bs = boxsize
    self.ps = pscale

# define fitting model
  def errfunc(self, pars):		# function for lsq and DE
#    X, Y, L, PA, BG, f0, sigma = pars
    rmap = mkRgrid(self.subarr, pars[0], pars[1], L=pars[2], PA=pars[3])
    self.mod = self.PSF_mod(rmap, pars[5:]) + pars[4]
    return np.sum(np.fabs(self.subarr-self.mod))

  def errf_cf(self, grid, pars):	# function for curve_fit
    rmap = mkRgrid(self.subarr, pars[0], pars[1], L=pars[2], PA=pars[3])
    self.mod = self.PSF_mod(rmap, pars[5:]) + pars[4]
    return self.mod

# trail fitting
  def fit_trail(self, profile, algorithm, iteration=1, L=0., PA=0., BG='', fp='', alpha=1., beta=1., sigma=1., FWHM=1.5, dXY=None):
    if profile not in ['G', 'M'] or algorithm not in ['DE', 'lsq']:
      return None

    Niteration = 0
    XY0 = self.XY0
    while Niteration < iteration:
    # subframe cutoff
      self.subarr, self.initXY0, self.initsubc = mk_subframe(self.fullarr, XY0, self.bs)
    # define object mask for initial guess
      obj_mask = np.zeros(np.shape(self.subarr), dtype=bool)
      obj_mask[int(self.initsubc[0])-3:int(self.initsubc[0])+3, int(self.initsubc[1])-3:int(self.initsubc[1])+3] = 1
    # initial guess
      ndmax = np.amax(np.shape(self.subarr))
      f0 = np.amax(self.subarr[obj_mask]) if fp == '' else float(fp)
      BG0 = np.median(self.subarr[~obj_mask]) if BG == '' else BG
      sig0 = FWHM/(self.ps * 2.*(2.*np.log(2))**0.5) if sigma == 1. else sigma
      L_bound = [0., ndmax/4.] if L == 0 else [L*0.5, L*2.]
      PA_bound = [0., 180.] if PA == 0 else [PA-30., PA+30.]
      XY_bound = [ndmax*1/4.,ndmax*3/4.] if dXY == None else [ndmax/2.-dXY,ndmax/2.+dXY]
      a_bound = [1e-5,10.] if alpha == 1. else [alpha*0.8, alpha*1.25]
      b_bound = [1e-5,10.] if beta == 1. else [beta*0.8, beta*1.25]

    # define function for fit, bounds, and initial params
      if profile == 'G':
        self.PSF_mod = G_profile
        par0 = self.initsubc[0], self.initsubc[1], L, PA, BG0, f0, sig0
        par_bounds = np.array([XY_bound, XY_bound, L_bound, PA_bound, [BG0-500.,BG0+500.], [f0*0.8,f0*1.25], [sigma*0.5,sigma*2.0]])
      if profile == 'M':
        self.PSF_mod = M_profile
        par0 = self.initsubc[0], self.initsubc[1], L, PA, BG0, f0, alpha, beta
        par_bounds = np.array([XY_bound, XY_bound, L_bound, PA_bound, [BG0-500.,BG0+500.], [f0*0.8,f0*1.25], a_bound, b_bound])
      par_bounds[5] = par_bounds[5] if par_bounds[5][0]<par_bounds[5][1] else par_bounds[5][::-1]

    # model fitting
      if algorithm == 'DE':
        modparams = differential_evolution(self.errfunc, bounds=par_bounds)
        self.par = modparams.x
      if algorithm == 'lsq':
        modparams = least_squares(self.errfunc, par0, bounds=par_bounds.T)
        self.par = modparams.x
      self.XYfit = self.initXY0[::-1] + self.par[:2]	# the center coordinate in the full frame
      self.dXY = self.par[:2] - self.initsubc
      XY0 = self.XYfit[::-1]  # remember to reverse XY
      Niteration += 1
    return self.par

# caculate the FWHM
  def FWHM(self):	# [in "]
    return self.ps * self.par[6] * 2.*(2.*np.log(2))**.5 if len(self.par) == 7 else self.ps * 2*self.par[6]*(2**(1/self.par[7])-1)**.5

# make the 2D PSF model
  def mkmod(self, arr_in, XYc):
    Rmap = mkRgrid(arr_in, XYc[0], XYc[1], L=self.par[2], PA=self.par[3])
    return self.PSF_mod(Rmap, self.par[5:]) + self.par[4]

# aperture photometry, only availiable for full array
  def apphot(self, ap, gain, FWHM=None, SNR_threshold=3.):
    self.apfwhm = self.FWHM()/self.ps if FWHM == None else FWHM/self.ps	# [in pixel]
    self.apfwhm = self.apfwhm if self.apfwhm <= 10. else 10.
    tL = self.par[2]
    ap_size = int(round(self.apfwhm * max(ap) + tL))*2 + 5	# identify the size of subarr for apphot
    ap_size = ap_size if ap_size <= 50 else 50

  # subframe cutoff
    self.apsubarr, self.apXY0, self.apsubc = mk_subframe(self.fullarr, self.XYfit[::-1], ap_size)
  # create R-grid and ap-mask
    ap_shape = np.shape(self.apsubarr)
    self.ap_grid = np.meshgrid(np.arange(ap_shape[1]), np.arange(ap_shape[0]))
    self.dist = mkRgrid(self.apsubarr, self.apsubc[1], self.apsubc[0], L=self.par[2], PA=self.par[3])
    self.ap_mask = np.zeros(np.shape(self.dist), dtype=int)
    for r in ap:
      self.ap_mask[self.dist <= self.apfwhm*r] += 1
  # apphot
    Na = np.sum(self.ap_mask==3)	# number of pixel in aperture
    flux_w_bg = np.sum(self.apsubarr[self.ap_mask==3])
    if np.sum(self.ap_mask==1) <= 5:
      return
    BG_idv = sigmaclip(self.apsubarr[self.ap_mask==1], low=3., high=3.)[0]
    Nb = len(BG_idv)			# number of pixel in background
    BG_mean, BG_median, BG_stdev = np.mean(BG_idv), np.median(BG_idv), np.std(BG_idv)
    flux = flux_w_bg - BG_median*Na
    SNR = self.par[5]/BG_stdev
    if SNR < SNR_threshold or flux <= 0.:
      return
    # uncertainty in ADU, 
    err = flux/float(gain) + (Na+(np.pi*Na*Na)/(Nb*2.))*BG_stdev**2
    # uncertainty in magnitude, https://amostech.com/TechnicalPapers/2020/Non-Resolved-Object-Characterization/Castro.pdf
    err_m = 2.5*np.log10(np.exp(err**0.5/flux))
    return np.array([flux, err**0.5, SNR, err_m, flux_w_bg, BG_median, Na, Nb])

# plot summary of the PSF fitting
  def mkplot(self, title_str='', output_basename='', ap=False, mp=False):
    if mp == True:	# for multiprocessing task only
      import matplotlib
      matplotlib.use('agg')
    f = plt.figure(figsize=(9.5, 6))
    gs1 = GridSpec(2,3)
    ax1 = plt.subplot(gs1[0,0])
    ax2 = plt.subplot(gs1[1,0])
    ax3 = plt.subplot(gs1[0:,1:])
    gs1.update(left=0.05, right=0.975, top=0.875, bottom=0.05, hspace=0.1, wspace=0.1)
    zsc = ZScaleInterval()

    subarr, best_model = self.subarr, self.mod
    XYinit, XYfit = self.initsubc, self.par[0:2]
    if ap == True:
      plt.suptitle('%sbest fitted objXY=(%.3f, %.3f), initial objXY=(%.3f, %.3f), dXY=(%.3f, %.3f)\npeak flux=%.2f ADU, BG=%.2f, L=%.2f pix, PA=%.2f deg., FWHM=%.2f arcsec' %(title_str, self.XY0[1]+self.dXY[0], self.XY0[0]+self.dXY[1], self.XY0[1], self.XY0[0], self.dXY[0], self.dXY[1], self.par[5], self.par[4], self.par[2], self.par[3], self.apfwhm*self.ps))
      subarr, best_model = self.apsubarr, self.mkmod(self.apsubarr, self.apsubc[::-1])
      XYinit, XYfit = self.apsubc[::-1], self.apsubc[::-1]
      ax1.contour(self.ap_grid[0], self.ap_grid[1], self.ap_mask, levels=[0.5, 1.5, 2.5], colors=['yellow', 'yellow', 'red'], alpha=0.5)
    else:
      plt.suptitle('%sbest fitted objXY=(%.3f, %.3f), initial objXY=(%.3f, %.3f), dXY=(%.3f, %.3f)\npeak flux=%.2f ADU, BG=%.2f, L=%.2f pix, PA=%.2f deg., FWHM=%.2f arcsec' %(title_str, self.XY0[1]+self.dXY[0], self.XY0[0]+self.dXY[1], self.XY0[1], self.XY0[0], self.dXY[0], self.dXY[1], self.par[5], self.par[4], self.par[2], self.par[3], self.FWHM()))
    residual = subarr - best_model

    Zmin, Zmax = zsc.get_limits(subarr)
    ax1.imshow(subarr, vmin=Zmin, vmax=Zmax, cmap='gray')
    ax1.scatter(XYinit[0], XYinit[1], color='blue', marker='x')

    ax2.imshow(best_model, vmin=Zmin, vmax=Zmax, cmap='gray')
    ax2.scatter(XYfit[0], XYfit[1], color='orange', marker='+')

    Zmin, Zmax = zsc.get_limits(residual)
    ax3.imshow(residual, vmin=Zmin, vmax=Zmax, cmap='gray')
    ax3.scatter(XYinit[0], XYinit[1], color='blue', label='init. center', marker='x')
    ax3.scatter(XYfit[0], XYfit[1], color='orange', label='fitted center', marker='+')
    ax3.legend()

    if output_basename != '':
      plt.savefig(output_basename)
    else:
      plt.show()
    plt.close()


