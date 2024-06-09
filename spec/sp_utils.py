#! /usr/bin/env python3

import os, sys
import warnings
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from astropy.stats import sigma_clip
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from matplotlib.gridspec import GridSpec
from astropy.visualization import ZScaleInterval
zsc = ZScaleInterval()
#from scipy import stats
#ae, loce, scalee = stats.skewnorm.fit(sample)

# Gaussian profile for peak identification
def Gprofile(r, a, x, s, bg):
  return a * np.exp(-(r-x)**2 / (2*s**2)) +bg

# profile fitting function to find the peak location
def centeriodX(xarr, yarr, X0, Warc, fwhm=5., showplt=False):
  Xmask = np.zeros(len(xarr), dtype=bool)
  Xmask[round(X0)-Warc:round(X0)+Warc+1] = 1

#  print ([np.amax(yarr[Xmask]), np.mean(xarr[Xmask]), fwhm, np.amin(yarr[Xmask])])
#  plt.plot(xarr[Xmask], yarr[Xmask], label='obs')
#  plt.legend()
#  plt.show()

#  popt, pcov = curve_fit(Gprofile, xarr[Xmask], yarr[Xmask], p0=[np.amax(yarr[Xmask]), np.mean(xarr[Xmask]), fwhm, 0.])
  popt, pcov = curve_fit(Gprofile, xarr[Xmask], yarr[Xmask], p0=[np.amax(yarr[Xmask]), np.mean(xarr[Xmask]), fwhm, np.amin(yarr[Xmask])], maxfev=2500)
  a, x, s, bg = popt
  if showplt:
#    print (popt, np.diag(pcov)**0.5)
    plt.title('a=%.2f, x=%.2f, s=%.2f, bg=%.2f' %(a, x, s, bg))
    plt.plot(xarr[Xmask], yarr[Xmask], label='obs')
    plt.plot(xarr[Xmask], Gprofile(xarr[Xmask], *popt), label='fit')
    plt.legend()
    plt.show()
  return popt, np.diag(pcov)**0.5, xarr[Xmask], Gprofile(xarr[Xmask], *popt)

# load spectrophotometric standard for deriving the response curve
def load_stddata(stdID, source):
  # define path of the std data
  if source in ['iraf', 'IRAF']:
   DBpath = '/home/ycc/python/sp/spectrophotometric_standards/IRAF/'
   """ function reserved """
  
  if source in ['eso', 'ESO']:
   DBpath = '/home/ycc/python/sp/spectrophotometric_standards/ESO/'
   DB_list = glob('%s*/f%s*' %(DBpath, stdID))	# 
   if len(DB_list) != 0:
     for DB_path in DB_list:
       print (DB_path)
       stdDB = np.loadtxt(DB_path, usecols=(0, 1)).T
     return stdDB[0], stdDB[1]
   return None, None

  if source in ['2nd']:
    DBpath = '/home/ycc/python/sp/spectrophotometric_standards/2nd/'
    DB1 = np.loadtxt(DBpath+'table1.dat', usecols=[2]+list(range(16,103)))	# 10**-4 erg/(cm**2 s cm)
    stdID1 = DB1[:,0].astype(int).astype(str)
    if stdID in stdID1:
      DBflux1 = DB1[:,1:]
      stdidx = np.argmax(stdID1==stdID)
      wl = np.linspace(3225, 7525, 87)
      return wl, DBflux1[stdidx]

    DB3 = np.loadtxt(DBpath+'table3.dat')	# erg/(cm**2 s cm)
    stdID3 = DB3[:,0].astype(int).astype(str)
    if stdID in stdID3:
      DBflux3 = DB3[:,1:]
      stdidx = np.argmax(stdID3==stdID)
      wl = np.linspace(3225, 7675, 90)
      return wl, DBflux3[stdidx]

    return None, None
    DB5 = np.loadtxt(DBpath+'table5.dat')	# erg/(cm**2 s cm)
    stdID5 = DB5[:,0].astype(int).astype(str)
    if stdID in stdID5:
      DBflux5 = DB5[:,1:]
      stdidx = np.argmax(stdID5==stdID)
      wl = np.linspace(5875, 10875, 98)
      return wl, DBflux5[stdidx]



# SP tracing for producing the mask of the SP and BG (ver:240409)
class APmask():
  def __init__(self, sp2D):
    self.sparr = sp2D
    self.Ny, self.Nx = np.shape(self.sparr)

  def flat_mask(self, flatarr, thershold=0.5):
    self.sparr[flatarr < thershold] = np.nan

  def traceAP(self, Ncell=16, X0cut=0, Yinit='', Ywidth=10., plt_yfit=False):
  # Ncell: number of cells for AP tracing (should be an even number)
  # X0cut: edge cut at the blue end
  # Yinit: the initial Y location (at mid-X) for forced extracting 1D spec in multiple objects
  # Ywidth: constraint the width of the "forced" 1D spec (not smaller than 15)
  # plt_yfit: show the fitted Y location
    Nxcol = int((self.Nx - X0cut)/Ncell)	# number of X in each cell
    Tpars = []
    # reorder cid
    cid_order = np.arange(Ncell).reshape(2,int(Ncell/2))
    cid_order[0] = cid_order[0][::-1]
    # trace Y-center
    for idx, cid in enumerate(cid_order.T.reshape(Ncell)):
      Xc = X0cut+(Nxcol*(cid+0.5))		# center of each x-bin
      Ymask = ~np.zeros(self.Ny, dtype=bool)	# define Ymask for 1D Y profile (True=accepted for Gfit)
      if len(Tpars) >= 2:
        Y04fit = np.polyval(lpars, Xc)
      if Yinit != '':
        Y0 = Yinit if len(Tpars) < 2 else Y04fit
#        print (Y0, Ywidth, np.arange(self.Ny))
        Ymask[Y0-Ywidth > np.arange(self.Ny)] = 0
        Ymask[np.arange(self.Ny) > Y0+Ywidth] = 0
      Yprof = np.mean(self.sparr[:,X0cut+Nxcol*cid:X0cut+Nxcol*(cid+1)], axis=1)
#      Yprof = np.median(sigma_clip(self.sparr[:,X0cut+Nxcol*cid:X0cut+Nxcol*(cid+1)], axis=1), axis=1)
#      Yprof = np.median(self.sparr[:,X0cut+Nxcol*cid:X0cut+Nxcol*(cid+1)], axis=1)
      Ymask *= ~np.isnan(Yprof)

      try:
        y4f, i4f = np.arange(self.Ny)[Ymask], Yprof[Ymask]
        if len(Tpars) < 2:
          Y04fit = y4f[np.argmax(i4f)]
        bounds = [[0., np.inf], [Y04fit-10, Y04fit+10], [1., 10.], [np.median(i4f)-100, np.median(i4f)+100]]	# [a, x, s, bg]
        popt, pcov = curve_fit(Gprofile, y4f, i4f, p0=[np.amax(i4f)-np.median(i4f), Y04fit, 3., np.median(i4f)], bounds=np.array(bounds).T)
        Tpars.append([Xc, popt[1], popt[2], pcov[1,1]**0.5])
        if plt_yfit:
          plt.title(str(Xc)+': '+str(X0cut+Nxcol*cid)+','+str(X0cut+Nxcol*(cid+1)))
          plt.scatter(y4f, i4f, label='yi4f')
          plt.plot(Yprof, label='full_yprof')
          plt.plot(y4f, Gprofile(y4f, *popt), label='fitted')
          plt.xlim(0,self.Ny)
          plt.legend()
          plt.show()
          plt.close()
      except RuntimeError:
        _ = 1

      if len(Tpars) >= 2:
        lcorDB = np.array(Tpars)
        lpars = np.polyfit(lcorDB[:,0], lcorDB[:,1], 1)

  # polyfit of ap center
    Tpars = np.array(Tpars)
    self.Ycpars = np.polyfit(Tpars[:,0], Tpars[:,1], 3, w=1./Tpars[:,3])
    self.Wpars = np.polyfit(Tpars[:,0], Tpars[:,2], 3, w=1./Tpars[:,3])
    self.Yc = np.polyval(self.Ycpars, np.arange(self.Nx))		# Y center
    self.W = np.polyval(self.Wpars, np.arange(self.Nx))			# width of the Gaussian profile (sigma)
    return self.Yc, self.W, Tpars

  def fieldstar_mask(self, maskarr):
    self.sparr[maskarr] = np.nan	# array mask for AP tracing (True=BLOCKED pixels for Gfit)

  def mk_mask(self, Wap=2., bg_gap=10, Yoffset=0.):
    Yc = self.Yc+Yoffset
    ap_grid = np.meshgrid(np.arange(self.Nx), np.arange(self.Ny))
    self.ap_mask =   (ap_grid[1] <= Yc+Wap*2.335*self.W/2.)        & (ap_grid[1] >= Yc-Wap*2.335*self.W/2.)
    self.bg_mask = ~((ap_grid[1] <= Yc+Wap*2.335*self.W/2.+bg_gap) & (ap_grid[1] >= Yc-Wap*2.335*self.W/2.-bg_gap))
    return self.ap_mask, self.bg_mask
 # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #




class wlcal():
  def __init__(self, arc1D, FWHM=5., arclist_path='/home/ycc/python/sp/ArNe_LISA.arc'):
    # arc1D: 1D arc spectrum for wavelength identification
    # FWHM: initial FWHM of the arc emission for gaussian fitting (depend on the slit width)
    self.fwhm = FWHM
    self.arc1D = arc1D
    self.arc_list = np.loadtxt(arclist_path, dtype=str)

  def autoID(self, Llist, res, podr, showfit=False, showplt=False):
    for line in Llist:
      wl = float(line[0])
      x_init = self.X0 + (wl-self.wl0)/res if len(self.arcDB) < 2 else np.polyval(self.polypars, wl)
      if 0 > x_init or x_init > len(self.arc1D):
        continue
      Xmask = np.zeros(len(self.arc1D), dtype=bool)
      Xmask[int(x_init)-self.Warc:int(x_init)+self.Warc+1] = 1

#      print (self.arcDB)
#      try:
      print ('%4s %s: ' %(line[1], line[0]))
      Gpar, Gerr, Y, Yfit = centeriodX(self.Xlist, self.arc1D, x_init, self.Warc, fwhm=self.fwhm, showplt=showplt)
      Xc, Xerr = Gpar[1], Gerr[1]
      if showfit == True:
        print ('%4s %s: %8.3f %8.3f (%6.3f)' %(line[1], line[0], x_init, Xc, Xc-x_init))
      if Xerr <= self.thershold and Gpar[0] > 0. and Gpar[2] < 2.5*self.fwhm and Gpar[1] != 0.:
        self.arcDB.append([line[0], line[1], x_init, Xc, Xerr,     Y, Yfit])
#                         [     wl,   arcID,     X0, Xc, Xerr, Yprof, Yfit]
        if len(self.arcDB) >= 2:
          RDB = np.array(self.arcDB, dtype=object)
          fodr = podr if len(self.arcDB) > podr else len(self.arcDB)-1
          self.polypars   = np.polyfit(RDB[:,0].astype(float), RDB[:,3].astype(float), fodr, w=1./RDB[:,4].astype(float))
          self.polypars_r = np.polyfit(RDB[:,3].astype(float), RDB[:,0].astype(float), fodr, w=1./RDB[:,4].astype(float))
          warnings.filterwarnings('ignore')
          warnings.warn('Covariance of the parameters could not be estimated',)

#      except RuntimeError:
#        _ = line
#        if showfit == True:
#          print ('%4s %s: %8.3f' %(line[1], line[0], x_init))
#      except SystemError:
#        _ = line
#        if showfit == True:
#          print ('%4s %s: %8.3f' %(line[1], line[0], x_init))


  def identify(self, res, wl0, Warc=10, thershold=0.2, podr=1, wl0isline=True, pfit=False, idvplt=False, test=False):
    # res: nominail resolution [A/pix], 0.31 for UVEX1200, 
    # wl0: wavelength of the strongest emission in the arc frame
    # Warc: width (number of pixel) for gaussian fitting [pix]
    # thershold: thershold for the uncertainty of gaussian fitting
    # podr: polynomial order of the wavelength-pixelX correlation
    # pfit: print best-fitted params
    # idvplt: plot individual fitting result
    # test: if in testing mode, a arc plot will be shown
    self.wl0 = wl0
    self.Warc = Warc
    self.thershold = thershold
    if test == True:
      plt.plot(self.arc1D)
      plt.show()
      sys.exit()

    self.arcDB = []
  # priliminary identification of wl0
    self.Xlist = np.arange(len(self.arc1D))
    arcX0 = np.argmax(self.arc1D)

    Gpar, Gerr, Y, Yfit = centeriodX(self.Xlist, self.arc1D, arcX0, self.Warc, fwhm=self.fwhm, showplt=idvplt)
    self.X0, Xerr = Gpar[1], Gerr[1]
    if wl0isline:
      arcID = self.arc_list[self.arc_list[:,0] == '%.4f' %(wl0)][0][1]
      print ('%4s %.4f:          %8.3f' %(arcID, wl0, self.X0))
      self.arcDB.append([wl0, arcID, self.X0, self.X0, Xerr, Y, Yfit])

  # searching arcs to the red end
    arcR = self.arc_list[self.arc_list[:,0].astype(float)>wl0]
    self.autoID(arcR, res, podr, showfit=pfit, showplt=idvplt)

  # searching arcs to the blue end
    arcB = self.arc_list[self.arc_list[:,0].astype(float)<wl0][::-1]
    self.autoID(arcB, res, podr, showfit=pfit, showplt=idvplt)

  # convert from X to wl
    self.Xwl = np.polyval(self.polypars_r, self.Xlist)
#    return np.polyval(polypars_r, Xlist)
    print 
    return self.polypars, self.polypars_r


  def refine_arc(self, Warc='', iteration=3, podr=3, idvplt=False):
  # refine the fitting
#    self.Warc -= 0
    Warc = self.Warc if Warc == '' else Warc
    for i in range(iteration):
      all_list_X = np.polyval(self.polypars, self.arc_list[:,0].astype(float))
      list_mask = (0 <= all_list_X) & (all_list_X <= len(self.arc1D))
      arcDB = []
      for idx, x_init in enumerate(all_list_X[list_mask]):
        try:
          Gpar, Gerr, Y, Yfit = centeriodX(self.Xlist, self.arc1D, x_init, Warc, fwhm=self.fwhm, showplt=False)
        except RuntimeError:
          continue
        Xc, Xerr = Gpar[1], Gerr[1]
#        print (Xc, Xerr, Gpar[0], Gpar[2])
        if Xerr <= self.thershold and Gpar[0] > 0. and Gpar[2] < 2.5*self.fwhm and Xerr != 0.:
#          print (Xc, Xerr, Gpar[0], Gpar[2])
          arcDB.append([self.arc_list[list_mask][idx][0], self.arc_list[list_mask][idx][1], x_init, Xc, Xerr, Y, Yfit])
#          print (self.arc_list[list_mask][idx][0], self.arc_list[list_mask][idx][1], Xc, Xerr)
#      dd = input('')
      self.arcDB = np.array(arcDB, dtype=object)
      self.polypars = np.polyfit(self.arcDB[:,0].astype(float), self.arcDB[:,3].astype(float), podr, w=1./self.arcDB[:,4].astype(float))
    self.polypars_r = np.polyfit(self.arcDB[:,3].astype(float), self.arcDB[:,0].astype(float), podr, w=1./self.arcDB[:,4].astype(float))
    self.Xwl = np.polyval(self.polypars_r, self.Xlist)
    return self.polypars, self.polypars_r


  def plt_wlcal(self, title='', savepath='', pltraw=False, rawarc=[None]):
    arcminmax = np.amin(self.arc1D), np.amax(self.arc1D)
    dwl = np.amax(self.Xwl) - np.amin(self.Xwl)
    pixres = dwl/len(self.Xlist)
    f = plt.figure(figsize=(22, 15))
    plt.suptitle(title)
    gs1 = GridSpec(4,1)
    gs1.update(left=0.10, right=0.95, hspace=0.0)
    ax1 = plt.subplot(gs1[0,0])
    ax2 = plt.subplot(gs1[1:3,0], sharex=ax1)
    ax3 = plt.subplot(gs1[3,0], sharex=ax1)

    ax1.plot(self.Xwl, self.arc1D)
    for i in self.arcDB:
      ax1.plot(np.polyval(self.polypars_r, i[5]), i[6], color='C1')
    ax2.plot(self.Xwl, self.arc1D)
    for i in self.arcDB:
#      print (i[0], i[3])
      ax2.plot(np.polyval(self.polypars_r, i[5]), i[6], color='C1')
      ax2.annotate('%s %s' %(i[1], i[0]), (np.polyval(self.polypars_r, i[3])-3., arcminmax[0]-(arcminmax[1]-arcminmax[0])/27.5), ha='left', rotation=60, color='C2')
    ax2.set_ylim(arcminmax[0]-(arcminmax[1]-arcminmax[0])/22.5, arcminmax[1]/10.)
#    ax1.scatter(RDB[:,0], RDB[:,1])
    arcDB = np.array(self.arcDB, dtype=object)
#    print (arcDB)
    ax3.scatter(arcDB[:,0].astype(float), arcDB[:,0].astype(float)-np.polyval(self.polypars_r, arcDB[:,3].astype(float)), label='N = %d' %(len(arcDB)))
    ax3.plot((self.Xwl[:-1]+self.Xwl[1:])/2., 0.*np.diff(self.Xwl), '-', color='gray', alpha=0.25)
    ax3.plot((self.Xwl[:-1]+self.Xwl[1:])/2., np.diff(self.Xwl), '--', color='gray', alpha=0.33, label=r'1 pix=%.3f $\AA$' %(pixres))
    ax3.plot((self.Xwl[:-1]+self.Xwl[1:])/2., -1*np.diff(self.Xwl), '--', color='gray', alpha=0.33)
    ax3.set_xlim(np.amin(self.Xwl)-dwl/50, np.amax(self.Xwl)+dwl/50)
    ax3.set_xlabel('Wavelength [$\AA$]')
    ax3.set_ylabel(r'd$\lambda$ [$\AA$]')
    ax3.legend()
    if savepath != '':
      plt.savefig(savepath)
    else:
      plt.show()
    plt.close()

#    plt.scatter(arcDB[:,0], arcDB[:,3])
#    plt.plot(np.polyval(self.polypars_r, arcDB[:,3].astype(float)), arcDB[:,3])
#    plt.xlabel('wavelength')
#    plt.ylabel('pixel')
#    plt.show()


def sp_interpolate(sps, step=0.1):
  # type(sps) = list
  # find wlmin and wlmax
  wlmin, wlmax = -np.inf, np.inf
  for sp in sps:
    if np.amin(sp[0]) > wlmin:
      wlmin = np.amin(sp[0])
    if np.amax(sp[0]) < wlmax:
      wlmax = np.amax(sp[0])
  # interpolation
  wl_list = np.arange(wlmin, wlmax, step)
  newsp_list = []
  for sp in sps:
    f = interp1d(sp[0], sp[1])
    newsp_list.append(np.array([wl_list, f(wl_list)]))
  return np.array(newsp_list)

def sp_norm(sps, xrange):
  # type(sps) = ndarray
  nFs = []
  newsp = []
  for sp in sps:
    Xmask = (xrange[0]<sp[0]) * (sp[0]<xrange[1])
    nF = np.mean(sp[1][Xmask])
    nFs.append(nF)
    newsp.append([sp[0], sp[1]/np.mean(sp[1][Xmask])])
  return np.array(nFs), np.array(newsp)

def sp_comb(sps, op, norm=''):
  newsp = sp_interpolate(sps, step=0.1)
  if norm != '':
    newsp = sp_norm(newsp, norm)[1]
  if op == 'sum':
    wl = newsp[0][0]
    sps = newsp[:,1]
    return np.array([wl, np.sum(sps, axis=0)])
  if op == 'mean':
    wl = newsp[0][0]
    sps = newsp[:,1]
    if norm != '':
      sps = sps/nfact
    return np.array([wl, np.mean(sps, axis=0)])
  if op == 'median':
    return np.array([sp1[0], sp1[1]+sp2[1]])

def sp_math(sp1, sp2, op, norm=[]):
  if len(norm) != 0:
    sp1, sp2 = sp_norm(np.array([sp1]+[sp2], dtype=object))
  if op == '+':
    return np.array([sp1[0], sp1[1]+sp2[1]])
  if op == '-':
    return np.array([sp1[0], sp1[1]-sp2[1]])
  if op == '*':
    return np.array([sp1[0], sp1[1]*sp2[1]])
  if op == '/':
    return np.array([sp1[0], sp1[1]/sp2[1]])

def senfunc(stdsp, fluxarr, podr=8, threshold=0, stepsize=250, wlmin_cutoff=3700, interpolate=False):
  newsp = sp_interpolate([stdsp]+[fluxarr])
  sfuncwl = newsp[0][0]
  Nsteps = int(np.shape(newsp)[2]/stepsize)
  X0_position = np.shape(newsp)[2] - stepsize*Nsteps
  sfuncwl = np.mean(np.reshape(sfuncwl[X0_position:],(Nsteps,stepsize)), axis=1)
  sfunc = np.mean(np.reshape(newsp[1][1][X0_position:],(Nsteps,stepsize)), axis=1)/np.mean(np.reshape(newsp[0][1][X0_position:],(Nsteps,stepsize)), axis=1)

  wl_mask = sfuncwl>wlmin_cutoff
  if interpolate:
    return interp1d(sfuncwl[wl_mask], sfunc[wl_mask], bounds_error=False, fill_value='extrapolate')
  ppars = np.polyfit(sfuncwl[wl_mask], sfunc[wl_mask], podr)
#  dy = np.polyval(ppars, sfunc[0])
#  chi2 = np.sum(dy**2/)
#  while np.amax(dy) > 

  plt.plot(sfuncwl[wl_mask], sfunc[wl_mask], label='senf')
  plt.plot(sfuncwl[wl_mask], np.polyval(ppars, sfuncwl[wl_mask]), label='sf_poly')
  plt.legend()
  plt.show()
  plt.close()
  return sfuncwl, ppars


class sp_math_():
  def __init__(self, data_array, rstep=0.1, data_array1=''):
  # data_array = [[wl, sp], [wl, sp], ...]
  # rstep is unitless
    # resample the input sp
    self.data_array = np.array(data_array)
    wl_min, wl_max = np.amax(self.data_array[:,0][:,0]), np.amin(self.data_array[:,0][:,-1])
    if data_array1 != '':
      self.data_array1 = np.array(data_array1)
      wl_min = np.amax(np.concatenate((self.data_array1[:,0][:,0], [wl_min])))
      wl_max = np.amin(np.concatenate((self.data_array1[:,0][:,-1], [wl_max])))
    self.newwl_list = np.arange(wl_min, wl_max, rstep)
    self.newsp_list = []
    for sp in self.data_array:
      f = interp1d(sp[0], sp[1])
      self.newsp_list.append(f(self.newwl_list))
    if data_array1 != '':
      for sp in self.data_array1:
        f = interp1d(sp[0], sp[1])
        self.newsp_list.append(f(self.newwl_list))
    self.newsp_list = np.array(self.newsp_list)

  def norm(self, wlrange):
  # wlrange = [wlmin, wlmax]
    self.norm_fact = []
    wl_mask = (wlrange[0]<=self.newwl_list) * (self.newwl_list<=wlrange[1])
    for idx, sp in enumerate(self.newsp_list):
      self.norm_fact.append(np.mean(sp[wl_mask]))
      self.newsp_list[idx] = self.newsp_list[idx]/np.mean(sp[wl_mask])
    self.norm_fact = np.array(self.norm_fact)
    return self.norm_fact

  def comb(self, method):
    if method == 'mean':
      self.sp_final = np.nanmean(self.newsp_list, axis=0)
    if method == 'median':
      self.sp_final = np.nanmedian(self.newsp_list, axis=0)
    return self.newwl_list, self.sp_final

  def math(self, method):
    if method == '%':
      return self.newwl_list, self.newsp_list[0]/self.newsp_list[1]

  def mkplot(self, title='', vlines=[], hlines=[]):
    for idx, sps in enumerate(self.data_array):
      plt.plot(sps[0], sps[1]/self.norm_fact[idx], color='gray', alpha=0.25)
    plt.plot(self.newwl_list, self.sp_final)
    if vlines != []:
      for v in np.array(vlines[1:], dtype=float):
        plt.axvline(v, linestyle=vlines[0])
    plt.title(title)
    plt.show()
    plt.close()


"""
if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser(description="Online star catalogs ver. 210712")
  parser.add_argument('-ra',  type=float, required=True, help="initial RA [deg]")
  parser.add_argument('-dec', type=float, required=True, help="initial Dec [deg]")
  parser.add_argument('-sr',  type=float, default=0.2, help="searching radius [deg]")
  parser.add_argument('-N',   type=int,   default=10,  help='return detections with at least N measurements')
  parser.add_argument('-db',  type=str,   default='apass', choices=['ps1', 'PS1', 'sdss', 'SDSS', 'apass', 'APASS'], help='select dateset to be request')
  parser.add_argument('-showURL', action="store_true", help='show request URL')
  parser.add_argument('-saveTXT', type=str, default=None, help='save result as given file name')
  args = parser.parse_args()

  if args.db in ['ps1', 'PS1']:
    print (ps1_cat(args.ra, args.dec, radius=args.sr, Nmea=args.N, showURL=args.showURL, savetxt=args.saveTXT))
  if args.db in ['sdss', 'SDSS']:
    print (sdss_cat(args.ra, args.dec, radius=args.sr, showURL=args.showURL, savetxt=args.saveTXT))
  if args.db in ['apass', 'APASS']:
    print (apass_cat(args.ra, args.dec, radius=args.sr, showURL=args.showURL, savetxt=args.saveTXT))
"""

