import os, sys
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from astropy.io import fits
sys.path.append('/home/ycc/python')
#import sp.sp_utils as spu
import sp.sp_utils_new as spu



Ylim = [1200, 1860]
arcwl = sys.argv[1]

for i in glob('LOT20240522/proc/1200_*.fit'):
  if arcwl+'.fit' not in i:
    continue
  arr = fits.getdata(i)
#  os.system('ds9 -zscale %s &' %(i))

  if arcwl == '950':
    arc1D = np.median(arr[1590:1610,:], axis=0)
    wlc = spu.wlcal(arc1D, FWHM=3., arclist_path='/home/ycc/python/sp/ArNe_UVEX1200_23um')
    polypars, polypars_r = wlc.identify(res=0.292, wl0=9122.9670, Warc=7, podr=2, thershold=1., idvplt=False)	# 950
    wlc.plt_wlcal()
    wlc.refine_arc(podr=3, Warc=5, iteration=1)

  if arcwl == '900':
    arc1D = np.median(arr[1590:1610,:], axis=0)
    wlc = spu.wlcal(arc1D, FWHM=3., arclist_path='/home/ycc/python/sp/ArNe_UVEX1200_23um')
    polypars, polypars_r = wlc.identify(res=0.295, wl0=9122.9670, Warc=7, podr=2, thershold=1., idvplt=True)	# 900
    wlc.plt_wlcal()
    wlc.refine_arc(podr=5, Warc=5, iteration=1)

  if arcwl == '850':
    arc1D = np.median(arr[1590:1610,:], axis=0)
    wlc = spu.wlcal(arc1D, FWHM=3., arclist_path='/home/ycc/python/sp/ArNe_UVEX1200_23um')
    polypars, polypars_r = wlc.identify(res=0.300, wl0=8115.3110, Warc=10, podr=2, thershold=1., idvplt=False)	# 850
    wlc.plt_wlcal()
    wlc.refine_arc(podr=5, Warc=5, iteration=1)

  if arcwl == '800':
    arc1D = np.median(arr[1590:1610,:], axis=0)
    wlc = spu.wlcal(arc1D, FWHM=3., arclist_path='/home/ycc/python/sp/ArNe_UVEX1200_23um')
    polypars, polypars_r = wlc.identify(res=0.302, wl0=7503.8690, Warc=10, podr=1, thershold=1., idvplt=False)	# 800
    wlc.plt_wlcal()
    wlc.refine_arc(podr=5, Warc=5, iteration=1)

  if arcwl == '750':
    arc1D = np.median(arr[1590:1610,:], axis=0)
    wlc = spu.wlcal(arc1D, FWHM=3., arclist_path='/home/ycc/python/sp/ArNe_UVEX1200_23um')
    polypars, polypars_r = wlc.identify(res=0.305, wl0=7503.8690, Warc=10, podr=2, thershold=1., idvplt=False)	# 750
    wlc.plt_wlcal()
    wlc.refine_arc(podr=5, Warc=5, iteration=1)

  if arcwl == '700':
    arc1D = np.median(arr[1590:1610,:], axis=0)
    wlc = spu.wlcal(arc1D, FWHM=3., arclist_path='/home/ycc/python/sp/ArNe_UVEX1200_23um')
    polypars, polypars_r = wlc.identify(res=0.307, wl0=6965.4310, Warc=10, podr=1, thershold=1., idvplt=False)	# 700
    wlc.plt_wlcal()
    wlc.refine_arc(podr=5, Warc=5, iteration=1)

  if arcwl == '650':
    arc1D = np.median(arr[1590:1610,:], axis=0)
    wlc = spu.wlcal(arc1D, FWHM=3., arclist_path='/home/ycc/python/sp/ArNe_UVEX1200_23um')
    polypars, polypars_r = wlc.identify(res=0.309, wl0=6402.2480, Warc=8, podr=2, thershold=1., idvplt=False)	# 650
    wlc.plt_wlcal()
    wlc.refine_arc(podr=5, Warc=5, iteration=1)

  if arcwl == '600':
    arc1D = np.median(arr[1590:1610,:], axis=0)
    wlc = spu.wlcal(arc1D, FWHM=3., arclist_path='/home/ycc/python/sp/ArNe_UVEX1200_23um')
    polypars, polypars_r = wlc.identify(res=0.311, wl0=5852.4878, Warc=8, podr=2, thershold=1., idvplt=False)	# 600
    wlc.plt_wlcal()
    wlc.refine_arc(podr=5, Warc=5, iteration=1)

  if arcwl == '550':
    arc1D = np.median(arr[1590:1610,:], axis=0)
    wlc = spu.wlcal(arc1D, FWHM=3., arclist_path='/home/ycc/python/sp/ArNe_UVEX1200_23um')
    polypars, polypars_r = wlc.identify(res=0.312, wl0=5852.4878, Warc=6, podr=2, thershold=1., idvplt=False)	# 550
    wlc.plt_wlcal()
    wlc.refine_arc(podr=5, Warc=5, iteration=1)

  if arcwl == '500':
    arc1D = np.median(arr[1590:1610,:], axis=0)
    wlc = spu.wlcal(arc1D, FWHM=3., arclist_path='/home/ycc/python/sp/ArNe_UVEX1200_23um')
    polypars, polypars_r = wlc.identify(res=0.313, wl0=4764.8644, Warc=10, podr=2, thershold=1., idvplt=False)	# 500
    wlc.plt_wlcal()
    wlc.refine_arc(podr=5, Warc=8, iteration=1)

  if arcwl == '450':
    arc1D = np.median(arr[1580:1620,:], axis=0)
    wlc = spu.wlcal(arc1D, FWHM=3., arclist_path='/home/ycc/python/sp/ArNe_UVEX1200_23um')
    polypars, polypars_r = wlc.identify(res=0.313, wl0=4200.6740, Warc=10, podr=2, thershold=1., idvplt=False)	# 450
    wlc.plt_wlcal()
    wlc.refine_arc(podr=5, Warc=8, iteration=1)

  if arcwl == '400':
    arc1D = np.sum(arr[1575:1625,:], axis=0)
    wlc = spu.wlcal(arc1D, FWHM=3., arclist_path='/home/ycc/python/sp/ArNe_UVEX1200_23um')
    polypars, polypars_r = wlc.identify(res=0.314, wl0=4200.6740, Warc=10, podr=2, thershold=1., idvplt=False)	# 400
    wlc.plt_wlcal()
    wlc.refine_arc(podr=5, Warc=8, iteration=1)

#  polypars, polypars_r = wlc.identify(res=0.290, wl0=9657.7860, Warc=10, podr=1, thershold=1., idvplt=False)	# 1000
  wlc.plt_wlcal(title=i, pltraw=True, rawarc=arr[1511:1689,:])
  wlx = wlc.Xwl
  print ('dL = %.3f' %((wlx[-1]-wlx[0])/len(arc1D)))

sys.exit()



dlist = {}
for i in glob('RAW/dark4arc/*'):
  arr, hdr = fits.getdata(i, header=True)
  expT = int(hdr['exptime'])
  if expT not in dlist.keys():
    dlist[expT] = []
  dlist[expT].append(arr)

for d in dlist.keys():
  d_median = np.median(dlist[d], axis=0)
  arr2fit(d_median, 'proc/D%03d.fit' %(d), arrfmt='int32')


arclist = {}
for i in glob('RAW/seq4_arctest/*'):
  g, wl = i.split('/')[-1].split('-')[0].split('_')[1:]
  arr, hdr = fits.getdata(i, header=True)
  expT = float(hdr['exptime'])
  md = fits.getdata('proc/D%03d.fit' %(expT))
  arr = np.array(arr, dtype='int32') - md
  if g+wl not in arclist.keys():
    arclist[g+wl] = []
  arclist[g+wl].append(arr)

for arc in arclist.keys():
  print (arc)
  arc_median = np.median(arclist[arc], axis=0)
  arr2fit(arc_median, 'proc/%s.fit' %(arc), arrfmt='int32')



