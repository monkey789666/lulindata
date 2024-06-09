from astropy.io import fits
import trail_fit_univ as tfu
from photutils.detection import DAOStarFinder
from astropy.stats import sigma_clipped_stats, sigma_clip
from image_reduction import random_index

from glob import glob
import os, json
import matplotlib.pyplot as plt
import numpy as np

def pixel_scale(telescope, binning):
    ps_list = {'LOT' : 0.344, 'SLT' : 0.76, 'LWT' : 1.218, 'ACP->Driver for telescope connected through TheSky' : 1.218,}
    #'SLT' : 0.705(andor 936), 'SLT' : 0.529(U9000)
    return ps_list[telescope] * binning

def brightest_sources(sources, number):
    sources.sort('flux')
    return sources[-number:]

def measure_FWHM(file_path, boxsize = 20, star_number=10, plot=False):
    fitting_model = 'M'
    
    imgarr, imghdr = fits.getdata(file_path , header=True)
    imgtype = imghdr['IMAGETYP']
    if imgtype != 'LIGHT': return
    telescope = imghdr['TELESCOP']
    datetime_obs = f"{imghdr['DATE-OBS'][:10]} {imghdr['TIME-OBS']}"
    jd = float(imghdr['JD'])
    binning = imghdr['XBINNING']
    filter_obs = imghdr['FILTER']
    
    data = {
        'datetime': datetime_obs,
        'jd': jd,
        'filter': filter_obs,
        'file_path': file_path.split('/')[-1].replace('_proc',''),
    }
    file = file_path.split('/')[-1].replace('_proc','').replace('.fits','')
    
    try:
        airmass = imghdr['AIRMASS']
        data['airmass'] = airmass
    except:
        data['airmass'] = 'NaN'
    
    pscale = pixel_scale(telescope, binning)
    mean, median, std = sigma_clipped_stats(imgarr[~np.isnan(imgarr)])
    daofind = DAOStarFinder(fwhm=3, sharphi=0.7 ,threshold=5.*std)  
    sources = daofind(imgarr - median)
    if sources is not None:
        #remove bad pixel
        if telescope == 'LOT':
            sources = sources[~(((sources['xcentroid']-1293)**2 < 25) & ((sources['ycentroid']-1143)**2 < 25))]
        # if telescope == 'SLT':
        #     sources = sources[(sources['xcentroid']-2782)**2 > 25]#
        sources_in_range = sources[(sources['peak'] > 3000) & (sources['peak'] < 60000) & (sources['xcentroid'] > boxsize*1.5) & (sources['xcentroid'] < int(imghdr['NAXIS1']) - boxsize*1.5) & (sources['ycentroid'] > boxsize*1.5) & (sources['ycentroid'] < int(imghdr['NAXIS1']) - boxsize*1.5)]
#         return sources_in_range
        
        if len(sources_in_range['xcentroid']) >= star_number:
            FWHMs = []
#             fitting_star = sources_in_range[random_index(sources_in_range, star_number)]
            fitting_star = brightest_sources(sources_in_range, star_number)
            for star in fitting_star:
                TF = tfu.Trail_Fit(imgarr, (star['xcentroid'],star['ycentroid']) , pscale=pscale)
                tf_par = TF.fit_trail(fitting_model, 'DE')

                if plot == True:
                    TF.mkplot(
                        title_str=f"{file}_({star['xcentroid']},{star['ycentroid']})",
                        output_basename=f"/home/wjhou/lulin_data/plot/{file}_x{int(star['xcentroid'])}_y{int(star['ycentroid'])}.jpg",
                        ap=False, 
                        mp=False)
                if TF.FWHM()>0.5:
                    FWHMs.append(TF.FWHM())
#             FWHMs = sigma_clip(FWHMs)
            if len(FWHMs) > 5:
                meanFWHM = np.mean(FWHMs)
                medianFWHM = np.median(FWHMs)
                smeFWHM = np.std(FWHMs)/(len(FWHMs)**0.5)
                data['fwhms'] = FWHMs
                data['mean_fwhm'] = meanFWHM
                data['sme_fwhm'] = smeFWHM
                data['median_fwhm'] = medianFWHM
                data['fitting_model'] = fitting_model
                return data
    

class FWHMCurve:
    def __init__(self, seeing_dir, json_name):
        self.json_name = json_name
        self.path = os.path.join(seeing_dir, json_name)
        
    def write_to_json(self, output):
        output = [i for i in output if i is not None]
        data = {
            'datetime' : [i['datetime'] for i in output],
            'jd' : [i['jd'] for i in output],
            'filter': [i['filter'] for i in output],
            'fwhms': [i['fwhms'] for i in output],
            'mean_fwhm': [i['mean_fwhm'] for i in output],
            'median_fwhm':[i['median_fwhm'] for i in output],
            'sme_fwhm': [i['sme_fwhm'] for i in output],
            'file_path': [i['file_path'] for i in output],
            'airmass': [i['airmass'] for i in output],
            'fitting_model': output[0]['fitting_model']
        }

        import json
        data_json = json.dumps(data)
        with open(self.path, 'w') as f:
            f.write(data_json)
        
    def get_data(self):
        with open(self.path, 'r') as f:
            data = json.loads(f.read())
    
        self.jd = np.array(data['jd'])
        self.utc = ((np.array(self.jd)%1) * 24) + 12
        self.filters = np.array(data['filter'])
        self.mean_fwhm = np.array(data['mean_fwhm'])
        self.sme_fwhm = np.array(data['sme_fwhm'])
        self.file_path = np.array(data['file_path'])
        self.airmass = np.array(data['airmass'])
        
        new_data = {'data': []}
        for i in range(len(data['datetime'])):
            obj_fwhm = {}
            for key in data.keys():
                obj_fwhm[f'{key}'] = data[key][i]
            new_data['data'].append(obj_fwhm)
        return new_data

        
        
    def plot(self, path='', filter_label=True, figsize=(16,4)):
        plt.figure(figsize=figsize)
        if filter_label == True:
            for fil in np.unique(self.filters):
#                 plt.plot(self.utc[self.filters == fil], self.mean_fwhm[self.filters == fil], '.', label=fil)
                plt.errorbar(self.utc[self.filters == fil], self.mean_fwhm[self.filters == fil], yerr=self.sme_fwhm[self.filters == fil], fmt='.', elinewidth=1.5, label=fil)
            plt.legend()
        if filter_label == False:
            plt.plot(self.utc, self.mean_fwhm, '.')
        plt.title(f'{self.json_name.replace(".json","")}\nmean fwhm = {np.mean(self.mean_fwhm):.2f}\nmin fwhm = {np.min(self.mean_fwhm):.1f}')
        plt.xlabel('TIME OBS (UT)')
        plt.ylabel('FWHM (arcsec)')
        plt.xlim(10,22)
#         plt.ylim(0,4)
        
        if path != '':
            plt.savefig(path)
        plt.show()
        plt.close()