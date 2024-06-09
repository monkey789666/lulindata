import os, glob, random
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats, sigma_clip
import matplotlib.pyplot as plt
'''
imred = image_reduction.ImageReduction()

'''


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def random_index(sample, k=20):
    if k > len(sample):
        k = len(sample)
    return [i for i in random.sample(range(len(sample)),k)]

def img_show(data,size=15, vmin='', vmax='', mark='', wcs='', save=''):
    if type(data) == str:
        from astropy.io.fits import getdata
        data = getdata(data, header=False)
    
    plt.figure(figsize=(size,size))
    if wcs != '':
        plt.subplot(projection=wcs)
#         overlay = ax.get_coords_overlay('fk5')
        plt.grid(color='white', ls='solid')
    
    mean, median, std = sigma_clipped_stats(data[~np.isnan(data)])
    std_without_clip = np.std(data[~np.isnan(data)])
    if vmin == '':
        vmin = median - 1*std
    if vmax == '':
        vmax = median + 0.5*std_without_clip
    if mark != '':
        plt.scatter(mark[0], mark[1], c='r', marker='x', alpha=0.5, s=size*5)
    print(mean, median, std)
    print(f'vmin = {vmin}, vmax = {vmax}')
    plt.imshow(data, vmin=vmin , vmax=vmax, cmap='gray')
    if save != '':
        plt.savefig(save)
    else:
        plt.show()
    plt.close()

class ImageReduction:
    def __init__(self, curpath, sciFolder_exclude=[], biasdarkFolder='', flatFolder='', masterFolder='', procFolder='', ):
        self.curpath = curpath
        
        # bias-dark Dir
        if biasdarkFolder != '':
            self.biasdarkFolder = biasdarkFolder
        else:   
            self.biasdarkFolder = os.path.join(curpath, 'bias-dark')
        
        # flat Dir
        if flatFolder == 'No':
            self.flat = False
        else:
            if flatFolder != '':
                self.flatFolder = flatFolder
            else:
                self.flatFolder = os.path.join(curpath, 'flat')
            self.flat = os.path.isdir(self.flatFolder)
        
        # temp Dir
        tempath = '/home/wjhou/lulin_data/temp'
        
        if masterFolder == "":
            self.masterFolder = os.path.join(tempath, 'masterFrames')
        else:
            self.masterFolder = masterFolder
            sciFolder_exclude += [masterFolder.replace(curpath,'')]
        try:
            os.mkdir(self.masterFolder)
        except: 
            pass
            
        if procFolder == "":
            self.procFolder = os.path.join(tempath, 'processed')
        else:
            self.procFolder = procFolder
            sciFolder_exclude += [procFolder.replace(curpath,'')]
        try:
            os.mkdir(self.procFolder)
        except: 
            pass
        
        # sci Dirs list
        sciFolder_exclude += ['bias-dark', 'flat']
        self.sciFolders = []
        for folder in os.listdir(self.curpath):
            
            if folder not in sciFolder_exclude:
                self.sciFolders.append(os.path.join(self.curpath, folder))

    def dist_biasdark(self):
        BList = dict()
        DList = dict()
        for file in os.listdir(self.biasdarkFolder):
            
            if os.path.splitext(file)[-1] not in ['.fts','.fits','.fit']: continue
            fileDir = os.path.join(self.biasdarkFolder,file)
            data, header = fits.getdata(fileDir, header=True)

            # BIAS
            if header['IMAGETYP'] == 'BIAS' or header['IMAGETYP'] == 'Bias Frame':
                if f'bin{header["XBINNING"]}' not in BList.keys():
                    BList[f'bin{header["XBINNING"]}'] = []
                    BList[f'bin{header["XBINNING"]}'].append(fileDir)
                else:
                    BList[f'bin{header["XBINNING"]}'].append(fileDir)

            # DARK
            elif header['IMAGETYP'] == 'DARK'or header['IMAGETYP'] == 'Dark Frame':
                if f'bin{header["XBINNING"]}' not in DList.keys():
                    DList[f'bin{header["XBINNING"]}'] = dict()

                if header['EXPTIME'] not in DList[f'bin{header["XBINNING"]}'].keys():
                    DList[f'bin{header["XBINNING"]}'][header['EXPTIME']] = []
                    DList[f'bin{header["XBINNING"]}'][header['EXPTIME']].append(fileDir)
                else:
                    DList[f'bin{header["XBINNING"]}'][header['EXPTIME']].append(fileDir)
        self.biasList, self.darkList = BList, DList

    def dist_flat(self, nofilter=False):
        self.nofilter = nofilter
        if self.flat:
            if self.nofilter == True:
                FList = dict()
                for file in os.listdir(self.flatFolder):
                    fileDir = os.path.join(self.flatFolder,file)
                    data, header = fits.getdata(fileDir, header=True)
                    if header['IMAGETYP'] != 'FLAT': continue
                    if f'bin{header["XBINNING"]}' not in FList.keys():
                        FList[f'bin{header["XBINNING"]}'] = []
                    FList[f'bin{header["XBINNING"]}'].append(fileDir)
            else:
                FList = dict()
                for file in os.listdir(self.flatFolder):
                    if os.path.splitext(file)[-1] not in ['.fts','.fits','.fit']: continue
                    fileDir = os.path.join(self.flatFolder,file)
                    data, header = fits.getdata(fileDir, header=True)
                    if header['IMAGETYP'] != 'FLAT': continue

                    if f'bin{header["XBINNING"]}' not in FList.keys():
                        FList[f'bin{header["XBINNING"]}'] = dict()

                    if header['FILTER'] not in FList[f'bin{header["XBINNING"]}'].keys():
                        FList[f'bin{header["XBINNING"]}'][header['FILTER']] = []
                        FList[f'bin{header["XBINNING"]}'][header['FILTER']].append(fileDir)
                    else:
                        FList[f'bin{header["XBINNING"]}'][header['FILTER']].append(fileDir)
            self.flatList = FList
        
    def list_scifiles(self):
        SciFitsList = []
        for sciFolder in self.sciFolders:
            for dirPath, dirNames, fileNames in os.walk(sciFolder):
                for fileName in fileNames:
                    if os.path.splitext(fileName)[-1] not in ['.fts','.fits','.fit']: continue
                    filePath = os.path.join(dirPath, fileName)
                    data, header = fits.getdata(filePath, header=True)
                    if header['IMAGETYP'] != 'LIGHT': continue
                    SciFitsList.append(os.path.join(dirPath, fileName))
        self.sciList = SciFitsList
    
    def bias_combine(self, path=''):
        MBias = dict()
        for binning in self.biasList.keys():
            data, header = fits.getdata(self.biasList[binning][0], header=True)
            ny = header['NAXIS1']
            nx = header['NAXIS2']
            numBiasFiles = len(self.biasList[binning])
            if numBiasFiles > 10:
                numBiasFiles = 10
                self.biasList[binning] = np.array(self.biasList[binning])[random_index(self.biasList[binning], k=10)]
            biasImages = np.zeros((numBiasFiles, nx, ny))

            for i in range(numBiasFiles):
                biasImages[i,:,:] = fits.getdata(self.biasList[binning][i])

            MBias[binning] = np.median(biasImages, axis=0)
            if path != '':
                fits.writeto(os.path.join(path,f'Masterbias_{binning}.fits'), MBias[binning].astype('float32'), header, overwrite=True)
        return MBias


    def adj_dark(self, binning, expTime):
        if expTime in self.masterdarklist.keys():
            MD = self.masterdarklist[f'bin{binning}'][expTime]
        else:
            nearest_time = find_nearest([*self.masterdarklist[f'bin{binning}']], expTime)
            MD = self.masterdarklist[f'bin{binning}'][nearest_time]*expTime/nearest_time
        return MD
    
    def dark_combine(self, binning, exp, path=''):
        data, header = fits.getdata(self.darkList[binning][exp][0],header=True)
        ny = header['NAXIS1']
        nx = header['NAXIS2']
        numFiles = len(self.darkList[binning][exp])
        if numFiles > 10:
            numFiles = 10
            self.darkList[binning][exp] = np.array(self.darkList[binning][exp])[random_index(self.darkList[binning][exp], k=10)]
        darkImages = np.zeros((numFiles, nx, ny))

        for i in range(numFiles):
            data = fits.getdata(self.darkList[binning][exp][i])
            darkImages[i,:,:] = data - self.masterbiaslist[binning]
            darkImages[i,:,:][darkImages[i,:,:]<0] = 0

        MDark = np.median(darkImages, axis=0)
        if path != '':
            fits.writeto(os.path.join(path,f'Masterdark_{binning}_{exp}s.fits'), MDark.astype('float32'), header, overwrite=True)
        return MDark
    
    def flat_combine(self, binning, filter_, path=''):
        if not self.nofilter:
            data, header = fits.getdata(self.flatList[binning][filter_][0], header=True)
        else:
            data, header = fits.getdata(self.flatList[binning][0], header=True)
        ny = header['NAXIS1']
        nx = header['NAXIS2']
        binning = header['XBINNING']
        exptime = header['EXPTIME']
        MDark = self.adj_dark(binning, exptime)
        if not self.nofilter:
            try:
                filter_ = header['FILTER']
            except:
                pass
            numFiles = len(self.flatList[f'bin{binning}'][filter_])
            if numFiles > 10:
                numFiles = 10
                self.flatList[f'bin{binning}'][filter_] = np.array(self.flatList[f'bin{binning}'][filter_])[random_index(self.flatList[f'bin{binning}'][filter_], k=10)]
            flatImages = np.zeros((numFiles, nx, ny))

            for i in range(numFiles):
                flatImages[i,:,:] = fits.getdata(self.flatList[f'bin{binning}'][filter_][i])
                flatImages[i,:,:] -= (self.masterbiaslist[f'bin{binning}'] + MDark)
                mean, median, stddev = sigma_clipped_stats(flatImages[i,:,:][~np.isnan(flatImages[i,:,:])],sigma_upper=2)
                flatImages[i,:,:] /= mean
                mean, median, stddev = sigma_clipped_stats(flatImages[i,:,:][~np.isnan(flatImages[i,:,:])],sigma_upper=2)
                flatImages[i,:,:] /= mean
                flatImages[i,:,:][flatImages[i,:,:]<=0]=99999
            MFlat = np.median(flatImages, axis=0)
            if path != '':
                fits.writeto(os.path.join(path,f'Masterflat_bin{binning}_{filter_}.fits'), MFlat.astype('float32'), header, overwrite=True)
            return MFlat
        else:
            numFiles = len(self.flatList[f'bin{binning}'])
            if numFiles > 10:
                numFiles = 10
                self.flatList[f'bin{binning}'] = np.array(self.flatList[f'bin{binning}'])[random_index(self.flatList[f'bin{binning}'], k=numFiles)]
            flatImages = np.zeros((numFiles, nx, ny))
            for i in range(numFiles):
                flatImages[i,:,:] = fits.getdata(self.flatList[f'bin{binning}'][i])
                flatImages[i,:,:] -= (self.masterbiaslist[f'bin{binning}'] + MDark)
                normfactor = np.median(sigma_clip(flatImages[i,:,:][~np.isnan(flatImages[i,:,:])], masked=False))
                flatImages[i,:,:] /= normfactor
                flatImages[i,:,:][flatImages[i,:,:]<=0]=99999
            MFlat = np.median(flatImages, axis=0)
            if path != '':
                fits.writeto(os.path.join(path,f'Masterflat_bin{binning}.fits'), MFlat.astype('float32'), header, overwrite=True)
            return MFlat
    def proc_calibimg(self):
        # bias
        self.masterbiaslist = self.bias_combine(self.masterFolder)
    
        # dark
        self.masterdarklist = dict()
        for binning in self.darkList.keys():
            self.masterdarklist[binning] = dict()
            for exp in self.darkList[binning].keys():
                self.masterdarklist[binning][exp] = self.dark_combine(binning, exp, self.masterFolder)
        
        # flat
        if self.flat:
            if not self.nofilter:
                self.masterflatlist = dict()
                for binning in self.flatList.keys():
                    self.masterflatlist[binning] = dict()
                    for filter_ in self.flatList[binning].keys():
                        self.masterflatlist[binning][filter_] = self.flat_combine(binning, filter_, self.masterFolder)
            else:
                self.masterflatlist = dict()
                for binning in self.flatList.keys():
                    self.masterflatlist[binning] = self.flat_combine(binning, self.masterFolder)
                
    def proc_sciimg(self, images_number=100):
        
        if images_number == 'all':
            selected_list = self.sciList
        else:
            selected_list = np.array(self.sciList)[random_index(self.sciList,images_number)]
        for sciFile in selected_list:
            data, header = fits.getdata(sciFile, header=True)
            exptime = header['EXPTIME']
            binning = header["XBINNING"]
            
            masterbias = self.masterbiaslist[f'bin{binning}']
            masterdark = self.adj_dark(binning, exptime)

            if self.flat:
                if not self.nofilter:
                    filter_ = header['FILTER']
                    masterflat = self.masterflatlist[f'bin{binning}'][filter_]
                else:
                    masterflat = self.masterflatlist[f'bin{binning}']
            else:
                masterflat = 1.
            data = (data - masterbias - masterdark)/masterflat

            fits.writeto(os.path.join(self.procFolder,os.path.splitext(os.path.basename(sciFile))[0]+'_proc.fits')
                         , data.astype('float32'), header, overwrite=True)
            
    def clear_temp(self):
        os.system(f'rm -r {self.masterFolder}/*')
        os.system(f'rm -r {self.procFolder}/*')