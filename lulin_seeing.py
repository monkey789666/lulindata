import sys, os, glob, getopt, traceback
from multiprocessing import Process, Pool
import image_reduction, FWHM

ftp_path = '/export/AutoFTP2/'


def usage():
    print(f'''\nlulin_seeing Usage:
    
    {sys.argv[0]} [-d YYYYMMDD] [-t telescope,] [--exclude list_file]
    
        --bd_path=path                  Absolute path of bias_dark folder
        --f_path=path                   Absolute path of flat folder
        --noflat                        Do not calbriate image by flat field (or --f_path=no)
    -d  --date=yyyymmdd                 Date (yyyymmdd)
    -e, --exclude=path                  Absolute path of text file of exclude list
        --exclude_folder=str (or list)  Folders which should be ignored in target folder 
    -h, --help                          Help
    -n, --n_sample=all(or int)          Number of samples (all or int)
    -p, --path=path                     Path of observation dir
    -t  --telescope=str(or list)        Telescope (telescopeA,telescopeB,... if mutiple objects)
    ''')

def log(e):
    e+='\n'
    with open('/home/wjhou/lulin_data/seeing/log.txt', 'a') as f:
        f.write(e)

def main(date, teles_list, exclude_file='', sciFolder_exclude='', dir='',bd_path='',f_path='', images_number=100):
    print(exclude_file)
    folders = []
    
    if dir=='':
        dir = ftp_path

    if len(glob.glob(f'{dir}*{date}')) == 0:
        log(f'No such file or directory \"{ftp_path}*{date}\"')
        sys.exit()
    
    if sciFolder_exclude != '':
        sciFolder_exclude = sciFolder_exclude.split(',')
    else:
        sciFolder_exclude = []
    
    for i in glob.glob(f'{dir}*{date}'):
        for selected_telescope in teles_list:
            if selected_telescope in i:
                folders.append(i)
    print(folders)
    log(f"[{date}] {folders}")
    
    for telescope in folders:
        try:
            print(f"now proccessing {telescope}")
            log(f"{telescope}")
            imred = image_reduction.ImageReduction(telescope, sciFolder_exclude,biasdarkFolder=bd_path,flatFolder=f_path)
            imred.clear_temp()
            imred.dist_biasdark()
            imred.dist_flat()
            imred.list_scifiles()

            if exclude_file!='':
                with open(exclude_file, 'r') as f:
                    exclude_list = f.read()
                    exclude_list = exclude_list.split('\n')
                imred.sciList = [i for i in imred.sciList if i not in exclude_list]
            imred.proc_calibimg()
            imred.proc_sciimg(images_number=images_number)

            selected_files = glob.glob(imred.procFolder+'/*')
            pool = Pool(3)
            output = pool.map(FWHM.measure_FWHM, selected_files)
            pool.close()
            pool.join()

            seeing_dir = '/home/wjhou/lulin_data/seeing'
            json_name = f'{os.path.basename(imred.curpath)}.json'
            fc = FWHM.FWHMCurve(seeing_dir, json_name)
            fc.write_to_json(output)
            del imred, fc
            log(f'\t{telescope} successed.')
        except Exception as e:
            error_class = e.__class__.__name__ #取得錯誤類型
            detail = e.args[0] #取得詳細內容
            cl, exc, tb = sys.exc_info() #取得Call Stack
            lastCallStack = traceback.extract_tb(tb)[-1] #取得Call Stack的最後一筆資料
            fileName = lastCallStack[0] #取得發生的檔案名稱
            lineNum = lastCallStack[1] #取得發生的行號
            funcName = lastCallStack[2] #取得發生的函數名稱
            errMsg = "File \"{}\", line {} in {}: [{}] {}".format(fileName, lineNum, funcName, error_class, detail)
            print(traceback.format_exc())
            log(str(traceback.format_exc()))

            
if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(
            sys.argv[1:], 
            "hd:t:e:p:n:", 
            ['date=', 'telescope=','exclude=','exclude_folder=','path=','bd_path=','f_path=','noflat','n_sample='])
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err)  # will print something like "option -a not recognized"
        usage()
        sys.exit(2)
        
    from datetime import datetime
    command = ''.join(str(s)+' ' for s in sys.argv)
    log(f'[{datetime.strftime(datetime.now(),"%Y-%m-%d %H:%M:%S")}] {command}')
        
    from datetime import date, timedelta
    yesterday = date.today() - timedelta(days=1)
    d = yesterday.strftime('%Y%m%d')
    telescopes=['LOT', 'slt', 'LWT_']
    exclude = ''
    dir = ''
    bd_path = ''
    f_path = ''
    sciFolder_exclude = ''
    images_number = 100
    for name, value in opts:
        if name in ('-h'):
            usage()
            sys.exit()
        if name in ('-d', '--date'):
            d = value 
        if name in ('-t', '--telescope'):
            telescopes = value.split(',')
        if name in ('-e', '--exclude'):
            exclude = value
        if name in ('--exclude_folder'):
            sciFolder_exclude = value
        if name in ('-p', '--path'):
            dir = value
        if name in ('--bd_path'):
            bd_path = value
            print(f'bias-dark: {bd_path}')
        if name in ('--f_path'):
            f_path = value
            print(f'flat: {f_path}')
        if name in ('--noflat'):
            f_path = 'no'
        if name in ('-n', '--n_sample'):
            if value != 'all':
                value = int(value) 
            images_number = value
    main(d, telescopes, exclude, sciFolder_exclude, dir, bd_path, f_path, images_number)
    