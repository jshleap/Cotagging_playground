'''
D uncertainty
'''
import os, re, argparse, itertools
from subprocess import Popen
from glob import glob as G
from time import sleep
import pandas as pd
import numpy as np
import pickle as P
try:
    import matplotlib
    matplotlib.style.use('ggplot')
    import matplotlib.pyplot as plt
except:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')

## Constants
qsub='''#!/bin/bash
#PBS -l nodes=1:ppn=%d,walltime=1:00:00
#PBS -A ams-754-aa
#PBS -o %s.out
#PBS -e %s.err
#PBS -N %s
    
module load foss/2015b 
module load Python/3.5.2

cd  $PBS_O_WORKDIR

%s
'''   

#----------------------------------------------------------------------
def PlinkLD(pref, bedpref, keep, plinkexe='/home/jshleap/bins', windowkb=10000,
            hwe=True, maf=0.01, chrom='22', parallel=True, sel=None, 
            verbose=True):
    """
    Compute plink LD matrix
    :param str sel: file with the selected snps to compute LD from
    """
    if verbose: print('Generating LD')
    comm = '%s --bfile %s --keep %s --chr %s '
    if sel:
        comm += '--extract %s '%(sel)
    comm += '--keep-allele-order --maf %.2f --out %s --r d with-freqs '
    #comm += '--ld-window-r2 0 ' #cannot be used with --r
    if hwe:
        comm += '--hwe 0.0001 midp include-nonctrl'
    comm = comm%(plinkexe, bedpref, keep, chrom, maf, pref)
    if parallel:
        return comm
    else:
        if not os.path.isfile('%s.ld'%(pref)):
            a = Popen(comm, shell=True)
            a.communicate()
            if verbose: print('Command %s launched'%comm)
        else:
            if verbose:
                print('File %s.ld exists, using it!'%(pref))

#----------------------------------------------------------------------
def readBIM(bimfile):
    """
    read a bim file
    """
    header=['Chromosome', 'SNP', 'Position', 'BP',  'Allele 1', 'Allele 2']
    bim = pd.read_table(bimfile, delim_whitespace=True, header=None,
                        names=header)
    return bim
        
      
#----------------------------------------------------------------------
def read_LD(fn, verbose=True):
    '''
    read the LD file as outputted by plink
    '''
    if verbose:
        print('Reading file %s'%(fn))
    df = pd.read_table(fn, delim_whitespace=True)
    ## Drop Nans
    df = df.dropna()
    ## Drop MAFs
    df = df[(df.MAF_A > 0.01) & (df.MAF_A < 0.99) & 
            (df.MAF_B > 0.01) & (df.MAF_B < 0.99)]
    ## compute the distance
    df.loc[:,'Distance (Bp)'] = abs(df.BP_A - df.BP_B)
    if verbose: print('\tDONE\n')
    return df    

#----------------------------------------------------------------------
def readSamples(keep):
    """ Read the samples file """
    return open(keep).read().strip().split('\n')

#----------------------------------------------------------------------
def JacknifeSamples(keep):
    """
    Given a list of samples (as read readSamples), create a generator that 
    returns n-1
    """
    #n = len(keep)
    for k in range(len(keep)):
        yield keep[:k] + keep[k+1:]

#----------------------------------------------------------------------
def LD4JACK(pref, bedpref, keep, plinkexe='/home/jshleap/bins', windowkb=10000,
            hwe=True, maf=0.1, chrom='22', parallel=True, sel=1000, 
            verbose=True):
    """
    Compute the LDs in each jacknife in parallel
    """
    if sel:
        selSNPs = readBIM(bedpref + '.bim').SNP
        np.savetxt('snps2extract.txt', np.random.choice(selSNPs, sel), fmt='%s')
        sel = 'snps2extract.txt'
            
    files = []
    jack = JacknifeSamples(readSamples(keep))
    for i, n in enumerate(itertools.chain([keep], jack)):
        fn = 'temp%d.keep'%(i)
        if not os.path.isfile(fn):
            with(open(fn,'w')) as F:
                F.write('\n'.join(n))        
        temp = '%s_%d'%(pref, i)
        if not parallel:
            com = PlinkLD(temp, bedpref, fn, plinkexe=plinkexe, maf=maf, 
                          hwe=hwe, windowkb=windowkb, chrom=chrom, 
                          parallel=False, sel=sel)            
        else:
            com = PlinkLD(temp, bedpref, fn, plinkexe=plinkexe, maf=maf,
                          windowkb=windowkb, hwe=hwe, chrom=chrom, 
                          sel=sel)
            if not os.path.isfile('%s.ld'%(emp)):
                if verbose: print('Sending %s to queue'%com)
                qsub = qsub%(16, temp, temp, temp, com)
                Qsub = Popen(qsub, shell=True)
                Qsub.communicate()
        files.append(temp)
    return files

#----------------------------------------------------------------------
def plotEstimation(D, meanjack, low, high, prefix):
    """
    plot the estimated D vs the jackknife mean with a 95% CI
    """
    fig, ax = plt.subplots()
    ax.fill_between(D, high, low, color='gray', alpha=0.5)
    ax.plot(D, meanjack, 'ko')#, label='Observed Values')
    #ax.plot(x, y_pred, 'k--', label='Predicted Model')
    #ax.plot(x, y_true, 'r-', label='True Mt odel')
    #ax.legend(loc='upper left')
    plt.xlabel('Estimated D')
    plt.ylabel('Mean Jackkifed D')
    plt.savefig('%s_jackknife.pdf'%(prefix))    
    
#----------------------------------------------------------------------
def ProcessOutput(prefix, expectedfiles, parallel=True, verbose=True):
    """
    wait for outputs and process them
    """
    if verbose: print('Processing LDs')
    processed = []
    R = []
    Rappend = R.append
    D = []
    Dappend = D.append
    if parallel:
        while len(set(expectedfiles).difference(processed)) != 0:
            files = G('*.ld')
            if not files: sleep(10)
            actual = [f for f in files if f not in processed]
            for fn in actual:
                pref = fn[:fn.find('.ld')]
                iappend(int(re.findall('\d+', pref)[0]))
                #data[(idx, pref)] = read_LD(fn)
                data = read_LD(fn)
                Dappend(data.D)
                Rappend(data.R)
                processed.append(pref)
                #os.remove(fn)
    #sorted_keys = sorted(data.keys(), key=lambda v: v[0])
    #full = sorted_keys.pop(0)
    
        with open('%s_DR.pickle'%(pref), 'wb') as DR:
            P.dump((D, R), DR)
        pyline  = "import os, re, argparse, itertools;"
        pyline += "from subprocess import Popen;from glob import glob as G;"
        pyline += "from time import sleep;import pandas as pd;"
        pyline += "import numpy as np;import pickle as P;"
        pyline += "picklefile = '%s_DR.pickle'"%(pref)
        pyline += "with open(picklefile, 'rb') as RDR: D, R = P.load(RDR);"
        pyline += "oriD = D.pop(0); D = pd.concat(D, axis=1);n = D.shape[1];"
        pyline += "thetadot = D.apply(np.mean,axis=1);"
        pyline += "l2tract = [thetadot - D.loc[:,col] for col in D.columns];"
        pyline += "iminusdot = pd.concat(l2tract, axis=1);" 
        pyline += "Var_j = np.divide((n-1),n) * (iminusdot**2).sum();"
        pyline += "Bias = (n-1) * (thetadot - oriD);" 
        pyline += "D_corrected = (n * D) - ((n-1) * thetadot);"
        pyline += "na = '%s_Djack.pickle';"%(pref)
        pyline += "obj = (D, thetadot, Var_j, Bias, D_corrected)"
        pyline += "with open(na,'wb') as DF: P.dump(obj,DF);del D, D_corrected;"
        pyline += "report = 'Jackknife-estimated D uncertainty description: ';"
        pyline += "report += str(Var_j.describe());"
        pyline += "report += 'Jackknife D bias: ' + str(Bias.describe());"
        pyline += "oriR = R.pop(0); R =pd.concat(R, axis=1); n = R.shape[1];"
        pyline += "thetadot = R.apply(np.mean, axis=1);"
        pyline += "l2tract = [thetadot - R.loc[:,col] for col in R.columns]"
        pyline += "iminusdot = pd.concat(l2tract, axis=1);"
        pyline += "Var_j = np.divide((n-1),n)* (iminusdot**2).sum();"
        pyline += "Bias = (n-1) * (thetadot - oriR);"
        pyline += "R_corrected = (n * R) - ((n-1) * thetadot);"
        pyline += "obj = (R, thetadot, Var_j, Bias, R_corrected)"
        pyline += "nam = '%s_Rjack.pickle'"%(pref)
        pyline += "with open(nam, 'wb') as RF: P.dump(obj, RF);"
        pyline += "del R, R_corrected, iminusdot, obj, l2tract;"
        pyline += "report += 'Jackknife-estimated uncertainty of R descrition:'"
        pyline += "+str(Var_j.describe());report += 'Jackknife bias of R"
        pyline += " description:\n' + str(Bias.describe());"
        pyline += "print('Uncertainty report:\n, report);"
        comm = 'python -c "%s"'%(pyline)
        name = '%s_processed'
        a = Popen(qsub%(1, name, name, name, comm), shell=True)
        a.communicate()        
        
    else:
        for fn in G('*.ld'):
            pref = fn[:fn.find('.ld')]
            name = 'jack%s'%(re.findall('\d+', pref)[0])
            #data[(idx, pref)] = read_LD(fn)
            data = read_LD(fn)
            data.D.name = name
            data.R.name = name
            Dappend(data.D)
            Rappend(data.R)
            processed.append(pref)            
        ##Process D
        oriD = D.pop(0)
        D = pd.concat(D, axis=1)#.dropna()#.rename(columns={x:'n-%d'%(x) for x in indices})
        mean = D.apply(np.mean,axis=1)
        n = D.shape[1]
        PSi = pd.concat([(n * oriD) - ((n-1) * D.loc[:,col]) 
                         for col in D.columns], axis=1)
        PS = PSi.apply(np.mean,axis=1) 
        Vps = ((PSi - PS)**2).sum(axis=1) * (1 / (n - 1))
        e = (1.960) * np.sqrt(Vps/n)
        low, high = (PS - e), (PS + e)
        plotEstimation(oriD, mean, low, high, '%s_D'%(prefix))
        #thetadot = D.apply(np.mean,axis=1)
        #iminusdot = pd.concat([thetadot - D.loc[:,col] for col in D.columns], 
        #                      axis=1)
        #Var_j = ((n-1)/n) * (iminusdot**2).sum()
        #Bias = (n-1) * (thetadot - oriD)
        #D_corrected = (n * oriD) - ((n-1) * thetadot)
        with open('%s_Djack.pickle'%prefix, 'wb') as DF:
            P.dump((D, PSi, PS, Vps, low, high), DF)
        #del D, D_corrected
        #report = 'Jackknife-estimated uncertainty of D descrition:\n %s \n'%(
            #Var_j.describe())
        #report += 'Jackknife bias of D description: %s\n'%(Bias.describe())
        ##Process R
        oriR = R.pop(0)
        R = pd.concat(R, axis=1)
        mean = R.apply(np.mean,axis=1)
        PSi = pd.concat([(n * oriR) - ((n-1) * R.loc[:,col]) 
                         for col in R.columns], axis=1)
        PS = PSi.apply(np.mean,axis=1) 
        Vps = ((PSi - PS)**2).sum(axis=1) * (1 / (n - 1))
        e = (1.960) * np.sqrt(Vps/n)
        low, high = (PS - e), (PS + e)
        plotEstimation(oriR, mean, low, high, '%s_R'%(prefix))        

        ##.dropna()#.rename(columns={x:'n-%d'%(x) for x in indices})
        #n = R.shape[1]
        #thetadot = R.apply(np.mean,axis=1)
        #iminusdot = pd.concat([thetadot - R.loc[:,col] for col in R.columns], 
                                      #axis=1)        
        #Var_j = ((n-1)/n) * (iminusdot**2).sum()
        #Bias = (n-1) * (thetadot - oriR)
        #R_corrected = (n * oriR) - ((n-1) * thetadot)
        with open('%s_Rjack.pickle'%(prefix), 'wb') as RF:
            P.dump((R,  PSi, PS, Vps, low, high), RF)
        #del R, R_corrected
        #report += 'Jackknife-estimated uncertainty of R descrition:\n %s'%(
            #Var_j.describe())
        #report+= 'Jackknife bias of R description: %s'%(Bias.describe())        
        if verbose: print('Processing done...\n')
        #print('Uncertainty report:\n%s'%(report))

#----------------------------------------------------------------------
def execute(args):
    """Execute the code with a given arguments. Args is an argsparse instance"""
    print('Options chosen:')
    for k, v in args.__dict__.items():
        print('\t', k, v, type(v))
    expected_out = LD4JACK(args.prefix, args.bedprefix, args.keep, 
                           plinkexe=args.plinkexe, windowkb=args.windowkb,
                           hwe=args.hwe, maf=args.maf, chrom=args.chromosomes,
                           parallel=args.parallel, sel=args.selection)
    ProcessOutput(args.prefix, expected_out, parallel=args.parallel)
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prefix', help='prefix for outputs',
                        required=True)    
    parser.add_argument('-b', '--bedprefix', help='prefix of bed fileset',
                        required=True)     
    parser.add_argument('-k', '--keep', help='file with individual codes',
                        required=True)     
    parser.add_argument('-P', '--plinkexe', help='path to plink executable',
                        default='/home/jshleap/bins')
    parser.add_argument('-w', '--windowkb', help='size of the window for LD',
                        default=1000, type=int)
    parser.add_argument('-d', '--hwe', help='Do not perform hwe filtering',
                        default=True, action='store_false')    
    parser.add_argument('-m', '--maf', help='MAF threshold for filtering',
                        default=0.01, type=float)    
    parser.add_argument('-c', '--chromosomes', help='Chromosomes to analyse \
    the format is the same as in plink, e.g 1-22 analyze all the range',
                                               default='22')    
    parser.add_argument('-l', '--parallel', help='Not Guillimin', default=True, 
                        action='store_false')
    parser.add_argument('-s', '--selection', help='Number of snps to select', 
                        default=1000, type=int)    
    args = parser.parse_args()
    execute(args)