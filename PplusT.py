#!/usr/bin/env python
#coding:utf-8
"""
  Author:  Jose Sergio Hleap --
  Purpose: Implement the P + T method for unambigous data. This works better
  for phased or simulation data, otherwise use PRSice
  Created: 06/22/17
"""

## Libraries ###################################################################
#import unittest
import pickle
import tarfile
import numpy as np
import pandas as pd
from time import sleep
from tqdm import tqdm
from scipy import stats
from glob import glob
from patsy import dmatrices
import argparse, os, shutil
import statsmodels.api as sm
from subprocess import Popen, PIPE
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from matplotlib.offsetbox import AnchoredText
################################################################################
    
## Functions ###################################################################
#----------------------------------------------------------------------
def qstat(debug=False):
    """
    Returns the qstat output for a given job
    """
    run = Popen('qstat | grep `whoami`', shell=True, stderr=PIPE, stdout=PIPE)
    o,e = run.communicate()  
    if debug:
        print('@@qstat %s, %s'%(o, e))
    return o

#----------------------------------------------------------------------
def allDONE(jobs, verbose=True, debug=False):
    """ 
    Check if file have been done in the queue 
    :param list jobs: list of jobs committed
    """
    qcols = ['JobID', 'Name', 'User', 'Time', 'Use', 'status', 'Queue']
    if debug:
        print('@@JOBS BEING PROCESSED'), jobs
    if (jobs is None):
        warnings.warn('JOBs are empty!!!')
        return True, []
    jobs = [j for j in jobs if j is not None]
    if verbose: print('checking alldone')
    qst = qstat(debug=debug)
    r = pd.read_table(StringIO(unicode(qst)), delim_whitespace=True,
                      header=None, names=qcols)
    if debug:
        print('@@r DataFrame\n', r)
    #if debug: r.to_pickle('qstat.dataframe'
    ## all this shit to make it not perun dependant
    idx = [any([str(x) in j for j in jobs]) for x in r.jobID]
    r = r[idx]
    running = r[(r.state == 'r') | (r.state == 'R')].jobID
    queue = r[(r.state == 'q') | (r.state == 'Q') | (r.state == 'qw') | 
              (r.state == 'Qw') | (r.state == 'QW')].jobID
    if debug:
        print('@@r', r)
        print('@@running', running)
        print('@@queue', queue)
    if not running.empty:
        if debug:
            print('@@running is not empty')
        return False, r
    elif not queue.empty:
        if debug:
            print('@@queue is not empty')         
        return False, r
    else:
        if debug:
            print('@@Both running and queue are empty')            
            return True, r 
    
#----------------------------------------------------------------------
def execute(line):
    """
    Execute a given line using Popen.
    :param str line: Line to be executed
    :returns: stderr and stdout
    """
    exe = Popen(line, shell=True, stderr=PIPE, stdout=PIPE)
    o, e = exe.communicate()
    return o.strip(), e.strip() 


################################################################################
class PplusT:
    """
    Class to implement a p+t approach when the input does not have ambiguities
    (e.g simulations)
    """

    #----------------------------------------------------------------------
    def __init__(self, bfile, plinkexe, outprefix, window, sumstats, pheno,
                 rstart=0.1, rstop=0.99, rstep=0.1, pstart=1E-8, pstop=1, 
                 customPrange=None, clean=False, validate=False):
        """
        Constructor
        
        :param str bfile: prefix for genotype from which to compure r2 (plink)
        :param str plinkexe: path to plink executable
        :param str outprefix: Prefix for output files
        :param int window: Physical distance threshold for clumping
        :param str sumstats: File name of the sumary statistics
        :param str pheno: File name of the phenotype file
        :param float rstart: lower bound for the range of R2
        :param float rstop: upper bound for the range of R2
        :param float rstep: step for the range of R2
        :param float pstart: smallest pvalue threshold
        :param float pstop: largest pvalue threshold
        :param str customPrange: string with comma separated floats determining
        the range for pvalues
        :param bool clean: clean up the files
        :param bool validate: use train and test subsets for parameter tunning
        """
        self.validate = validate
        self.clean = clean
        self.bfile = bfile
        self.sumstats = sumstats
        self.sumstatsdf = pd.read_table(sumstats, delim_whitespace=True)
        self.plinkexe = plinkexe
        self.outpref = outprefix
        self.window = window
        self.phenofn = pheno
        self.pheno = self.read_pheno(pheno)
        #self.bim = self.read_Bim()
        self.results = pd.DataFrame(columns=['File', 'LDthresh', 'Pthresh', 
                                             'pR2'])
        self.LDs = [x if x <= 0.99 else 0.99 for x in sorted(np.arange(
            rstart, rstop + rstep, rstep), reverse=True)]
        if customPrange:
            Ps = [float('%.1g' % float(x)) for x in customPrange.split(',')] 
        else:
            sta, sto = np.log10(pstart), np.log10(pstop)
            Ps = [float('%.1g' % 10**(x)) for x in np.concatenate(
                (np.arange(sta, sto), [sto]), axis=0)]
        self.Ps = sorted(Ps, reverse=True)
        self.executeRange()
    
    #----------------------------------------------------------------------
    def read_pheno(self, pheno):
        """
        Read a bim/fam files from the plink fileset
        
        :param str prefix: prefix of the bim file
        """
        if 'FID' in open(pheno).readline():
            ## asumes that has 3 columns with the first two with headers FID adn
            ## IID
            pheno = pd.read_table(pheno, delim_whitespace=True)
            pheno.rename(columns={pheno.columns[-1]: 'Pheno'}, inplace=True)
        else:
            Pnames = ['FID', 'IID', 'Pheno']
            pheno = pd.read_table(pheno, delim_whitespace=True, header=None,
                                  names=Pnames)
        return pheno

    #----------------------------------------------------------------------
    def read_Bim(self):
        """
        Read a bim/fam files from the plink fileset
        
        :param str prefix: prefix of the bim file
        """
        Bnames = ['CHR', 'SNP', 'cM', 'BP', 'A1', 'A2']
        bim = pd.read_table('%s.bim'%(self.bfile), delim_whitespace=True, 
                            header=None, names=Bnames)
        return bim

    #---------------------------------------------------------------------------
    def clumpVars(self, pval, r2):
        """
        Use plink to clump variants based on a pval and r2 threshold 
        
        :param float pval: Significance threshold for index SNPs
        :param float r2: LD threshold for clumping       
        """
        #outfn = '%s_%.2f_%d'%(self.outpref, str(pval), r2, self.window)
        outfn = '%s_%.2f_%d'%(self.outpref, r2, self.window)
        plink = ('%s --bfile %s -clump %s --clump-p1 %g --clump-p2 1 --clump-r2'
                 ' %f --clump-kb %d --out %s --allow-no-sex --keep-allele-order'
                 ' --pheno %s')
        if not os.path.isfile('%s.clumped'%(outfn)):
            plink = plink%(self.plinkexe, self.bfile, self.sumstats, pval, r2,
                           self.window, outfn, self.phenofn)
            o, e = execute(plink)
        ## read Clump file
        fn = '%s.clumped'%(outfn)
        try:
            table = pd.read_table(fn, delim_whitespace=True)  
        except FileNotFoundError as err:
            # check if the error is because of lack of significant clums
            if 'No significant --clump results' in open('%s.log'%outfn).read():
                table = None
            else:
                raise FileNotFoundError(err)
        return outfn, table      

    #---------------------------------------------------------------------------
    def clumpVars2(self, r2):
        """
        Use plink to clump variants based on a pval and r2 threshold 
        :param float r2: LD threshold for clumping       
        """
        outfn = '%s_%.2f_%d'%(self.outpref, r2, self.window)
        #plink = ('%s --bfile %s -clump %s --clump-p1 %g --clump-p2 1 --clump-r2'
        #         ' %f --clump-kb %d --out %s --allow-no-sex --keep-allele-order'
        #         ' --pheno %s')
        if not os.path.isfile('%s.clumped'%(outfn)):
            plink = plink%(self.plinkexe, self.bfile, self.sumstats, pval, r2,
                           self.window, outfn, self.phenofn)
            o, e = execute(plink)
        ## read Clump file
        fn = '%s.clumped'%(outfn)
        try:
            table = pd.read_table(fn, delim_whitespace=True)  
        except FileNotFoundError as err:
            # check if the error is because of lack of significant clums
            if 'No significant --clump results' in open('%s.log'%outfn).read():
                table = None
            else:
                raise FileNotFoundError(err)
        return outfn, table    
    
    
    #----------------------------------------------------------------------
    def range_profiles(self, name, range_tuple, r2, qfiledf):
        """
        read single profile from the q-range option
        """
        range_label = range_tuple.name
        assert float(range_tuple.name) == range_tuple.Max
        nsps = qfiledf[qfiledf.P <= range_tuple.Max].shape[0]
        profilefn = '%s.%s.profile' % (name, range_label)
        score = pd.read_table(profilefn, delim_whitespace=True)    
        score = score.loc[:,['FID', 'IID', 'SCORE']].merge(self.pheno,
                                                           on=['FID',
                                                               'IID'])
        score.rename(columns={'SCORE':'PRS'}, inplace=True)
        ## check if peno is binary:
        if set(score.Pheno) <= set([0,1]):
            score['pheno'] = score.Pheno - 1
            y, X = dmatrices('pheno ~ PRS', score, return_type = 'dataframe'
                             )
            logit = sm.Logit(y, X)
            logit = logit.fit(disp=0)
            ## get the pseudo r2 (McFadden's pseudo-R-squared.)
            pR2 = logit._results.prsquared
        else:
            ## Linear/quantitative trait
            slope, intercept, pR2, p_value, std_err = stats.linregress(
                score.Pheno, score.PRS)
            score['pheno'] = score.Pheno 
            pR2=pR2**2
        err = ((score.pheno - score.PRS)**2).sum()
        d = r'$\sum(Y_{AFR} - \widehat{Y}_{AFR|EUR})^{2}$'
        ## get parameter from name of clumped file
        return {'File':'%s.%s' % (name, range_label), 'LDthresh':r2, 
                'Pthresh':range_label, 'pR2':pR2,'pval':p_value,'SNP kept':nsps,
                d:err}     
        

    #---------------------------------------------------------------------------
    def ScoreClumped(self, clumped, r2):
        """
        Compute the PRS for the clumped variants
        
        :param str clumpPref: prefix of the clumpped file (plink format) 
        """
        ## merge to get specific clumps
        name, clumped = clumped
        if not os.path.isfile('%s.pickle' % (name)):
            if clumped is None:
                return 
            merge = self.sumstatsdf[self.sumstatsdf.SNP.isin(clumped.loc[:, 
                                                                         'SNP'])
                                    ]
            if not 'OR' in merge.columns:
                cols = ['SNP', 'A1', 'BETA']
            else:
                cols = ['SNP', 'A1', 'OR']
                merge['OR'] = np.log(merge.OR)
            merge = merge.loc[:, cols]
            ## write file for scoring
            merge.to_csv('%s.score'%(name), sep=' ', header=False, 
                         index=False)
            ## Score using plink
            qrange, qfile, qr, qf = self.qfile(clumped, r2)
            score = ('%s --bfile %s --score %s.score '
                     '--q-score-range %s %s --allow-no-sex --keep-allele-or'
                     'der --out %s --pheno %s')
            score = score%(self.plinkexe, self.bfile, name, qrange, qfile, name,
                           self.phenofn)
            if self.validate:
                score += ' --keep %s_test.keep' % self.bfile            
            o,e = execute(score)
            l = [self.range_profiles(name, range_label,r2,qf) for range_label in 
                 qr.itertuples()]
            with open('%s.pickle' % name, 'wb') as f:
                pickle.dump(l, f)
        else:
            with open('%s.pickle' % name, 'rb') as f:
                l = pickle.load(f)
        self.results = self.results.append(l).reset_index(drop=True)
        top = glob('%s.profile' % self.results.nlargest(1, 'pR2').File.tolist(
            )[0])
        with tarfile.open('Profiles_%.2f.tar.gz' % r2, mode='w:gz') as t:
            for fn in glob('*.profile'):
                if fn not in top:
                    t.add(fn)
                    os.remove(fn)        
    
    #----------------------------------------------------------------------
    def qfile(self, clumped, r2):
        """
        generate the qfile for --q-score-range
        """
        qrange = '%s_%.2f.qrange' % (self.outpref, r2)
        qfile = '%s%.2f.qfile' % (self.outpref, r2)
        qr = pd.DataFrame({'name':[str(x) for x in self.Ps], 'Min':np.zeros(len(
            self.Ps)), 'Max': self.Ps})
        order = ['name', 'Min', 'Max']
        qr.loc[:, order].to_csv(qrange, header=False, index=False, sep =' ')
        qf = clumped.loc[:,['SNP','P']]
        qf.to_csv(qfile, header=False, index=False, sep=' ')
        return qrange, qfile, qr, qf
    
    #---------------------------------------------------------------------------
    def executeRange(self):
        """
        perform the grid optimization
        """
        #combos = ((x, y) for y in self.Ps for x in self.LDs)
        print('Performing clumping in %s...' % self.outpref)
        #for r2, pval in tqdm(combos, total=(len(self.Ps)*len(self.LDs))):
        for r2 in tqdm(self.LDs, total=len(self.LDs)):
            self.ScoreClumped(self.clumpVars(1.0, r2), r2)
        self.results.sort_values('pR2', inplace=True, ascending=False)
        self.results.to_csv('%s.results'%(self.outpref), sep='\t', index=False)
        # plot results for comparisons
        piv = self.results.pivot_table(index='SNP kept', values=['pval', 'pR2'])
        piv.plot()
        plt.savefig('%s_PpT.pdf' % self.outpref)
        
        top10 = self.results.nlargest(10, 'pR2').reset_index(drop=True)
        top = top10.nlargest(1, 'pR2')
        for i in glob('%s.*' % top.File[0]):
            shutil.copy(i, '%s.BEST'%(i))        
        if self.clean:
            self.cleanup(top10)
        if not os.path.isdir('LOGs'):
            os.mkdir('LOGs')
        for f in glob('*.log'):
            shutil.move(f, 'LOGs')
        
    #----------------------------------------------------------------------
    def cleanup(self, top10):
        """
        cleanup files and report the top 10 predictions
        """
        print('Cleaning up ...')
        files = self.results.File
        tocle = files[~files.isin(top10.File)]
        tocle = [x for y in tocle for x in glob('%s*' % y)]
        profiles = glob('*.profile')                
        for fi in tqdm(tocle, total=len(tocle)):
            if os.path.isfile(fi):
                os.remove(fi)

#----------------------------------------------------------------------
def populate_n_qsub(pref, template, cml):
    """
    Populate the template and qsub it
    """
    with open(template) as F, open('%s.sh'%(pref),'w') as H:
        H.write(F.read().replace('$$', pref).replace('@@', cml))
    o, e = execute('qsub %s.sh'%(pref))
    return o
    
#----------------------------------------------------------------------
def paralell(args):
    """
    Execute all slices of the grid search in parallel
    """
    nproc = args.nprocs
    exeline = 'python %s -b %s -p %s -s %s -P %s -n %s -l %d -c %f -d %f -e %f ' 
    exeline += '-C %s, -N 4 -z -L %s'
    LDs = sorted(np.arange(args.rstart, args.rstop + args.rstep, args.rstep), 
                 reverse=True)
    if args.customP:
        Ps = [float('%.1g' % float(x)) for x in args.customP.split(',')]
    else:
        sta, sto = np.log10(pstart), np.log10(pstop)
        Ps = [float('%.1g' % 10**(int(x))) for x in np.concatenate((np.arange(
            sta, sto), [sto]), axis=0)]
    Ps = sorted(Ps, reverse=True)    
    combos = enumerate(np.array_split([(x, y) for y in Ps for x in LDs], nproc))
    jobs = []
    for i, c in combos:
        Rs = [x[0] for x in c]
        Ps = ','.join(list(set(['%.1g' % x[1] for x in c])))
        rstart = min(Rs)
        rstop = max(Rs)
        prefix = '%s_%d'%(args.prefix, i)
        exeline = exeline%(__file__, args.bfile, prefix, args.sumstats, 
                           args.pheno, args.plinkexe, int(args.LDwindow), 
                           float(rstart), float(rstop), float(args.rstep), 
                           Ps, args.label)
        jobs.append(populate_n_qsub(prefix, args.parallel, exeline))
    
    while not allDONE(jobs): sleep(10)
    execute('tail -n +2 %s_*.results > %s.results' % (args.prefix, args.prefix))
    cols = ['File', 'LDthresh',	'Pthresh',	'pR2']
    results = pd.read_table('%s.results' % args.prefix, delim_whitespace=True, 
                            header=None, names=cols, comment='=',
                            skip_blank_lines=True)
    return results
        
        
    #TODO: codify the merge

#----------------------------------------------------------------------
def plotPRSvsPheno(profilefn, label='EUR', pheno=None, plottype='png'):
    """
    Read the P+T best result and plot it against the phenotype
    """
    prefix = profilefn[:profilefn.rfind('.')]
    df = pd.read_table(profilefn, delim_whitespace=True)
    if pheno:
        phe = pd.read_table(pheno, header=None, names=['FID', 'IID', 'PHENO'],
                            delim_whitespace=True)
        df['PHENO'] = phe.PHENO
    score = 'SCORESUM' if 'SCORESUM' in df.columns else 'SCORE'
    df = df.rename(columns={'PHENO':'$Y_{%s}$' % label, 
                            score:'$PRS_{%s}$' % label})
    slope, intercept, r, p_value, std_err = stats.linregress(
        df.loc[:,'$Y_{%s}$' % label], df.loc[:,'$PRS_{%s}$' % label])      
    r2 = r**2
    ax = df.plot.scatter(x='$Y_{%s}$' % label, y='$PRS_{%s}$' % label)
    ax.add_artist(AnchoredText('$R^{2} = %.3f $' % r2, 2))  
    plt.tight_layout()
    label = label.replace('|', '-')
    plt.savefig('%s_PhenoVsPRS_%s.%s' % (prefix, label, plottype))
           
#----------------------------------------------------------------------
def main(args):
    """
    execute code
    """
    if args.parallel:
        res = paralell(args)
    else:
        PpT = PplusT(args.bfile, args.plinkexe, args.prefix, args.LDwindow, 
                 args.sumstats, args.pheno, args.rstart, args.rstop, args.rstep,
                 args.pstart, args.pstop, args.customP, args.clean, 
                 args.validate)
        res = PpT.results
    fil = res.nlargest(1,'pR2').reset_index(drop=True).loc[0,['File','Pthresh']
                                                           ].tolist()
    profilefn = '%s.%s.profile' % tuple(fil)
    plotPRSvsPheno(profilefn, label=args.label)

if __name__ == '__main__':
    #unittest.main()
    defplink = '/lb/project/gravel/hleap_projects/UKBB/PRSice_v1.25/plink_1.9_l'
    defplink += 'inux_160914'    
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--bfile', help='plink fileset prefix', 
                        required=True)    
    parser.add_argument('-p', '--prefix', help='prefix for outputs',
                        required=True)
    parser.add_argument('-s', '--sumstats', help='Filename of sumary statistics',
                        required=True)    
    parser.add_argument('-P', '--pheno', help='Filename of phenotype file',
                        required=True)    
    parser.add_argument('-n', '--plinkexe', help='Path and executable file of \
    plink', default= defplink) 
    parser.add_argument('-l', '--LDwindow', help='Physical distance threshold '+
                        'for clumping in kb (250kb by default)', type=int, 
                        default=250)
    parser.add_argument('-c', '--rstart', help='minimum R2 threshold. '
                        'Default: 0.1', type=float, default=0.1)   
    parser.add_argument('-d', '--rstop', help='maximum R2 threshold. '
                        'Default: 0.8', type=float, default=0.8)    
    parser.add_argument('-e', '--rstep', help='step for R2 threshold. '
                        'Default: 0.1', type=float, default=0.1)
    parser.add_argument('-v', '--pstart', help='Minimum value for for the Pval'+
                        ' range. Default: 1E-8', type=float, default=1E-8)
    parser.add_argument('-w', '--pstop', help='Maximum value for for the Pval'+
                        ' range. Default: 1', type=float, default=1)    
    parser.add_argument('-C', '--customP', help='Custom pvalue range.' + 
                        'Default: (None)', default=None)
    parser.add_argument('-R', '--parallel', help='Use a cluster. Pass the '+
                        'filename of the qsub template, with @@ in the command'+
                        ' section, and $$ where the name (prefix) should fo.', 
                        default=None)    
    parser.add_argument('-N', '--nprocs', help='If paralell is used, the number'
                        + 'of nodes to use. The number of processors per node' +
                        ' have to be set in the template file.', default=8, 
                        type=int) 
    parser.add_argument('-z', '--clean', help='Cleanup the clump and profiles', 
                        default=False, action='store_true')    
    parser.add_argument('-L', '--label', help='Label of the populations being' +
                        ' analyzed.', default='EUR')     
    parser.add_argument('-S', '--validate', help=('Filename of individuals to '
                                                  'validate with in plink forma'
                                                  't for the --keep flag'), 
                                            default=None)      
    
    args = parser.parse_args()
    main(args)