#!/usr/bin/env python
#coding:utf-8
"""
  P+T+C PRS optimization using Co-tagging
  Author:  Jose Sergio Hleap --
  Purpose: Implement the P + T + C method
  Created: 09/12/17
"""

## Libraries ###################################################################
#import unittest
import numpy as np
import pandas as pd
from time import sleep
from tqdm import tqdm
from scipy import stats
from glob import glob as G
from patsy import dmatrices
import argparse, os, shutil
import statsmodels.api as sm
import matplotlib.pyplot as plt
from subprocess import Popen, PIPE
from collections import defaultdict
from matplotlib.offsetbox import AnchoredText
################################################################################
plt.style.use('ggplot')
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

class PpTpC:
    """
    Class to implement a P+T+C approach when the input does not have ambiguities
    (e.g simulations) or it has been corrected and QCd
    """

    #----------------------------------------------------------------------
    def __init__(self, bfile, cotag, outprefix, window, sumstats, pheno,
                 plinkexe, tag='Cotagging', rstart=0.1, rstop=0.8, rstep=0.1, 
                 pstart=1E-8, pstop=1, customPrange=None, clean=False):
        """
        Constructor
        
        :param str bfile: prefix for genotype from which to compure r2 (plink)
        :param str cotag: Filename of the cotagfile
        :param str outprefix: Prefix for output files
        :param int window: Physical distance threshold for clumping
        :param str sumstats: File name of the sumary statistics
        :param str pheno: File name of the phenotype file
        :param str plinkexe: path to plink executable
        :param str tag: Column from the cotagging file to be used as score
        :param float rstart: lower bound for the range of R2
        :param float rstop: upper bound for the range of R2
        :param float rstep: step for the range of R2
        :param float pstart: smallest pvalue threshold
        :param float pstop: largest pvalue threshold
        :param str customPrange: string with comma separated floats determining
        the range for pvalues
        :param bool clean: clean up the files
        """
        self.clean = clean
        self.bfile = bfile
        self.cotagfn = cotag
        self.cotagdf = pd.read_table(cotag, sep='\t')
        self.tagcol = tag
        self.sumstats = sumstats
        self.sumstatsdf = pd.read_table(sumstats, delim_whitespace=True)
        self.plinkexe = plinkexe
        self.outpref = outprefix
        self.window = window
        self.phenofn = pheno        
        self.pheno = self.read_pheno(pheno)
        self.LDs = sorted(np.arange(rstart, rstop + rstep, rstep), reverse=True)
        if customPrange:
            Ps = [float('%.1g' % float(x)) for x in customPrange.split(',')] 
        else:
            sta, sto = np.log10(pstart), np.log10(pstop)
            Ps = [float('%.1g' % 10**(x)) for x in np.concatenate(
                (np.arange(sta, sto), [sto]), axis=0)]
        self.Ps = sorted(Ps, reverse=True)  
        self.results = []
    
    #----------------------------------------------------------------------
    def composite_score(self):
        """
        include the scores
        """
        merged = self.sumstats.merge(self.cotagdf, on='SNP')
        merged['PnC_pval'] = (1-merged.P) * merged.loc[:, self.tagcol]
        merged['PnC_B'] = merged.BETA * merged.loc[:, self.tagcol]
        merged['PnC_B2'] = (merged.BETA**2) * merged.loc[:, self.tagcol]
        merged['PnC_absB'] = abs(merged.BETA) * merged.loc[:, self.tagcol]
        return merged
    
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
        
    #---------------------------------------------------------------------------
    def clumpVars(self, pval, r2):
        """
        Use plink to clump variants based on a pval and r2 threshold 
        
        :param float pval: Significance threshold for index SNPs
        :param float r2: LD threshold for clumping       
        """
        outfn = '%s_%s_%.2f_%d'%(self.outpref, str(pval), r2, self.window)
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
    
    #----------------------------------------------------------------------
    def get_tagSNP(self, clumpdf, mergedf):
        """
        Loop over a clumped dataframe, extract the clumps and score them with
        the developed scores in merged
        """
        scores = ['PnC_pval', 'PnC_B', 'PnC_B2', 'PnC_absB']
        d = defaultdict(list)
        for row in clumpdf.itertuples():
            snps = [x[:x.find('(')] for x in row.SP2.split(',') if row.SP2 
                    != 'NONE'] + row.SNP.tolist()
            subdf = mergedf[mergedf.SNP.isin(snps)]
            for col in scores:
                d[col].append(subdf.nlargest(1, col).loc[:,['SNP', 'A1', 'BETA']
                                                         ])
        d['P+T'].append(self.sumstatsdf[self.sumstatsdf.SNP.isin(clumpeddf.loc[
            :,'SNP'])])
        return d
    
    #----------------------------------------------------------------------
    def extract_n_score(self, pref, snps):
        """
        Extract the snps and score them
        """
        # Write extract file
        snps.SNP.to_csv('%s.extract' % pref, index=False, header=False, sep=' ')
        # Write score file
        snps.to_csv('%s.score' % pref, index=False, header=False, sep=' ')  
        # score
        score = ('%s --bfile %s --score %s.score --allow-no-sex '
                 '--keep-allele-order --out %s --pheno %s')
        score = score%(self.plinkexe, self.bfile, pref, pref, self.phenofn)  
        o,e = execute(score)
        return pd.read_table('%s.profile'%(name), delim_whitespace=True) 
    
    #---------------------------------------------------------------------------
    def ScoreClumped(self, clumped, pval, r2):
        """
        Compute the PRS for the clumped variants
        
        :param tuple clumped: Tuple with the prefix of clumpled file and its df
        """
        ## get scores
        mergedf = self.composite_score()
        ## get clumps
        name, clumpdf = clumped
        if clumpdf is None:
            return 
        ## Get the proxy SNPs as per strategy
        proxies = self.get_tagSNP(clumpdf, mergedf)
        for strategy, snps in proxies.items():
            pref = '%s_%.2f_%.1g' % (strategy, r2, pval)
            prof = self.extract_n_score(pref, snps)
            lr = stats.linregress(profile.PHENO, profile.SCORE)
            self.results.append({'File': '%s.profile' % pref, 'LDthresh': r2, 
                                 'Pthresh':pval, 'R2':lr.rvalue**2, 
                                 'pval':lr.pvalue, 'strategy':strategy})

    #---------------------------------------------------------------------------
    def executeRange(self):
        """
        perform the grid optimization
        """
        combos = ((x, y) for y in self.Ps for x in self.LDs)
        print('Performing clumping in %s...' % self.outpref)
        for r2, pval in tqdm(combos, total=(len(self.Ps)*len(self.LDs))):
            self.ScoreClumped(self.clumpVars(pval, r2))
        self.results = pd.DataFrame(self.results)
        self.results.sort_values('R2', inplace=True, ascending=False)
        self.results.to_csv('%s.results'%(self.outpref), sep='\t', index=False)
        top10 = self.results.nlargest(10, 'pR2')
        top = top10.nlargest(1, 'pR2')
        for i in G(top.File[0]):
            shutil.copy(i, '%s.BEST'%(i))        
        if self.clean:
            self.cleanup(top10)
        if not os.path.isdir('LOGs'):
            os.mkdir('LOGs')
        for f in G('*.log'):
            shutil.move(f, 'LOGs')
        
    #----------------------------------------------------------------------
    def cleanup(self, top10):
        """
        cleanup files and report the top 10 predictions
        """
        files = self.results.File
        tocle = files[~files.isin(top10.File)]
        tocle = [x for y in tocle for x in G('%s*' % y)]
        print('Cleaning up ...')
        for fi in tqdm(tocle, total=len(tocle)):
            if os.path.isfile(fi):
                os.remove(fi)
                
#----------------------------------------------------------------------
def plot_results(results, label='EUR', pheno=None, plottype='png'):
    """
    Read the P+T best result and plot it against the phenotype
    """
    df = df.groupby(['strategy'])['R2'].max()
    df.plot.bar()
    #df = pd.read_table(profilefn, delim_whitespace=True)
    #if pheno:
        #phe = pd.read_table(pheno, header=None, names=['FID', 'IID', 'PHENO'],
                            #delim_whitespace=True)
        #df['PHENO'] = phe.PHENO
    #score = 'SCORESUM' if 'SCORESUM' in df.columns else 'SCORE'
    #df = df.rename(columns={'PHENO':'$Y_{%s}$' % label, 
                            #score:'$PRS_{%s}$' % label})
    #slope, intercept, r2, p_value, std_err = stats.linregress(
        #df.loc[:,'$Y_{%s}$' % label], df.loc[:,'$PRS_{%s}$' % label])      
    #ax = df.plot.scatter(x='$Y_{%s}$' % label, y='$PRS_{%s}$' % label)
    #ax.add_artist(AnchoredText('$R^{2} = %.3f $' % r2, 2))  
    plt.tight_layout()
    #label = label.replace('|', '-')
    plt.savefig('Strategies_%s.%s' % (label, plottype))
    
#----------------------------------------------------------------------
def main(args):
    """
    execute code
    """
    PpTpC = PpTpC(args.bfile, args.plinkexe, args.prefix, args.LDwindow, 
                  args.sumstats, args.pheno, args.rstart, args.rstop,
                  args.rstep, args.pstart, args.pstop, args.customP, args.clean)
    res = PpT.results
    #profilefn = '%s.profile' % res.nlargest(1,'pR2').File[0]
    plot_results(res, label=args.label)

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
    
    args = parser.parse_args()
    main(args)