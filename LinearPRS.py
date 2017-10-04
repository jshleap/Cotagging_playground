'''
Continuous model of PRS to test the liability threshold model
'''

import argparse, os
import numpy as np
import pandas as pd
from scipy import stats
from subprocess import Popen, PIPE
from matplotlib import pyplot as plt
plt.style.use('ggplot')
from matplotlib.offsetbox import AnchoredText

#----------------------------------------------------------------------
def executeLine(line):
    """
    execute line with subprocess
    """
    pl = Popen(line, shell=True, stderr=PIPE, stdout=PIPE)
    o, e = pl.communicate()
    return o, e

#----------------------------------------------------------------------
def read_BimFam(prefix):
    """
    Read a bim/fam files from the plink fileset
    """
    Bnames = ['CHR', 'SNP', 'cM', 'BP', 'A1', 'A2']
    bim = pd.read_table('%s.bim'%(prefix), delim_whitespace=True, header=None,
                        names=Bnames)
    #Fnames = ['FID', 'IID', 'father', 'mother', 'Sex', 'Phenotype']    
    #fam = pd.read_table('%s.fam'%(prefix), delim_whitespace=True, header=None,
    #                    names=Bnames)    
    return bim#, fam

#----------------------------------------------------------------------
def read_freq(bfile, plinkexe):
    if not os.path.isfile('%s.frq.gz' % bfile):
        nname = os.path.split(bfile)[-1]
        frq = '%s --bfile %s --freq gz --keep-allele-order --out %s'
        line = frq % (plinkexe, bfile, nname)
        #print('Executing line %s' % line)
        o,e = executeLine(line)
        #print(o,e)
        frq = pd.read_table('%s.frq.gz' % nname, delim_whitespace=True)
    else:
        frq = pd.read_table('%s.frq.gz' % bfile, delim_whitespace=True)  
    #filter MAFs greater than 0.9 and smaller than 0.1
    return frq[(frq.MAF < 0.9) & (frq.MAF > 0.1)]

#----------------------------------------------------------------------
def TruePRS(outprefix, bfile, h2, ncausal, plinkexe, causalmean=0, snps=None, 
            frq=None, causaleff=None):
    """
    Compute the known polygenic risk score
    """
    if not os.path.isfile('%s.full'%(outprefix)):
        ## Read bim file
        if frq is None:
            frq = read_freq(bfile, plinkexe)
        print ('Total Number of variants available: %d' % frq.shape[0])
        totalsnps = '%s.totalsnps' % outprefix
        frq.to_csv(totalsnps, sep=' ', header=False, index=False)
        ## Get causal mutation indices randomly distributed
        if ncausal > frq.shape[0]:
            print('More causals than available snps. Setting it to %d' % 
                  frq.shape[0])
            ncausal = frq.shape[0]        
        if snps is None:
            causals = frq.sample(ncausal, replace=False)
        else:
            causals = frq[frq.SNP.isin(snps)]
        if causaleff is None:
            causals['eff'] = np.random.normal(loc=causalmean, scale=np.sqrt(
                h2/ncausal), size=ncausal)
        else:
            causals = causals.merge(causaleff, on='SNP')
        ## write snps and effect to score file
        causals.loc[:, 'norm'] = np.sqrt((2 * causals.MAF) * (1 - causals.MAF))
        causals.loc[:, 'beta'] = causals.loc[:, 'eff']/causals.norm      
        scfile = causals.sort_index()
        scfile.loc[:, ['SNP', 'A1', 'beta']].to_csv('%s.score'%(outprefix), 
                                                    sep=' ', header=False, 
                                                    index=False)
        scfile.to_csv('%s.full'%(outprefix), sep=' ', index=False)        
    else:
        scfile = pd.read_table('%s.full'%(outprefix), delim_whitespace=True, 
                               header=None, names=['SNP', 'Allele', 'beta'])
    ## Score using plink
    score = ('%s --bfile %s --score %s.score sum --allow-no-sex --extract %s '
             '--keep-allele-order --out %s')
    if not os.path.isfile('%s.profile'%(outprefix)):
        executeLine(score%(plinkexe, bfile, outprefix, totalsnps, outprefix))
    score = pd.read_table('%s.profile'%(outprefix), delim_whitespace=True)
    score = score.rename(columns={'SCORESUM':'gen_eff'})
    return score, scfile, totalsnps
        
#----------------------------------------------------------------------
def liabilities(prefix, h2, ncausal, prs_true, noenv=False):
    """
    Generate phenotypes and real betas. The name is for backwards compatibility
    
    """
    nind = prs_true.shape[0]
    if noenv:
        env_effect = np.zeros(nind)
    else:
        env_effect = np.random.normal(loc=0,scale=np.sqrt(1-h2), size=nind)
    prs_true['env_eff'] = env_effect
    prs_true['PHENO'] = prs_true.gen_eff + prs_true.env_eff
    prs_true.to_csv('%s.prs_pheno.gz'%(prefix), sep='\t', compression='gzip',
                    index=False)
    prs_true.loc[: , ['FID', 'IID', 'PHENO']].to_csv('%s.pheno'%(prefix), 
                                                         sep=' ', header=False, 
                                                         index=False)    
    return prs_true

#----------------------------------------------------------------------
def PlinkGWAS(plinkexe, bfile, outprefix, covs=None, nosex=False, 
              threads=False, maxmem=False, validate=None, validsnpsfile=None):
    """
    Execute plink gwas. This assumes binary phenotype
    
    :param str plinkexe: Path and executable of plink
    :param str bfile: Prefix for the bed fileset
    :param str outprefix: Prefix for the outputs
    :param bool nosex: If option --allow-no-sex should be used
    :param str covs: Filename with the covariates to use
    :param int/bool validate: To split the bed into test and training 
    """   
    ## for plinkgwas string:
    ## 1) plink path and executable
    ## 2) prefix for the bed fileset
    ## 3) Name of covariate file
    ## 4) names of the columns to use in the covariance file separated by "," 
    ## or '-' if range
    ## 5) prefix for outputs
    plinkgwas = "%s --bfile %s --assoc fisher-midp --linear --pheno %s.pheno"
    plinkgwas+= " --prune --out %s_gwas --ci 0.95 --keep-allele-order --vif 100"
    plinkgwas = plinkgwas%(plinkexe, bfile, outprefix, outprefix)
    if validsnpsfile is not None:
        plinkgwas+= " --extract %s" % validsnpsfile
    if covs:
        plinkgwas += " --covar %s keep-pheno-on-missing-cov" % covs
    if nosex:
        plinkgwas += ' --allow-no-sex'
    else:
        plinkgwas += ' --sex'
    if threads:
        plinkgwas += ' --threads %s' % threads
    if maxmem:
        plinkgwas += ' --memory %s' % maxmem 
    if validate :
        plinkgwas += ' --keep %s_test.keep' % bfile
    out = executeLine(plinkgwas)  
    return pd.read_table('%s_gwas.assoc.linear' % outprefix, 
                         delim_whitespace=True)

#----------------------------------------------------------------------
def main(args):
    """ execute the code """
    prs_true, truebeta = TruePRS(args.outprefix, args.bfile, args.h2, 
                                 args.ncausal, args.plinkexe, args.causalmean,
                                 )
    prs_true = liabilities(args.outprefix, args.h2, args.ncausal, prs_true, 
                           noenv=args.noenv)
    if args.GWAS:
        gwas = PlinkGWAS(args.plinkexe, args.bfile, args.outprefix, args.covs, 
                  args.nosex, args.threads, args.maxmem, validate=args.validate)
        merged = gwas.merge(truebeta, on='SNP')
        slope, intercept, r2, p_value, std_err = stats.linregress(merged.beta, 
                                                                  merged.BETA)
        ax = merged.plot.scatter(x='beta', y='BETA', s=2, alpha=0.5)
        ax.add_artist(AnchoredText('$R^{2} = %.3f $' % r2**2, 2))  
        plt.tight_layout()
        plt.savefig('%s_truebetavsinferred.png'%(args.outprefix))  
        plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--ncausal', type=int, default=200)
    parser.add_argument('-n', '--ncontrols', type=int, default=10000)    
    parser.add_argument('-c', '--ncases', type=int, default=10000) 
    parser.add_argument('-b', '--h2', type=float, default=0.66)
    parser.add_argument('-p', '--prevalence', type=float, default=0.05)   
    parser.add_argument('-o', '--outprefix', default='sim0')
    parser.add_argument('-B', '--bfile', default='EUR')
    parser.add_argument('-P', '--plinkexe', default='~/Programs/plink_mac/plink'
                        )
    parser.add_argument('-G', '--GWAS', default=True, action='store_false')
    parser.add_argument('-C', '--covs', default=None, action='store')
    parser.add_argument('-s', '--nosex', default=False, action='store_true')
    parser.add_argument('-t', '--threads', default=False, action='store')
    parser.add_argument('-M', '--maxmem', default=False, action='store')
    parser.add_argument('-e', '--noenv', default=False, action='store_true')
    parser.add_argument('-f', '--causalmean', default=0, type=float)
    parser.add_argument('-v', '--validate', default=None)
    args = parser.parse_args()
    main(args)    