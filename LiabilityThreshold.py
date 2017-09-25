import argparse, os
import numpy as np
import pandas as pd
from subprocess import Popen, PIPE

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
def TruePRS(outprefix, bfile, h2, ncausal, plinkexe, causal_pos=None, 
            effects=[], norm=False, gcta=False, noenv=False):
    """
    Compute the known polygenic risk score
    """
    if os.path.isfile('%s.score'%(outprefix)):
        scfile = pd.read_table('%s.score'%(outprefix), delim_whitespace=True,
                               header=None, names=['SNP', 'A1', 'beta'])
    else:
        ## Read bim file
        bim = read_BimFam(bfile)        
        ## Get causal mutation indices evenly distributed
        if causal_pos:
            causal_mut_index = causal_pos
        else:
            causal_mut_index = np.linspace(0, bim.shape[0]-1, ncausal, dtype=int
                                           )    
        ## get betas or causal effects from a ~N(0,h2/ncausal)
        if os.path.isfile('%s.causaleff.npy'%(outprefix)):
            causal_effects = np.load('%s.causaleff.npy'%(outprefix))
        else:
            if effects:
                causal_effects = effects
            else:
                causal_effects = np.random.normal(loc=0,scale=h2/ncausal, 
                                                  size=ncausal)
                #np.repeat(h2/ncausal, ncausal)
        ## write snps and effect to score file
        causal_snps = bim.SNP[causal_mut_index]
        causal_Acod = bim.A1[causal_mut_index]
        scfile = pd.DataFrame({'SNP':causal_snps, 'Allele':causal_Acod, 
                               'beta':causal_effects}).loc[:, ['SNP', 'Allele', 
                                                               'beta']]
        scfile.to_csv('%s.score'%(outprefix), sep=' ', header=False, index=False
                      )
    if gcta:
        scfile.loc[:,['SNP','beta']].to_csv('%s_gcta.snplist'%(outprefix),
                                            sep=' ', header=False, 
                                            index=False)
    ## Score using plink
    #score = '%s --bfile %s --score %s.score sum --allow-no-sex '
    score = '%s --bfile %s --score %s.score --allow-no-sex '
    score += '--keep-allele-order --out %s'
    if not os.path.isfile('%s.profile'%(outprefix)):
        executeLine(score%(plinkexe, bfile, outprefix, outprefix))
    score = pd.read_table('%s.profile'%(outprefix), delim_whitespace=True)
    nind = score.shape[0]
    env_effect = np.random.normal(loc=0,scale=1-h2, size=nind)
    if noenv:
        env_norm = np.zeros(nind)
    else:
        if norm:
            env_norm = (env_effect - np.mean(env_effect)) / np.std(env_effect)
        else:
            env_norm = env_effect
            #np.random.normal(loc=0,scale=1-h2, size=nind)#
    score['env_norm'] = env_norm    
    if norm:
        score['norm'] = (score.SCORESUM-score.SCORESUM.mean())/score.SCORESUM.std()
    else:
        score['norm'] = score.SCORESUM#
    return score
    
#----------------------------------------------------------------------
def liabilities(prefix, bfile, h2, ncausal, prevalence, plinkexe, ncontrols, 
                ncases, causal_pos=None, effects=[], norm=False, noenv=False):
    """
    Generate phenotypes and real betas from the liability threshold model
    
    """
    prs_true = TruePRS(prefix, bfile, h2, ncausal, plinkexe, norm=norm,
                       causal_pos=causal_pos, effects=effects, noenv=noenv)
    nind = prs_true.shape[0]
    
    env_norm = prs_true.env_norm
    prs_norm = prs_true.norm
    if norm:
        total_liabi = (np.sqrt(h2) * prs_norm ) + (np.sqrt(1 - h2) * env_norm)
    else:
        total_liabi = prs_norm + env_norm
    #(np.sqrt(h2) * prs_norm ) + (np.sqrt(1 - h2) * env_norm)
    #if ncausal == 1:
        #vals = sorted(set(total_liabi))
        #if len(vals) != 3:
            #raise Exception
        #cases = np.where(total_liabi == vals[-1])[0]
        #controls = np.where(total_liabi == vals[0])[0]
        #inter = np.where(total_liabi == vals[1])[0]
        #intcont = int(inter.shape[0] * (prevalence))
        #controls = sorted(np.concatenate((controls, inter[ : intcont])))
        #cases = sorted(np.concatenate((cases, inter[intcont : ])))
    #else:
    sorted_liability = sorted(total_liabi)
    liabThresh = sorted_liability[int((1-prevalence) * len(sorted_liability))]
    cases = [i for (i, x) in enumerate(total_liabi) if x >= liabThresh]  
    controls = np.setdiff1d(np.arange(nind), cases)
    prs_true['Liab_Thresh'] = liabThresh
    prs_true.loc[cases,'PHENO'] = 2
    prs_true.loc[controls,'PHENO'] = 1
    prs_true['Liability'] = total_liabi
    prs_true.to_csv('%s.prs_pheno.gz'%(prefix), sep='\t', compression='gzip',
                        index=False)
    #ncases = min(len(cases), ncases)
    #sampCases = np.random.choice(cases, ncases, replace=False)
    #ncontrols = min(len(controls), ncontrols)
    #sampControls = np.random.choice(controls, ncontrols, replace=False) 
    #sample = np.concatenate((sampCases,sampControls))
    prs_true.loc[: , ['FID', 'IID', 'PHENO']].to_csv('%s.pheno'%(prefix), 
                                                         sep=' ', header=False, 
                                                         index=False)    
    return prs_true

#----------------------------------------------------------------------
def Logistic(prefix, bfile, h2, ncausal, plinkexe, ncontrols, ncases, prevalence,
             causal_pos=None, effects=[], norm=False, noenv=False, 
             threshold=False):
    """
    Generate phenotypes and real betas from the logistic model
    
    """
    ## eta = B0 + (Bj * Xij)
    ## (Bj * Xij) = prs    
    prs_true = TruePRS(prefix, bfile, h2, ncausal, plinkexe, norm=norm, 
                       causal_pos=causal_pos, effects=effects, noenv=noenv) 
    eta = prs_true.SCORESUM
    Pyis1 = 1 / (1 + np.exp(-eta))
    prs_true['PHENO'] = np.random.binomial(1, Pyis1) + 1
    
    #def binarization(y, thresh=0):
        #if y > thresh:
            #return 2
        #else:
            #return 1
    
    #nind = prs_true.shape[0]
    ##env_effect = np.random.normal(loc=0,scale=1-h2, size=nind)
    #if noenv:
        #env_norm = np.zeros(nind)
    #else:
        #if norm:
            #env_norm = np.random.normal(size=nind)#(env_effect - np.mean(env_effect)) / np.std(env_effect)
        #else:
            #env_norm = np.random.normal(loc=0,scale=1-h2, size=nind)#env_effect
    #prs_true['env_norm'] = env_norm
    #if norm:
        #prs_true['unobserved_pheno'] = (np.sqrt(h2) * prs_true.norm ) + \
        #(np.sqrt(1 - h2) * prs_true.env_norm)
    #else:
        #prs_true['unobserved_pheno'] =  prs_true.norm + prs_true.env_norm
    #liab = sorted(prs_true.unobserved_pheno)
    #liab = liab[int((1-prevalence) * len(liab))] 
    #prs_true['PHENO'] = [binarization(x, thresh=liab) for x in 
                         #prs_true.unobserved_pheno]
    ##[binarization(x) for x in prs_true.unobserved_pheno]
    prs_true.to_csv('%s.prs_pheno.gz'%(prefix), sep='\t', compression='gzip',
               index=False)
    prs_true.loc[: , ['FID', 'IID', 'PHENO']].to_csv('%s.pheno'%(prefix), sep=' ', 
                                                header=False, index=False)    
    return eta#prs_true    
    
#----------------------------------------------------------------------
def gcta(prefix, bfile, h2, prevalence, ncontrols, ncases, ncausal, plinkexe,
         noenv=False):
    """
    Try simulating the phenotypes based on gcta required gcta on path
    """
    true_prs = TruePRS(prefix, bfile, h2, ncausal, plinkexe, gcta=True, 
                       noenv=noenv)
    snplist = '%s_gcta.snplist'%(prefix)
    cmdl = 'gcta_mac --bfile %s --simu-cc %d %d --simu-causal-loci %s '
    cmdl += '--simu-hsq %f  --simu-k %f --out %s'
    executeLine(cmdl % (bfile, ncontrols, ncases, snplist, h2, prevalence,
                        '%s' % prefix))
    os.rename('%s.phen' % prefix, '%s.pheno' % prefix)
    
#----------------------------------------------------------------------
def PlinkGWAS(plinkexe, bfile, outprefix, covs=None, nosex=False, geno=False,
              recesive=False, dominant=False, adjust=False, threads=False, 
              maxmem=False, beta=False):
    """
    Execute plink gwas. This assumes binary phenotype
    
    :param str plinkexe: Path and executable of plink
    :param str bfile: Prefix for the bed fileset
    :param str outprefix: Prefix for the outputs
    :param bool nosex: If option --allow-no-sex should be used
    :param str covs: Filename with the covariates to use
    """   
    ## for plinkgwas string:
    ## 1) plink path and executable
    ## 2) prefix for the bed fileset
    ## 3) Name of covariate file
    ## 4) names of the columns to use in the covariance file separated by "," 
    ## or '-' if range
    ## 5) prefix for outputs
    if beta:
        plinkgwas = "%s --bfile %s --assoc fisher-midp --logistic beta no-x-sex"
    else:
        plinkgwas = "%s --bfile %s --assoc fisher-midp --logistic no-x-sex"
    plinkgwas+= " --pheno %s.pheno --prune --out %s_gwas --ci 0.95 "
    plinkgwas+="--keep-allele-order"
    plinkgwas = plinkgwas%(plinkexe, bfile, outprefix, outprefix)
    if geno:
        plinkgwas += ' --genotypic'
    elif recesive:
        plinkgwas += ' --recessive'
    elif dominant:
        plinkgwas += ' --dominant'
    if covs:
        plinkgwas += " --covar %s keep-pheno-on-missing-cov"%(covs)
    if adjust:
        plinkgwas += ' --adjust'
    if nosex:
        plinkgwas += ' --allow-no-sex'
    else:
        plinkgwas += ' --sex'
    if threads:
        plinkgwas += ' --threads %s'%(threads)
    if maxmem:
        plinkgwas += ' --memory %s'%(maxmem)
    out = executeLine(plinkgwas)    
    
#----------------------------------------------------------------------
def Liability2Logistic(prevalence):
    """
    Model the apropriate liability-logistic model
    """
    pass
    
#----------------------------------------------------------------------
def main(args):
    """ execute the code """
    #prs_true = TruePRS(args.outprefix, args.bfile, args.h2, args.ncausal,
    #                   args.plinkexe)
    if args.liabity:
        prs_true = liabilities(args.outprefix, args.bfile, args.h2, args.ncausal, 
                               args.prevalence, args.plinkexe, args.ncontrols, 
                               args.ncases, causal_pos=args.capos, 
                               effects=args.teff, norm=args.normalize, 
                               noenv=args.noenv)
    elif args.logistic:
        prs_true = Logistic(args.outprefix, args.bfile, args.h2, args.ncausal, 
                            args.plinkexe, args.ncontrols, args.ncases,
                            args.prevalence, norm=args.normalize, 
                            noenv=args.noenv)
    else:
        gcta(args.outprefix, args.bfile, args.h2, args.prevalence, 
             args.ncontrols, args.ncases, args.ncausal, args.plinkexe, 
             noenv=args.noenv)
    if args.GWAS:
        PlinkGWAS(args.plinkexe, args.bfile, args.outprefix, args.covs, 
                  args.nosex, args.geno, args.recesive, args.dominant, 
                  args.adjust, args.threads, args.maxmem)    

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
    parser.add_argument('-g', '--geno', default=False, action='store_true')
    parser.add_argument('-r', '--recesive', default=False, action='store_true')
    parser.add_argument('-d', '--dominant', default=False, action='store_true')
    parser.add_argument('-a', '--adjust', default=False, action='store_true')
    parser.add_argument('-l', '--liabity', default=False, action='store_true')
    parser.add_argument('-L', '--logistic', default=False, action='store_true')
    parser.add_argument('-N', '--normalize', default=False, action='store_true')
    parser.add_argument('-S', '--capos', default=None, type=int)
    parser.add_argument('-T', '--teff', default=[], action='append', type=float)
    
    
    args = parser.parse_args()
    main(args)    