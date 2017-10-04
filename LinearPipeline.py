'''
Pipeline to simulate a continous trait in both EUR and AFR, do the associations,
do the P+T in EUR and score and plot the null model in AFR
'''
import os
import shutil
import argparse
from PplusT import *
from glob import glob
from LinearPRS import *
from NullSNPPruning import *
from PlotCovariances import *
from matplotlib import pyplot as plt
plt.style.use('ggplot')
from matplotlib.offsetbox import AnchoredText

def read_singlefrq(bfile, plinkexe):
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
    return frq

#----------------------------------------------------------------------
def read_freqs(bfile1, label1, bfile2, label2, plinkexe):
    '''
    compute the frequencies, merge the files and filter it
    '''
    frq1 = read_singlefrq(bfile1, plinkexe)
    frq2 = read_singlefrq(bfile2, plinkexe)
    frqs = frq1.merge(frq2, on=['CHR', 'SNP'], suffixes=['_%s' % label1, 
                                                         '_%s' % label2])
    frqs = frqs[(frqs.loc[:, 'MAF_%s' % label1] < 0.9) & 
                (frqs.loc[:, 'MAF_%s' % label1] > 0.1) &
                (frqs.loc[:, 'MAF_%s' % label2] > 0.1) &
                (frqs.loc[:, 'MAF_%s' % label2] < 0.9)]
    frqs.plot.scatter(x='MAF_%s' % label1, y='MAF_%s' % label2)
    plt.savefig('%s_%s_MAFs.png' % (label1, label2))
    plt.close()
    frq1 = frq1[frq1.SNP.isin(frqs.SNP)]
    frq2 = frq2[frq2.SNP.isin(frqs.SNP)]
    #filter MAFs greater than 0.9 and smaller than 0.1
    return frq1, frq2

def train_test(bfile, splits=10):
    """
    Generate a list of individuals for training and a list for validation.
    The list is to be passed to plink. It will take one split as validation and
    the rest as training.
    
    :param str bfile: prefix for the bed fileset 
    :param int splits: Number of splits to be done
    """
    trainthresh = (splits - 1) / splits
    if not os.path.isfile('%s_train.keep' % (bfile)) and not os.path.isfile(
        '%s_test.keep' % (bfile)):
        fam = pd.read_table('%s.fam' % bfile, delim_whitespace=True, 
                            header=None,names=['FID', 'IID', 'a', 'b', 'c', 'd']
                            )
        msk = np.random.rand(len(fam)) < trainthresh
        fam.loc[msk, ['FID', 'IID']].to_csv('%s_train.keep' % (bfile), 
                                            header=False,index=False, sep=' ')
        fam.loc[~msk, ['FID', 'IID']].to_csv('%s_test.keep' % (bfile), 
                                             header=False, index=False, sep=' ')
        
    
def simulate_phenos(bed, label, plinkexe, ncausal, h2, thr=False, mem=False, 
                    snps=None, frq=None, causaleff=None, ext='png', 
                    splits=False):
    '''
    Simulate the phenotypes in both pops with the same causal variants and 
    effect.
    '''
    if os.path.isfile('pheno.pickle'):
        with open('pheno.pickle', 'rb') as f:
            gwas, truebeta, bestr2, bed = pickle.load(f)
    else:
        if not os.path.isfile('%s.full'%(label)):
            geneff, truebeta , validsnpfile = TruePRS(label, bed, h2, ncausal, 
                                                      plinkexe, snps=snps, 
                                                      frq=frq, 
                                                      causaleff=causaleff)
        else:
            truebeta = pd.read_table('%s.full'%(label), delim_whitespace=True) 
                                   #header=None, names=['SNP', 'Allele', 'beta'])
            geneff = pd.read_table('%s.profile'%(label), delim_whitespace=True)
            validsnpfile = '%s.totalsnps' % label
        # Generate the subset bfile
        subset = '%s --bfile %s --extract %s --make-bed --out %s'
        execute(subset % (plinkexe, bed, validsnpfile, label))  
        bed = os.path.join(os.getcwd(), label)
        if splits:
            train_test(label, splits=splits)        
        if not os.path.isfile('%s.prs_pheno.gz'%(label)):
            reference = liabilities(label, h2, ncausal, geneff)
        else:
            reference = pd.read_table('%s.prs_pheno.gz'%(label), sep='\t')
        
        if not os.path.isfile('%s_gwas.assoc.linear' % (label)):
            gwas = PlinkGWAS(plinkexe, bed, label, nosex=True, threads=thr, 
                             maxmem=mem, validate=splits, 
                             validsnpsfile=validsnpfile)
        else:
            gwas = pd.read_table('%s_gwas.assoc.linear' % (label), 
                                 delim_whitespace=True)
        merged = gwas.merge(truebeta, on='SNP')
        boole = merged.BETA.notnull()
        slope, intercept, r2, p_value, std_err = stats.linregress(
            merged.beta[boole], merged.BETA[boole])    
        ax = merged.plot.scatter(x='beta', y='BETA', s=2, alpha=0.5)
        ax.add_artist(AnchoredText('$R^{2} = %.3f $' % r2, 2))  
        plt.tight_layout()    
        plt.savefig('%s_truebetavsinferred.%s' % (label, ext))
        plt.close()
        bestr2 = stats.linregress(reference.PHENO, reference.gen_eff)[2]**2
        with open('pheno.pickle', 'wb') as f:
            pickle.dump((gwas, truebeta, bestr2, bed), f)
    return gwas, truebeta, bestr2, bed
  
def simulate_clump_ref(args, ref, f1, pthresh, split=False):
    if args.debug:
        rstep = 0.4
    else:
        rstep = 0.1
    if os.path.isfile('refclump.pickle'):
        with open('refclump.pickle', 'rb') as f:
            pick = pickle.load(f)
            causals, causalfn, truebeta2, gwas2, resE, bestr2ref = pick
    else:
        if not os.path.isfile('PhenoVsPRS_%s.png' % ref):
            ## Simulate phenotypes
            gwas2, truebeta2, bestr2ref, refbed = simulate_phenos(
                args.reference, ref, args.plinkexe, args.ncausal, args.h2, 
                args.threads, args.maxmem, frq=f1, ext=args.quality, 
                splits=split)
            args.reference = refbed
            causalfn = '%s.score'%(ref)
            causals = pd.read_table(causalfn, delim_whitespace=True,
                                    header=None, names=['SNP', 'A1', 'True_beta'
                                                        ])
            ## PpT
            PpTE = PplusT(ref, args.plinkexe, ref, 250, 
                         '%s_gwas.assoc.linear' % (ref), '%s.pheno' % (ref),
                         customPrange=pthresh, clean=False, validate=split != 
                         False, rstep=0.2)
            resE = PpTE.results    
            bestclumpedE = resE.nlargest(1,'pR2').File.iloc[0]
            profilefnE = '%s.profile' % bestclumpedE
            #clumpE = clumped('%s.score' % bestclumpedE)
            #cSNPsE = clumpE.SNP
            #topE = cSNPsE.shape[0]
            plotPRSvsPheno(profilefnE, label=ref, plottype=args.quality)
        else:
            resE = pd.read_table('%s.results' % ref, sep='\t')
            bestclumpedE = resE.nlargest(1,'pR2').File.iloc[0]
            profilefnE = '%s.profile' % bestclumpedE
            #clumpE = clumped('%s.score' % bestclumpedE)
            #cSNPsE = clumpE.SNP
            #topE = cSNPsE.shape[0]
            gwas2 = pd.read_table('%s_gwas.assoc.linear' % (ref), 
                                  delim_whitespace=True)
            truebeta2 = pd.read_table('%s.truebeta' % (ref), 
                                      delim_whitespace=True)
            causalfn = '%s.score'%(ref)
            causals = pd.read_table(causalfn, delim_whitespace=True,header=None, 
                                    names=['SNP', 'A1', 'True_beta']) 
        with open('refclump.pickle', 'wb') as f:
            obj = (causals, causalfn, truebeta2, gwas2, resE, bestr2ref)
            pickle.dump(obj, f)
    return causals, causalfn, truebeta2, gwas2, resE, bestr2ref
    
def main(args):   
    if args.debug:
        pthresh=('1.0,0.5,0.1,10E-3,10E-7') 
        args.step=10
    else:
        pthresh=('1.0,0.8,0.5,0.4,0.3,0.2,0.1,0.08,0.05,0.02,0.01,10E-3,10E-4,'
                 '10E-5,10E-6,10E-7,10E-8')           
    cwd = os.getcwd()
    ref, tar = args.labels
    if not os.path.isdir(ref):
        os.mkdir(ref)
    if not os.path.isdir(tar):
        os.mkdir(tar)
    f1, f2 = read_freqs(args.reference, ref, args.target, tar, args.plinkexe)
    allsnps= f2.SNP
    os.chdir(ref)
    causals, causalfn, truebeta2, gwas2, resE, bestr2ref = simulate_clump_ref(
        args, ref, f1, pthresh, split=args.split)
    gwasfn = os.path.join(cwd, ref, '%s_gwas.assoc.linear' % (ref))
    os.chdir(os.path.join(cwd, tar))
    ceff = truebeta2.loc[:, ['SNP','eff']]
    gwas1, truebeta1, bestr2tar, tarbed  = simulate_phenos(
        args.target, tar, args.plinkexe, args.ncausal, args.h2, args.threads, 
        args.maxmem,snps=truebeta2.SNP, frq=f2, causaleff=ceff, 
        ext=args.quality, splits=args.split)
    args.target = tarbed
    # perform P + T on target genotype with european sumstats
    if os.path.isfile('tarppt.pickle'):
        with open('tarppt.pickle', 'rb') as f:
            PpT = pickle.load(f)
    else:
        PpT = PplusT(args.target, args.plinkexe, '%s-%s' % (tar, ref), 250, 
                     '../%s/%s_gwas.assoc.linear' % (ref, ref), 
                     '%s.pheno' % (tar), customPrange=pthresh, 
                     validate=args.split, rstep=0.4 if args.debug else 0.1)
        with open('tarppt.pickle', 'wb') as f:
            pickle.dump(PpT, f)
    for f in glob('*.nosex'):
        os.remove(f)
    res = PpT.results    
    bestclumped = res.nlargest(1,'pR2').File.iloc[0]
    plotPRSvsPheno('%s.profile' % bestclumped, label=tar, pheno='%s.pheno' %tar, 
                   plottype=args.quality)        
    fn = '%s-%s' % (tar, ref)
    os.chdir(cwd)
    cotags = pd.read_table(args.cotagfn, sep='\t')
    cotags = cotags[cotags.SNP.isin(f2.SNP)]
    betaBeta(args.prefix, cotags, gwas1, gwas2, 10, [tar,ref], causals)
    if not os.path.isdir('Null'):
        os.mkdir('Null')
    os.chdir('Null')
    gwas = gwas2.merge(cotags, on='SNP')
    phenofn = os.path.join(cwd, tar, '%s.pheno' % tar)
    # Cotagging
    sortedcot, beforetail = smartcotagsort(args.prefix, cotags)
    # Tagging Target
    sortedtagT, beforetailTT = smartcotagsort(args.prefix, cotags, 
                                            column='Tagging %s' % tar)
    # Tagging Reference
    sortedtagR, beforetailTR = smartcotagsort(args.prefix, cotags,
                                              column='Tagging %s' % ref) 
    # Process clump if required
    if args.sortedclump is None:
        ppts = [(res, tar, 'c'), (resE, ref, 'm')]
        clum = None
    else:
        ppts = os.path.join(cwd, tar, bestclumped)
        if args.sortedclump != 'auto':
            clum = parse_sort_clump(args.sortedclump, allsnps)
        else:
            fil = resE.nlargest(1, 'pR2').File.iloc[0]
            refppt = os.path.join(cwd, ref, fil)
            clumtar = parse_sort_clump(args.sortedclump, allsnps, ppt=ppts)
            clumref = parse_sort_clump(args.sortedclump, allsnps, ppt=refppt)
            clum = [(clumref, ref), (clumtar, tar)]
        ppts = [(res, tar, 'c'), (resE, ref, 'm')]
    if args.qr:
        merge, col = prunebypercentage_qr(args.prefix, args.target, gwasfn, 
                                          phenofn, sortedcot, sortedtagT, 
                                          sortedtagR, args.plinkexe, 
                                          clumped=clum, step=args.step)      
    else:
        merge, col = prunebypercentage(args.prefix, args.target,gwasfn, phenofn, 
                                   sortedcot, sortedtagT, sortedtagR, 
                                   args.plinkexe, clumped=clum,
                                   step=args.step) 
    
    plotit(args.prefix, merge, col, args.labels, ppt=ppts,plottype=args.quality,
           hline=bestr2ref)
    plotit(args.prefix+'_pval', merge, '-log(P)', args.labels, ppt=ppts, 
           hline=bestr2ref, plottype=args.quality) 
    plotit(args.prefix+'_rval', merge, r'$R^{2}$', args.labels, ppt=ppts, 
           plottype=args.quality, hline=bestr2ref)
    return bestr2ref, args.reference, args.target       

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prefix', help='prefix for outputs', 
                        required=True)    
    parser.add_argument('-R', '--reference', help=('Bed fileset prefix of the re'
                                                  'ference population. This is '
                                                  'usually European'))
    parser.add_argument('-T', '--target', help=('Bed fileset prefix of the targ'
                                                'et population. This is usually'
                                                ' African'))
    parser.add_argument('-L', '--labels', help=('Labels for the population in t'
                                                'he same order as the reference'
                                                ' and target. This option has t'
                                                'o be passed twice, once with e'
                                                'ach of the populations'),
                                          action='append')
    parser.add_argument('-d', '--cotagfn', help=('Filename tsv with cotag '
                                                 'results'), required=True) 
    parser.add_argument('-c', '--ncausal', help='number of causal variants', 
                        default=200, type=int)     
    parser.add_argument('-H', '--h2', help='Heritability', default=0.66, 
                        type=float)    
    parser.add_argument('-P', '--plinkexe')
    parser.add_argument('-t', '--threads', default=False, action='store')
    parser.add_argument('-M', '--maxmem', default=False, action='store')    
    parser.add_argument('-s', '--step', help=('Step in the percentage range to'
                        ' explore. By deafult is 1'), default=1, type=float)
    parser.add_argument('-q', '--quality', help=('type of plots (png, pdf)'), 
                        default='png')  
    parser.add_argument('-v', '--split', help='number of splits for validation',
                        default=0, type=int)  
    parser.add_argument('-S', '--sortedclump', help='use clump file instead of'
                        ' res', default=None)   
    parser.add_argument('-D', '--debug', help='debug settings', 
                        action='store_true', default=False)     
    parser.add_argument('-Q', '--qr', help='Use q-range', default=False, 
                        action='store_true')   
    args = parser.parse_args()
    main(args)        
