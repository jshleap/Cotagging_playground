'''
Pipeline for GWAS
'''
## Imports #####################################################################
import argparse, os, time
import pandas as pd
from subprocess import Popen, PIPE
import numpy as np
from joblib import Parallel, delayed
try:
    import matplotlib
    matplotlib.style.use('ggplot')
    import matplotlib.pyplot as plt
except:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
#from dotproductV2 import norm
################################################################################
## Constants ###################################################################

################################################################################
## Functions ###################################################################


# ----------------------------------------------------------------------
def execline(line):
    """
    execute a line with system
    """
    exe = Popen(line, shell=True)
    exe.communicate()
    time.sleep(5)


# ----------------------------------------------------------------------
def norm(array, a, b):
    '''
    normalize an array between a and b
    '''
    ## normilize 0-1
    rang = max(array) - min(array)
    A = (array - min(array)) / rang
    ## scale
    range2 = b - a
    return (A * range2) + a    


# ----------------------------------------------------------------------
def processFAM(famfile, phefile):
    """
    Write a new fam file including the information in <phefile> in the phenotype
    information
    :param str famfile: FAM file name
    :param str phefile: Phenotype file name
    """
    newname = '%s_new.fam'%(famfile[:famfile.rfind('.fam')])
    ## read files
    famnames = ['FID', 'IID', 'father', 'mother', 'sex', 'pheno']
    fam = pd.read_table(famfile, header=None, names=famnames, 
                        delim_whitespace=True)
    phe = pd.read_table(phefile, delim_whitespace=True)
    phe = phe.rename(dict(zip(phe.columns,['FID', 'IID', 'pheno'])))
    ## merge them to assure matching
    merged = phe.merge(fam.loc[:,famnames[:-1]], on=['FID', 'IID'])
    merged = merged.loc[:, famnames]
    merged.to_csv(newname, header=False, index=False, sep=' ')
    return newname


# ----------------------------------------------------------------------
def ExeEIGEN(bedfile, prefix, executable, pcs, verbose=False):
    """
    Execute Eigenstrat
    :param str eigenstrat: Constant line set globablly but modifiable here
    :param str bedfile: Filename (including path) of the bed file
    :param str bimfile: Filename (including path) of the bim file
    :param str famfile: Filename (including path) of the fam file (phenotype incl)
    :param str prefix: prefix for outputs
    :param str executable: Eigenstrat executable with path
    :param int pcs: Number of PCs to be computed
    """
    sep = '-'*71
    ## for eigentrat string:
    ## 1) smartpca path and executable
    ## 2) genetic file (could be bed)
    ## 3) snp file (could be bim)
    ## 4) ind file (could be fam but with phenotype included)
    ## 5) prefix for output for pca
    ## 6) prefix for output for plot
    ## 7) prefix for output file of all eigenvalues
    ## 8) prefix for log file
    ## 9) number of components
    eigenstrat = '%s -i %s -a %s.snp -b %s.ind -o %s.pca -p %s.plot -e %s.eval '
    eigenstrat+= '-l %s.log -k %d'    
    eigenstrat = eigenstrat%(executable, bedfile, prefix, prefix, prefix, 
                             prefix, prefix, prefix, pcs)
    if verbose:
        print('    Executing EigenStrat...')
    exe = Popen(eigenstrat, shell=True)
    exe.communicate()
    if verbose:
        print('%s\n\n'%(sep))      
    return readEIGvecs('%s.pca'%(prefix), pcs)


# ----------------------------------------------------------------------
def PlinkGWAS(plinkexe, bedprefix, covarnames, outprefix, nosex=False,
              verbose=False, threads=False, maxmem=False):
    """
    Execute plink gwas. This assumes binary phenotype
    
    :param str plinkexe: Path and executable of plink
    :param str bedprefix: Prefix for the bed fileset
    :param str outprefix: Prefix for the outputs
    :param bool nosex: If option --allow-no-sex should be used
    :param str covar: Filename with the covariates to use
    :param str covarnames: column names in the covariate file to be used
    """
    sep = '-' * 71
    if verbose:
        print('%s\nPerforming logistic regression...'%(sep))    
    if os.path.isfile('%s_gwas.bed'%(outprefix)):
        if verbose:
            print('\tFileset %s_gwas already in path... moving on'%(outprefix))
    else:
        ## for plinkgwas string:
        ## 1) plink path and executable
        ## 2) prefix for the bed fileset
        ## 3) Name of covariate file
        ## 4) names of the columns to use in the covariance file separated by "," 
        ## or '-' if range
        ## 5) prefix for outputs
        plinkgwas = "%s --bfile %s --assoc fisher-midp --logistic --covar %s "
        plinkgwas+= "keep-pheno-on-missing-cov --covar-name %s --out %s_gwas "
        plinkgwas+= "--ci 0.95 --genotypic --tests all --keep-allele-order"# --all-pheno"
        if nosex:
            plinkgwas += ' --allow-no-sex'
        else:
            plinkgwas == ' --sex'
        if threads:
            plinkgwas += ' --threads %s'%(threads)
        if maxmem:
            plinkgwas += ' --memory %s'%(maxmem)
        covar = '%s.covs'%(bedprefix)
        plinkgwas = plinkgwas%(plinkexe, bedprefix, covar, covarnames, outprefix)
        print(plinkgwas)
        execline(plinkgwas)


# ----------------------------------------------------------------------
def ReadFAMnBIM(famfile, bimfile):
    """
    Read the FAM and BIM files
    
    :param str famfile: Filename of the FAM file
    :param str bimfile: Filename of the BIM file
    """
    fam = pd.read_table(famfile, delim_whitespace=True, header=None, names=[
        'FID','IID', 'mother', 'father', 'sex', 'phen'])
    bim = pd.read_table(bimfile, delim_whitespace=True, header=None, names=[
        'CHR', 'Var ID', 'Pos', 'BP', 'A1', 'A2']) 
    return fam, bim


# ----------------------------------------------------------------------
def SetupEIG(bedprefix, prefix, plinkexe, nosex=False, verbose=False):
    """
    Create the .ind and .snp files for eigenstrat
    
    :param str plinkexe: Path and executable of plink
    :param str bedprefix: Prefix for the bed fileset
    :param str prefix: Prefix for the outputs
    """
    sep = '-'*71
    if verbose:
        line = '%s\nEstimating population structure covariates using plink...\n'
        print(line%(sep))
        print("    Setting up the plink files for eigenstrat...")
    prune = '%s.prune.in'%(prefix)
    ## create a bed fileset without LD for PCA
    if os.path.isfile(prune):
        if verbose:
            print('\tFile %s exist, using it to extrant variants'%(prune))
        plink = '%s --bfile %s --out %s --extract %s --make-bed'
        if nosex:
            plink += ' --allow-no-sex'        
        plink = plink%(plinkexe, bedprefix, '%s_PCA'%(prefix), prune)
    else:
        if verbose:
            print('\tCreating an LD free file for eigenstrat')
        plink = ('%s --bfile %s --out %s --indep-pairwise 50 5 0.2 --keep-allel'
                 'e-order --make-bed')
        if nosex:
            plink += ' --allow-no-sex'         
        plink = plink%(plinkexe, bedprefix, '%s_PCA'%(prefix))
    exe = Popen(plink, shell=True)
    exe.communicate()
    famfile, bimfile = '%s_PCA.fam'%(prefix), '%s_PCA.bim'%(prefix)
    fam, bim = ReadFAMnBIM(famfile, bimfile)
    ## recode outputs
    if verbose:
        print('\tDecoding sex and phenotype')
    dic = {1:'M',2:'F',0:0}
    fam.sex = [dic[i] for i in fam.sex]
    dic = {2:'Case', 1:'Control', -9:'Missing', 0:'Missing'}
    fam.phen = [dic[i] for i in fam.phen]
    ind = fam.loc[:,['IID','sex', 'phen']]
    if verbose:
        print('\tWriting indiv and snp files to %s.ind and %s.snp'%(prefix, 
                                                                      prefix))    
    ind.to_csv('%s.ind'%(prefix), header=False, index=False, sep=' ')
    snp = bim.loc[:,['Var ID', 'CHR', 'Pos', 'BP', 'A1', 'A0']]
    snp.to_csv('%s.snp'%(prefix), header=False, index=False, sep=' ')
    touch = open('%s.pca.evec'%(prefix),'w')
    touch.close()


# ----------------------------------------------------------------------
def readEIGvecs(eigvec, npcs):
    """
    Read the eigenvector from eigenstrat
    
    :param str eigvec: Filename of the output of eigenstrat with eigenvectors
    """
    columns=["Sample"]
    ## determine the number of PCs
    columns = columns + ["PC%d"%(i) for i in range(1, npcs+1)] + ["Pop"]
    evecDat = pd.read_table(eigvec, header=None, comment='#', names=columns, 
                            delim_whitespace=True)
    return evecDat


# ----------------------------------------------------------------------
def QCnPHE(plinkexe, bedprefix, prefix, pheno, chromosome, relatedness=True,
           verbose=False, nosex=False, threads=False, maxmem=False, 
           parallel=False):
    """
    Perform QC and include the phenotyoe in a bed fileset. This assumes that
    a previous QC has been performed solving: 
    a) discordant sex
       plink --bfile <bedprefix> --check-sex --out <outprefix>
       grep PROBLEM <outprefix>.sexcheck > <outprefix>.sexprobs
       plink --bfile <bedprefix> --remove <outprefix>.sexprobs --out <bedrpefix>
    b) elevated missing data rates or outlying heterozygosity rate:
       plink --bfile <bedprefix> --missing --out <outprefix>
       plink --bfile <bedprefix> --het --out <outprefix>
       plot imiss-vs-het, select threshold and create an exlusion file
       plink --bfile <bedprefix> --remove <exclusionfile> --out <bedprefix>
    
    It also assumes that the varIDs are in RS
    
    :param str plinkexe: Path and executable of plink
    :param str bedprefix: Prefix for the bed fileset
    :param str prefix: Prefix for the outputs
    :param str pheno: Filename with phenotypes
    """
    logfile = '%s_QCDONE.txt'%(prefix)
    done = ''
    if os.path.isfile(logfile):
        with open(logfile) as M:
                done = M.read()
    sep = '-'*71
    if not 'plinkpheno' in done:
        if verbose:
            print('%s\nPerform QC and phenotype inclusion...\n'%(sep))
        ## For plinkpheno string (include phenotype, deal with missingness, maf
        ## and deviation from hwe):
        ## 1) plink path and executable
        ## 2) prefix for the bed fileset
        ## 3) out prefix
        ## 4) phenotype file
        #
        ## Filter MAFs less than 1%, extreme departures of HWE, missing call  
        ## rates exceeding 1%, and filters out all samples with missing 
        ## phenotypes.
        plinkpheno = '%s --bfile %s --out %s --pheno %s --make-bed --geno 0.01 '
        plinkpheno+= '--keep-allele-order --maf 0.01 --hwe 0.0001 midp --prune --1'
        if nosex:
            plinkpheno += ' --allow-no-sex'         
        plinkpheno = plinkpheno%(plinkexe, bedprefix, prefix, pheno)
        if chromosome:
            plinkpheno += ' --chr %s'%(chromosome)
        if verbose:
            print('    Filtering MAF (1%), HWE (1E-4), and missing phenotypes\n')
        if threads:
            plinkpheno += ' --threads %s'%(threads)
        if maxmem:
            plinkpheno += ' --memory %s'%(maxmem)    
        execline(plinkpheno)
        with open(logfile, 'a') as L:
            L.write('plinkpheno done ...\n')
        #
    if relatedness:
        if not 'LDprune' in done:
            if verbose:
                print('    Creating subset with no LD for IBD estimation')         
            ## Create a subset with LD pruned SNPs for IBD estimation
            LDprune = ('%s --bfile %s --indep-pairwise 50 10 0.2 --keep-allele-'
                       'order --out %s')
            LDprune = LDprune%(plinkexe, prefix, prefix)
            if nosex:
                LDprune += ' --allow-no-sex'        
            if threads:
                LDprune += ' --threads %s'%(threads)
            if maxmem:
                LDprune += ' --memory %s'%(maxmem)       
            execline(LDprune)
            #out = execline(LDprune)
            #if verbose:
                #print(out)
            with open(logfile, 'a') as L:
                    L.write('LDprune done ...\n')    
            #
        if not 'plinkIBD' in done:
            ## For plinkIBD string (generate criptic relatedness up to 5th degree)
            extra=''
            prune = '%s.prune.in'%(prefix)
            plinkIBD = ('%s --bfile %s --keep-allele-order --extract %s --out '
                        '%s --genome --min 0.05')
            plinkIBD = plinkIBD%(plinkexe, prefix, prune, prefix)
            if nosex:
                extra += ' --allow-no-sex'
            if threads:
                extra += ' --threads %s'%(threads)
            if maxmem:
                extra += ' --memory %s'%(maxmem)  
            if verbose:
                    print('    Estimating IBDs with PI_HAT >= 0.15')
            if not os.path.isfile('%s.genome'%(prefix)):
                if parallel:
                    for i in range(1,int(parallel)+1):
                        extra2 = '%s --parallel %d %s'%(extra, i, parallel)
                        execline('%s %s'%(plinkIBD, extra2))
                        #out = execline('%s %s'%(plinkIBD, extra2))
                        #if verbose:
                            #print(out)
                    cat = 'cat %s.genome.* > %s.genome; rm %s.genome.*'%(prefix, 
                                                                         prefix, 
                                                                         prefix)
                    execline(cat)
                else:
                    plinkIBD = plinkIBD + extra
                    execline(plinkIBD)
    
            processGenome('%s.genome'%(prefix), plinkexe, prefix, prefix, 
                          nosex=nosex, verbose=verbose, threads=threads)
            with open(logfile, 'a') as L:
                    L.write('plinkIBD done ...')    
        if verbose:
            print('%s\n\n'%(sep))    


# ----------------------------------------------------------------------
def prcgenthreads(chunk):
    """
    process the genome file by chunks and return a sub dataframe
    """
    rels = chunk[chunk.PI_HAT > 0.16]
    IIDs = pd.unique(pd.concat([rels.IID1, rels.IID2]))
    rels = rels[(rels.IID1.isin(IIDs) | rels.IID2.isin(IIDs))]
    return rels.loc[:, ['FID2', 'IID2']]


# ----------------------------------------------------------------------
def processGenome(genomefile, plinkexe, bedprefix, prefix, nosex=False, 
                  verbose=False, threads=-1):
    """
    Read the genome file (as produced by plink option --genome) and removes
    one of the individuals that have over 0.05 pi_hat (%5 inbreeding coeff.)
    
    :param str genomefile: filename of the genome
    :param str plinkexe: Path and executable of plink
    :param str bedprefix: Prefix for the bed fileset
    :param str prefix: Prefix for the outputs
    """
    if not threads:
        threads= - 1
    if verbose:
        print('    Processing %s file'%(genomefile))     
    gen = pd.read_table(genomefile, delim_whitespace=True, iterator=True,
                        chunksize=10000)
    rels = Parallel(n_jobs=int(threads))(delayed(prcgenthreads)(ch) for ch in 
                                         gen)
    rels = pd.concat(rels)    
    excludefile = '%s_relateds.toexclude'%(prefix)
    rels.drop_duplicates().to_csv(excludefile, sep=' ', header=False, 
                                       index=False)
    if verbose:
            print('\t %d Individuals excluded as relative of other individuals'%
                  (rels.shape[1]))     
    plremov = ('%s --bfile %s --remove %s_relateds.toexclude --make-bed --keep-'
               'allele-oder --out %s')
    if nosex:
        plremov += ' --allow-no-sex'
    plremov = plremov%(plinkexe, bedprefix, prefix, prefix)
    if verbose:
        print('    Removing individuals from file %s'%(excludefile))     
    execline(plremov)


# ----------------------------------------------------------------------
def processCovariates(prefix, pca, covs, verbose=False):
    """
    Read all covariates and pooled them in a single covariate file. Covariate 
    files must contain FID and IID columns and as names
    :param str pca: filename of the eigenvectors outputted by eigenstrat or simi
    :param lisr covs: list with filenames of non-pc covariates
    """
    sep = '-'*71
    if verbose:
        print('%s\nProcessing covariates...'%(sep))
    if isinstance(pca, str):
        PC = pd.read_table(pca, header=None, comment='#', delim_whitespace=True)
    else:
        ## assume is a pandas dataframe 
        PC = pca
    if covs:
        CO = pd.read_table(covs[0], delim_whitespace=True)
        for fn in covs[1:]:
            CO.merge(pd.read_table(fn, delim_whitespace=True), on=['FID', 'IID']
                     )
        COVS = PC.merge(CO, on=['FID', 'IID'])
    else:
        COVS = PC
    cols = list(COVS.columns)
    cols.pop(cols.index('FID'))
    cols.pop(cols.index('IID'))
    COVS.loc[:,['FID', 'IID'] + cols].to_csv('%s.covs'%(prefix), sep='\t', 
                                             index=False)
    if 'SOL' in cols:
        cols.pop(cols.index('SOL'))
    if verbose:
        print('%s\n\n'%(sep))      
    return ' '.join(cols)
    

# ----------------------------------------------------------------------
def renameVars(bedprefix, conversionfile, plinkexe, prefix, chromosome,
               nosex=False, verbose=False, threads=False, maxmem=False):
    """
    Rename all variants by the conversion file
    
    :param str bedprefix: Prefix for the bed fileset
    :param str conversionfile: File name of the conversion file
    :param str plinkexe: Path and executable of plink
    :param str prefix: Prefix for the outputs
    """
    sep = '-'*71
    ## For plinkconver str:
    ## 1) plink path and executable
    ## 2) prefix for the bed fileset
    ## 3) out prefix
    ## 4) conversion file name       
    plinkconver = ('%s --bfile %s --out %s --update-name %s --keep-allele-order'
                   ' --make-bed')%(plinkexe, bedprefix, prefix, conversionfile)
    if chromosome:
        plinkconver += ' --chr %s'%(chromosome)
    if threads:
        plinkconver += ' --threads %s'%(threads)
    if maxmem:
        plinkconver += ' --memory %s'%(maxmem) 
    if nosex:
        plinkconver += ' --allow-no-sex'
    if verbose:
        print('%s\nRenaming variant IDs based in %s...\n'%(sep, conversionfile))
    execline(plinkconver)
    #line = execline(plinkconver)
    #if verbose: 
        #print(line)
    with open('%s_rename.done'%(prefix), 'w') as R:
        R.write('Rename has been done')
    if verbose:
            print('%s\n\n'%(sep))   


#----------------------------------------------------------------------
def ManhattanPlot(df, prefix, grayscale=False, save=True, chromosome='All'):
    """
    Create a manhatanplot from dataframe. The data frame has to contain the 
    following columns:
       1) PVAL : pvalue
       2) CHR : Chromosome
       3) SNP : the SNP id
    
    :param :class:pandas.DataFrame df: a pandas dataframe with the info
    :param str prefix: prefix for outputs
    :param bool save: wether to save the plot to <prefix>.pdf or to return axis
    """
    # -log_10(pvalue)
    df['minuslog10pvalue'] = -np.log10(pd.to_numeric(df.PVAL))
    df.CHR = df.CHR.astype('category')
    #df.CHR = df.CHR.cat.set_categories(['ch-%i' % i for i in 
    #                                    range(1,len(df.CHR)+1)], ordered=True)
    df = df.sort_values(['CHR', 'BP'])
    df['ind'] = range(len(df))
    if chromosome != 'All':
        df = df[df.CHR == int(chromosome)]
    df_grouped = df.groupby(('CHR'))
    ax = plt.subplot()
    if grayscale:
        colors=['0.10', '0.50']
    else:
        colors = ['red','green','blue', 'yellow']
    x_labels = []
    x_labels_pos = []
    for num, (name, group) in enumerate(df_grouped):
        if group.empty:
            continue
        if grayscale:
            group.plot.scatter(x='ind', y='minuslog10pvalue', c=colors[
                num % len(colors)], ax=ax, legend=False, s=3, edgecolor='none')
        else:           
            group.plot.scatter(x='ind', y='minuslog10pvalue', color=colors[
                num % len(colors)], ax=ax, legend=False, s=3, edgecolor='none')
        x_labels.append(name)
        x_labels_pos.append((group['ind'].iloc[-1] - (group['ind'].iloc[-1] - 
                                                      group['ind'].iloc[0])/2))
    if save:
        ax.set_xticks(x_labels_pos)
        ax.set_xticklabels(x_labels)
        #ax.set_xlim([0, len(df)])
        #ax.set_ylim([0, 3.5])
        ax.set_xlabel('Chromosome')
        ax.set_ylabel('-Log(P-value)')        
        plt.savefig('%s_manhattan.pdf'%(prefix))
    else:
        return ax, df, x_labels_pos, x_labels


#----------------------------------------------------------------------
def plinkEIGEN(bedprefix, prefix, plinkexe, npc, verbose=False, threads=False, 
               nosex=False, maxmem=False):
    """
    Run clustering, PCA, and MDS on the data using plink
    """
    sep = '-'*71
    if verbose:
        line = '%s\nEstimating population structure covariates using plink...\n'
        print(line%(sep))
    if os.path.isfile('%s.eigenvec'%(prefix)):
        if verbose:
            print('    File %s.eigenvec exist. Using it ...'%(prefix))
    else:
        pline = '%s --bfile %s --cluster cc --pca %d '#--read-genome %s.genome
        pline += '--mds-plot %d --out %s --keep-allele-order'
        if nosex:
            pline += ' --allow-no-sex'
        pline = pline%(plinkexe, bedprefix, npc, npc, prefix)
        if threads:
            pline += ' --threads %s'%(threads)
        if maxmem:
            pline += ' --memory %s'%(maxmem)
            ## For plinkIBD string (generate criptic relatedness up to 5th degree)
        execline(pline)
        #out = execline(pline)
        #if verbose: 
            #print(out)
    covar = '%s.covs'%(bedprefix) 
    ## read and merge results
    pcacols = ['FID','IID']
    covsnames = ["PC%d"%(i) for i in range(1, npc+1)]
    pca = pd.read_table('%s.eigenvec'%(prefix), delim_whitespace=True, 
                        header=None, names=pcacols + covsnames)
    mds = pd.read_table('%s.mds'%(prefix), delim_whitespace=True) 
    mdscols = list(mds.columns)
    mdscols.pop(mdscols.index('SOL'))
    mer = pca.merge(mds, on=['FID', 'IID'])
    mer = mer.loc[:,covsnames + mdscols]
    if verbose:
        print('%s\n\n'%(sep))    
    return mer


#----------------------------------------------------------------------
def plotit(prefix, chrom, verbose=False):
    """
    Process logistic regression output and make a Manhattan plot
    """
    df = pd.read_table('%s_gwas.assoc.logistic'%(prefix), delim_whitespace=True)
    df = df.rename(columns={'P':'PVAL'})
    df = df[df.TEST == 'ADD'].dropna()
    ManhattanPlot(df, prefix, grayscale=True, chromosome=chrom)


#----------------------------------------------------------------------
def main(args):
    """
    execute the pipeline
    
    :param :class:argsparse args: command line argument
    """
    if os.path.isfile('%s.bed'%(args.prefix)):
        bed = args.prefix
    else:
        bed = args.bed
    boolean = os.path.isfile('%s_rename.done'%(args.prefix))
    if args.conversion and not boolean:
        ## If variables are not in rsID, provide the conversion (a.k.a map) file
        ## and convert it to rsIDs.
        renameVars(args.bed, args.conversion, args.plinkexe, args.prefix, 
                   args.chr, nosex=args.nosex, verbose=args.verbose, 
                   threads=args.threads, maxmem=args.maxmem)
        bed = args.prefix

    ## Do some required QC
    if not os.path.isfile('%s_relateds.toexclude'%(args.prefix)):
        QCnPHE(args.plinkexe, bed, args.prefix, args.phe, args.chr, 
               relatedness=args.relateds, nosex=args.nosex, verbose=args.verbose, 
               threads=args.threads, maxmem=args.maxmem, parallel=args.parallel)
        bed = args.prefix
    ## Estimate population stratification with EigenStrat
    if 'plink' in args.eigexe:
        ## If plink in eigen, perform a plink-based popstruct 
        pca = plinkEIGEN(args.prefix, args.prefix, args.plinkexe, args.npc,
                         nosex=args.nosex, verbose=args.verbose, 
                         threads=args.threads, maxmem=args.maxmem)
    else:
        SetupEIG(bed, args.prefix, args.plinkexe, nosex=args.nosex,
                 verbose=args.verbose)
        pca = ExeEIGEN('%s_PCA'%(args.prefix), args.prefix, args.eigexe, 
                       args.npc, verbose=args.verbose)
    covarnames = processCovariates(args.prefix, pca, args.nonPCcov)
    if not os.path.isfile('%s_gwas.assoc.logistic'%(args.prefix)):
        PlinkGWAS(args.plinkexe, bed, covarnames, args.prefix, nosex=args.nosex, 
                  threads=args.threads, maxmem=args.maxmem)
    if args.plot:
        if args.chr:
            chromosome = args.chr
        else:
            chromosome = 'All' 
        plotit(args.prefix, chromosome, verbose=args.verbose)
    
    
    
    
################################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prefix', help='prefix for outputs',
                        required=True)    
    parser.add_argument('-b', '--bed', help='bed fileset path and prefix',
                        required=True)
    parser.add_argument('-P', '--phe', help='phenotype file path and name. If \
    used will replace the values in the fam file', default='')      
    parser.add_argument('-k', '--npc', help='number of pcs to be computed',
                        default=10, type=int)     
    parser.add_argument('-e', '--eigexe', help='path and executable of EIG',
                        required=True)
    parser.add_argument('-n', '--plinkexe', help='path and executable of Plink',
                        required=True)
    parser.add_argument('-c', '--nonPCcov', help='Non-PC covariates file(s)',
                        action='append',  default=[])
    parser.add_argument('-C', '--conversion', help='covert variant names based\
    on this file', default='')
    parser.add_argument('-v', '--verbose', help='Be loud, annoying', 
                        action='store_true')    
    parser.add_argument('-m', '--chr', help='chromosome', default='')
    parser.add_argument('-l', '--plot', help='To plot Manhattan', default=False, 
                        action='store_true')
    parser.add_argument('-t', '--threads', help='Max threads to use. By default\
    all available', default=False)    
    parser.add_argument('-M', '--maxmem', help='Max memory to use in megabites.\
    By default is half of all available RAM.', default=False)    
    parser.add_argument('-R', '--parallel', help='Use plink parallel for the \
    memory expensive steps. You need to provide the number of chunks', 
                                            default=False)
    parser.add_argument('-a', '--nosex', help='Use plink option --allow-no-sex', 
                        default=False, action='store_true')
    parser.add_argument('-r', '--relateds', help='Dont do IBD filtering', 
                            default=False, action='store_false')    
    args = parser.parse_args()
    main(args)