"""
from summary statistics and a merge of inputed and QCed data, runs PRSice and 
gets the polygenic risk scores

Summary statistics file (SummaryStats): 
A tab delimited file with at least the PRSice required headers: "SNP, A1, OR or 
BETA, P - containing SNP name (eg. rs number), effect allele (A1), effect size 
estimate as an odds ratio (binary phenotype) or continuous effect beta 
(continuous phenotype) and P-value for association."

Base or genotype files (GenotypeFile):
A plink formated binary files, including the BED, BIM and FAM files

Phenotype file (PhenotypeFile):
A phenotype file with at least IID (individual identifyer) and the corresponding
phenotype.

Individuals to keep file (KeepFile):
file with the IID and FID (family ID)


"""
__author__ = 'jshleap'

import argparse, os
import pandas as pd
from subprocess import Popen

# CONSTANTS
PRSiceCommand = """R --file=%s -q --args \
plink %s \
base %s \
target %s \
pheno.file %s \
figname %s \
clump.r2 %f \
covary F \
best.thresh.on.bar T \
report.individual.scores T \
quantiles T \
for.meta T \
remove.mhc T \
slower %f \
sinc %f \
supper %f"""


def check_pheno(phenofile, fam, simulation):
    ''' Check if phenotype in right format, and correct it if not '''
    ## Tell where we are
    line = 'Checking phenotype from file %s#'%(args.PhenotypeFile)
    ast = '#'*len(line)
    print('%s\n%s\n%s\n\n'%(ast,line,ast))
    ## read phenotype
    print('Reading file')
    if not simulation:
        pheno = pd.read_table(phenofile, delim_whitespace=True)
    else:
        pheno = pd.read_table(phenofile, delim_whitespace=True, header=None,
                              names=['FID', 'IID', 'pheno'])
    ## check if it has the correct number of dimensions
    if (pheno.shape[1] != 3) and ('FID' not in pheno.columns):
        print('Fixing file')
        ## Get the fam file from the target
        fa = pd.read_table(fam, delim_whitespace=True, header=None)
        ## Subset for the IDs present in pheno (in case it differs)
        subs = fa[fa.loc[:,0].isin(pheno.IID)]
        subs = subs.rename(columns={0:'IID'})
        merged = subs.merge(pheno, on='IID')
        col = pheno.columns[pheno.columns != 'IID'][0]
        merged = merged.loc[:,['IID',1,col]]
        merged = merged.rename(columns={1:'FID'})
        #merged.to_csv(phenofile, sep=' ', index=False)
        pheno = merged
    else:
        print('Phenotype file is OK... continuing\n\n')
    
    ### Drop missing phenotypes
    print('Dropping missing phenotypes')
    phename = pheno.columns[(pheno.columns != 'IID') & (pheno.columns != 'FID')]
    idx = (pheno.loc[:,phename] != -9) & (~pheno.loc[:,phename].isnull())
    pheno = pheno.loc[idx.loc[:,phename[0]],:]
    try:
        pheno = pheno[pheno[:,phename] != 'NA']
    except:
        pheno = pheno
    pheno.to_csv(phenofile, sep=' ', index=False)
    return pheno

def process_coarse(fn):
    ''' With the PRScise reults, get the fine grain range'''
    ## Read the results
    prs1 = pd.read_table(fn, delim_whitespace=True, low_memory=True, dtype={
    'thresh':float, 'p.out':float, 'r2.out':float, 'nsnps':int, 
    'coefficient':float, 's.err':float})
    ## Deal with equally good r2
    nprs = pd.DataFrame()
    for r2, df in prs1.groupby('r2.out'):
        nprs = nprs.append(df.nsmallest(1,'thresh'))
    ## Get the top 5 entries
    L = nprs.nlargest(5,'r2.out')
    ## set the range for fine grain
    slower = float(L.nsmallest(1,'thresh').thresh)
    supper = float(L.nlargest(1,'thresh').thresh)
    sinc = (supper - slower)/500
    return slower, supper, sinc


def Edit_SumSats(SummStats, prefix, plinkstats, simulation):
    '''
    Edit the summary statistics file dropping all duplicates, missing p-values,
    and ambiguous alleles
    '''
    ## Tell it like it is
    line = '# Processing Summary Statistics from %s #'%(SummStats)
    ast = '#'*len(line)
    print('%s\n%s\n%s\n'%(ast,line,ast))
    nodups='%s_SummaryStats_nodups.tsv'%(prefix)
    ext =  prefix+'_snps.extract'
    ##check if done, and skip it if done
    if not os.path.isfile(nodups) and not os.path.isfile(ext): 
        ## read in the summary stats in lowmemory (aka. chunks)
        print('Reading file')
        if not plinkstats:
            summ = pd.read_table(SummStats, sep='\t', delim_whitespace=True,
                                 low_memory=True, dtype={
                                     'SNP':str, 'CHR':int, 'BP': float, 
                                     'GENPOS':float, 'A1':str,'A2':str, 
                                     'A1FREQ':float, 'F_MISS':float, 
                                     'BETA':float, 'SE':float, 'P':float})
        else:
            summ = pd.read_table(SummStats, delim_whitespace=True)
        ## Get rid of duplicates
        print('Dropping duplicates')
        subset = summ.drop_duplicates(subset='SNP')
        ## free some memory
        del summ
        ## make sure duplicates are trully dealt with
        subset = subset[~subset.duplicated(subset='SNP')]
        ## Also deal with duplicates SNPs (by position)
        
        ## Get rid of lines that might have the title repeated (this probably 
        ## would) fail anyhow from dtype above, but just in case
        subset = subset[~(subset.SNP == 'SNP')]
        ## Get rid of missing Pvalues and missing alleles (no trustworthy data)
        print('Dropping missing p-values')
        if simulation:
            cols = ['A1','P']
        else:
            cols = ['A1','A2','P']
        subset = subset.dropna(subset=cols)
        ## Drop unreal pvalues
        print('Dropping unreal pvalues')
        idx = subset.P <= 1
        subset = subset[idx]
        ## Deal with ambiguities in alleles (also not trustworthy data)
        if not simulation:
            print('Dropping ambiguities')
            subset = subset[[True if len(x) == 1 else False for x in subset.A1]]
            subset = subset[[True if len(y) == 1 else False for y in subset.A2]]        
            ## strand ambiguity
            print('Removing A/T and C/G SNPs for potential strand ambiguity')
            subset[~((subset.A1 == 'A') & (subset.A2 == 'T')) | 
                   ~((subset.A1 == 'T') & (subset.A2 == 'A')) | 
                   ~((subset.A1 == 'G') & (subset.A2 == 'C')) | 
                   ~((subset.A1 == 'C') & (subset.A2 == 'G'))]
        ## Write to file
        print('Writing clean file to %s'%(nodups))
        subset.to_csv(nodups,sep= '\t', index=False, na_rep='NA')    
            
        ## Get a file with the remaining SNPs to filter in
        print('Writing subset SNPs to %s'%(ext))
        subset.SNP.to_csv(ext, index=False) 
        ## free up memory del subset (this is probably dealt with by the GC but 
        ## just in case)
        del subset
        print('\n\n')
    else:
        print('Another run have been done!!! using the available files\n\n')
    return nodups, ext

def Subset_genotype(args, snp, pheno):
    '''
    Based on the keep and snp, filter genotype
    '''
    ## Tell it
    line = '# Subsetting the genotype data in %s with individuals from %s and '
    line += 'SNPs from %s #'
    line = line%(args.GenotypeFile, args.KeepFile, snp)
    ast = '#'*len(line)
    print('%s\n%s\n%s\n\n'%(ast,line,ast))
    
    ## from the keep file drop the missing phenotypes
    if args.KeepFile is False:
        keep = pheno
        name = '%s.kept'%args.prefix
    else:
        name = args.KeepFile
        keep = pd.read_table(args.KeepFile, delim_whitespace=True, header=None)
        keep = keep.rename(columns={0:'FID',1:'IID'})
    keep = keep.merge(pheno.loc[:,['FID','IID']], on=['FID','IID'])
    keep.to_csv(name, index=False, header=False)
    keep.loc[:,['FID','IID']].to_csv(name, index=False, header=False, sep=' ')
    
    ## include phenotype in fam
    #fam = pd.read_table('%s.fam'%(args.GenotypeFile), delim_whitespace=True,
    #                    header=None)
    #fam = fam.rename(columns={0:'FID', 1:'IID', 2:'Father', 3:'Mother', 4:'sex',
    #                          5:pheno.columns[-1]})
    #keep.rename(index=keep.IID,inplace=True)
    #pheno.rename(index=pheno.IID,inplace=True)
    
    if os.path.isfile(args.prefix+'.bed') and os.path.isfile(args.prefix+'.bim')\
       and os.path.isfile(args.prefix+'.fam'):
        print('Step already done, using files with prefix %s'%(args.prefix))
    else:
        if args.KeepFile:
            command = '%s -bfile %s --keep %s --extract %s --make-bed -out %s'
            command = command%(args.PlinkExe, args.GenotypeFile, args.KeepFile, 
                               snp, args.prefix)
        else:
            command = '%s -bfile %s --extract %s --make-bed -out %s'
            command = command%(args.PlinkExe, args.GenotypeFile, snp, 
                               args.prefix)        
        
        print ('executing plink with command:\n%s\n\n'%(command))
        
        plink = Popen(command, shell=True)
        plink.communicate()

  
def main(args, PRSiceComm):
    prefix = args.prefix
    if not os.path.isdir(prefix):
        os.mkdir(prefix)
    os.chdir(prefix)
    cmm = '%s_command.txt'%(prefix)
    if not os.path.isfile(cmm):
        with open(cmm,'w') as C:
            C.write(str(args))
        
    ## Edit Summary statistics such that it has no duplicates and no ambiguous
    ## alleles
    nodups, snps = Edit_SumSats(args.SummaryStats, args.prefix, args.plinkstats,
                                args.simulation)
    
    ## Check phenotype
    pheno = check_pheno(args.PhenotypeFile, args.GenotypeFile+'.fam', 
                        args.simulation)
    
    ## Plink filter individuals and SNPs
    Subset_genotype(args, snps, pheno)

    ## Run PRSice coarsed-grained
    line = '# Running coarse-grained PRSice #'
    ast = '#'*len(line)
    print('%s\n%s\n%s\n\n'%(ast,line,ast))      
    out = args.prefix+'_coarse'
    PRSiceCommand = PRSiceComm%(args.PRSiceExe, args.PlinkExe, nodups, 
                                args.prefix, args.PhenotypeFile, out, 
                                args.LDthreshold, 0, (0.5)/250, 0.5)
    prsice1 = Popen(PRSiceCommand, shell=True)
    print('executing %s'%(PRSiceCommand))
    prsice1.communicate()

    ## get results and prepare the fine grain range
    slower, supper, sinc = process_coarse('%s_RAW_RESULTS_DATA.txt'%(out))
    
    ## Run PRSice fine-grained
    line = '# Running fine-grained PRSice #'
    ast = '#'*len(line)
    print('%s\n%s\n%s\n\n'%(ast,line,ast))
    out2 = args.prefix + '_fine'
    prsice2 = Popen(PRSiceComm%(args.PRSiceExe, args.PlinkExe, nodups, 
                                args.prefix, args.PhenotypeFile, out2, 
                                args.LDthreshold, slower,sinc,supper), 
                    shell=True)
    
    prsice2.communicate()
    
if __name__ == '__main__':
    ## Define the path of the default plink and PRSice which is in abacus 
    ##(McGill Genome Center cluster)
    defplink = '/lb/project/gravel/hleap_projects/UKBB/PRSice_v1.25/plink_1.9_l'
    defplink += 'inux_160914'
    defprsice = '/lb/project/gravel/hleap_projects/UKBB/PRSice_v1.25/PRSice_v1.'
    defprsice += '25.R'
    ## define CLA
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prefix', help='prefix for outputs', 
                        required=True)
    parser.add_argument('-s', '--SummaryStats', help='Filename of the summary \
    statistics', required=True)    
    parser.add_argument('-g', '--GenotypeFile', help='Path and prefix of the \
    genotype files (bed, bim, fam) (A.K.A. target)', required=True)
    parser.add_argument('-P', '--PhenotypeFile', help='Path and filename of the\
    phenotype file', required=True)
    parser.add_argument('-k', '--KeepFile', help='Path and filename of the file\
    with the individuals to keep', default=False)    
    parser.add_argument('-n', '--PlinkExe', help='Path and executable file of \
    plink', default= defplink)
    parser.add_argument('-r', '--PRSiceExe', help='Path and executable file of \
    PRSice', default= defprsice)
    parser.add_argument('-l', '--LDthreshold', help='float of the R2 cut-off \
    for LD prunning', default=0.5, type=float)
    parser.add_argument('-d', '--plinkstats', help='Summary stats in plink format',
                        default=False, action='store_true')
    parser.add_argument('-S', '--simulation', help='is it a simulation?',
                        default=False, action='store_true')    
    args = parser.parse_args()
    main(args, PRSiceCommand)    
