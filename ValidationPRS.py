'''
ValidationPRS.py

Split the input genotypic data into a validation and train set, run GWAS on
training, and compute a R2 and Pt thresholded PRS evaluating on the validation
set
'''
# # Imports ####################################################################
import argparse
from CrossValidatePlink import createFolds #createFolds(fn, plinkexe, nfolds=10)
import pandas as pd
from subprocess import Popen
from glob import glob as G
# ##############################################################################

# # Constants ##################################################################
merg = '%s --bfile %s --merge-list %s --allow-no-sex --keep-allele-order \
--pheno %s --chr-set 22 --make-bed --out %s'
sexi = '%s --bfile %s --snps-only just-acgt --check-sex --keep-allele-order \
--out %s'
hetm = '%s --bfile %s --missing --het --keep-allele-order --out %s'
miss = '%s --bfile %s --allow-no-sex --keep-allele-order --geno 0.01 \
--mind 0.01 --out %s'
mafs = '%s --bfile %s --allow-no-sex --keep-allele-order --maf 0.01 --out %s'
prun = ''


# ##############################################################################

# # Functions ##################################################################
#----------------------------------------------------------------------
def isCategorical(phenocolumn):
    """
    Check if a column with the phenotype is categorical (I'll check up to 
    trinary, but would be binary for the case/control situations). Returns a 
    boolean.
    
    :param :class`pandas.Series` phenocolumn: series with the phenotype values
    """
    ## drop missing NA, nan, -9
    s = phenocolumn.dropna() ## Null values a.k.a nan
    s = s[((s != -9) & (s != '-9') & (s != 'NA'))]
    uni = s.unique()
    if len(uni) <= 3:
        ## less than 3 unique values, assumed categorical
        return True
    else:
        ## more than 3 unique values, assumed continous
        return False
    
#----------------------------------------------------------------------
def PED2BED(prefix, path, plinkexe):
    """
    Convert all chromosomes in a ped file to a single bed file
    
    :param str prefix: prefix for output
    :param str path: path to the individual chr{i}.ped files
    :param str plinkexe: path to plink executable (including executable's name)
    """
    files = G('%s/*.ped'%(path.strip('/')))
    with open('mergelist.txt','w') as M:
        for ped in files:
            M.write('%s %s.fam'%(ped, ped[:ped.find('.')] ))
    line = '%s --merge-list mergelist.txt --make-bed -out %s'
    merge = Popen(line%(plinkexe, prefix), shell=True)
        
        
#----------------------------------------------------------------------
def QC(pref, phenofile, plinkexe):
    """
    Run QC pipeline. This should be done before CV
    """
    ## Get missing or problematic sex
    missingsex = Popen(sexi%(plinkexe, pref, pref), shell=True)
    missingsex.communicate()  
    grep = Popen('grep PROBLEM %s.sexcheck > %s.sexprobs'%(pref,pref), 
                 shell=True)
    grep.communicate()
    sexp = pd.read_table('%s.sexprobs'%(pref), delim_whitespace=True)
    
                       
                
# ##############################################################################

## QC the input file

## Create train and test, trains have to merge the folds
folds = createFolds(args.fn, args.plinkexe, phenofile=args.phenofile, 
                    nfolds=args.folds)
## Run GWAS on train

## Compute PRS on R2 and Pt thresholds, evaluating on validation set

## Perform Alkes' strategy 


if __name__ == '__main__':
    ## define CLA
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prefix', help='prefix of input bed/bim/fam', 
                        required=True)
    parser.add_argument('-f', '--folds', help='number of folds to be performed',
                        default=10, required=True) 
    parser.add_argument('-n', '--plinkexe',  help='Path to plink executable',
                        required=True)
    parser.add_argument('-P', '--phenofile', help='Path to phenotype file',
                        default=None)
    parser.add_argument('-s', '--Stratified', help='Make stratified CV based on\
    cases and controls', default=True, action='store_false')
    parser.add_argument('-t', '--Ped2bed', help='Take individual chromosomes in\
    ped format and translate them into a single BED', default=False, 
                                           action='store_true')    
    args = parser.parse_args()
    main(args)