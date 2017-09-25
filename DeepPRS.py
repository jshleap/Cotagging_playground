'''
Deep learning based PRS 
'''
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.cross_validation import train_test_split
from keras.layers import Input, Dense
from keras.models import Sequential, Model
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
import sys, os, argparse, gc
import pandas as pd



def read_LD(fn, maf_threshold=0.01):
    '''
    read the LD file as outputted by plink
    :param str fn: filename of the LD matrix (plink formatted)
    :param float maf_threshold: the cutoff of exclusion for mafs below or above
    1 - maf_threshold
    '''
    oth = 1 - maf_threshold
    df = pd.read_table(fn, delim_whitespace=True)
    ## Drop Nans
    df = df.dropna()
    ## Drop MAFs
    df = df[(df.MAF_A > maf_threshold) & (df.MAF_A < oth) & 
            (df.MAF_B > maf_threshold) & (df.MAF_B < oth)]
    return df

def read_summStats(fn):
    '''
    Read the summary statistics file in plink-compitible format. That means that
    must contain the following headers:
    1)'SNP' 2)'CHR' 3)'BP' 4)'GENPOS' 5)'A1' 6)'A2' 7)'A1FREQ' 8)'F_MISS' 
    9)'BETA'10)'SE' 11)P'
    
    :param str fn: filename of the summary statistics
    '''
    pref = fn[:fn.find('.')]
    ## Set data types for the columns
    dtype={'SNP':str, 'CHR':int, 'BP': float, 'GENPOS':float, 'A1':str, 
           'A2':str, 'A1FREQ':float, 'F_MISS':float, 'BETA':float, 'SE':float, 
           'P':float}    
    ## read in the file
    summ = pd.read_table(fn, sep='\t', delim_whitespace=True, low_memory=True, 
                         dtype=dtype)    
    ## Get rid of duplicates
    subset = summ.drop_duplicates(subset='SNP')  
    ## free some memory
    del summ
    gc.collect()
    ## make sure duplicates are trully dealt with
    subset = subset[~subset.duplicated(subset='SNP')]
    ## Get rid of lines that might have the title repeated (this probably 
    ## would) fail anyhow from dtype above, but just in case
    subset = subset[~(subset.SNP == 'SNP')]
    ## Get rid of missing Pvalues and missing alleles (no trustworthy data)
    subset = subset.dropna(subset=['A1','A2','P'])
    ## Drop unreal pvalues
    idx = subset.P <= 1
    subset = subset[idx]
    ## Deal with ambiguities in alleles (also not trustworthy data)
    subset = subset[[True if len(x) == 1 else False for x in subset.A1]]
    subset = subset[[True if len(y) == 1 else False for y in subset.A2]]
    ## strand ambiguity
    subset[~((subset.A1 == 'A') & (subset.A2 == 'T')) | 
           ~((subset.A1 == 'T') & (subset.A2 == 'A')) | 
           ~((subset.A1 == 'G') & (subset.A2 == 'C')) | 
           ~((subset.A1 == 'C') & (subset.A2 == 'G'))]
    ## Write to file
    nodups = '%s_nodups.txt'%(pref)
    subset.to_csv(nodups ,sep= '\t', index=False, na_rep='NA')    
    ## Get a file with the remaining SNPs to filter in
    ext = '%s_snps.extract'
    subset.SNP.to_csv(ext, index=False) 
    ## free up memory del subset (this is probably dealt with by the GC but 
    ## just in case)
    del subset
    gc.collect()
    return nodups, ext, subset    


## Get CLA and format the parameters acordingly
def format_input(args):
    '''
    Based on input summary statistics and and LD matrix, concatenate into
    an array of m (snps) x p (paramerters=concat(beta,vec(Dm)))
    '''
    
## built the network

## get output

## cross-validate


def main(args):
    '''
    Execute the script
    '''
if __name__ == '__main__':
    ## define CLA
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prefix', help='prefix for outputs', 
                        required=True)
    parser.add_argument('-s', '--SummaryStats', help='Filename of the summary \
    statistics', required=True)    
    parser.add_argument('-g', '--GenotypeFile', help='Path and prefix of the \
    genotype files (bed, bim, fam) (A.K.A. target)', required=True)
    args = parser.parse_args()
    main(args)