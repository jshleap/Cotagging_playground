'''
Dot product
'''
import pandas as pd
import pickle as P
import numpy as np
from scipy.sparse import lil_matrix
from scipy.stats import norm
import statsmodels.stats.power as smp

def read_LD(fn):
    '''
    read the LD file as outputted by plink
    '''
    df = pd.read_table(fn, delim_whitespace=True)
    ## Drop Nans
    df = df.dropna()
    ## Drop MAFs
    df = df[(df.MAF_A > 0.01) & (df.MAF_A < 0.9) & (df.MAF_B > 0.01) &
            (df.MAF_B < 0.9)]
    return df

def squarize_matrix(df):
    '''
    Make a square D matrix
    '''
    snpa = list(set(df.SNP_A))
    M = lil_matrix((len(snpa),len(snpa)))#, dtype=float)
    for i, s in enumerate(snpa):
        for ind, snp in enumerate(snpa):
            a = df[(df.SNP_A == s) & (df.SNP_B == snp)]
            if a.empty:
                #a=np.nan
                continue
            else:
                M[i,ind] = a.D
    return M

def fromTable(df1, df2):
    ''' Compute the dot product directly from table '''
    dot = df1.merge(df2,on=['SNP_A', 'SNP_B']).loc[:,['SNP_A', 'SNP_B', 'D_x', 
                                                      'D_y']]
    del df1
    del df2
    dot.loc[:,'DtD'] = dot.D_x * dot.D_y
    dp = pd.DataFrame()
    snps=set(dot.SNP_A.append(dot.SNP_B))
    for snp1 in snps:
        for snp2 in snps:
            eur = dot[(dot.SNP_A  == snp1) |(dot.SNP_B == snp1)]
            afr = dot[(dot.SNP_A  == snp2) |(dot.SNP_B == snp2)]
            dp.loc[snp1,snp2] = eur.merge(afr, on=['SNP_A','SNP_B','DtD']
                                          ).DtD.sum()
    return dp
    
    
afr = dot[(dot.SNP_A  == snp1) |(dot.SNP_B == snp1)]
    
ts = set(eur.loc[:,['SNP_A','SNP_B']].append(afr.loc[:,['SNP_A','SNP_B']]).stack(
    ).ravel())

eur.pivot(index='SNP_A', columns='SNP_B', values='D_y')
afr.pivot(index='SNP_A', columns='SNP_B', values='D_y')

eur.pivot(index='SNP_A', columns='SNP_B', values='D_y').merge(afr.pivot(index='SNP_A', columns='SNP_B', values='D_y'))

zip(afr.SNP_A, afr.SNP_B)
a = pd.Series(eur.D_x, index=)


a = pd.DataFrame(index=[snp1,snp1], columns=ts)


def F_transf(r, n, confval=0.95, power=0.8):
    '''
    Fisher transformation and the significance of a correlation r
    :param iterable r: vectror with all the correlation tested
    :param int n: sample size
    :param float confval: confidence value
    :param float power: power threshold (as beta for type 2 error)
    '''
    fisher =  np.arctanh(r)
    confval = 1-((1-self.confval)/2)
    z_val =  norm.isf(1-confval) / sqrt(self.n-3)    
    ns = map(lambda x: smp.NormalIndPower().solve_power(
        np.arctanh(x), alpha=0.05, ratio=0, power=0.8, alternative='two-sided') 
             + 3, fisher)
    
    