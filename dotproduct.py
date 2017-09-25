'''
Dot product
'''

import pickle as P
import numpy as np
import pandas as pd
import sys, argparse, os
from collections import defaultdict
try:
    import matplotlib
    matplotlib.style.use('ggplot')
    import matplotlib.pyplot as plt
except:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
    
def read_LD(fn,stack=False):
    '''
    read the LD file as outputted by plink
    '''
    df = pd.read_table(fn, delim_whitespace=True)
    ## Drop Nans
    df = df.dropna()
    ## Drop MAFs
    df = df[(df.MAF_A > 0.01) & (df.MAF_A < 0.99) & 
            (df.MAF_B > 0.01) & (df.MAF_B < 0.99)]
    ## compute the distance
    df.loc[:,'Distance (Bp)'] = abs(df.BP_A - df.BP_B)
    if stack:
        m = pd.DataFrame()
        m.loc[:,'SNPS'] = df.SNP_A.append(df.SNP_B)        
        m.loc[:,'MAF'] = df.MAF_A.append(df.MAF_B)
        m.loc[:,'SNPS'] = df.SNP_A.append(df.SNP_B)
        m.loc[:,'Position (BP)'] = df.BP_A.append(df.BP_B)
        df = m
    return df


def from_dataframes(df1, df2, label1, label2, prefix, typ='R'):
    '''
    try to get dot products
    '''
    dp = defaultdict(float)
    dpA = defaultdict(float)
    dpB = defaultdict(float)
    snpd = defaultdict(int)
    snpdA = defaultdict(int)
    snpdB = defaultdict(int)
    dot = df1.merge(df2,on=['SNP_A', 'SNP_B']).loc[:,['SNP_A', 'SNP_B', 'D_x', 
                                                      'D_y','%s_x'%(typ), 
                                                      '%s_y'%(typ)]]    

    dot.loc[:,'DtD'] = dot.D_x * dot.D_y
    cols = {'D_x':'%s D'%(label1), 'D_y':'%s D'%(label2), 
            '%s_x'%(typ):'%s %s'%(label1, typ), '%s_y'%(typ):'%s %s'%(label2, 
                                                                      typ)}
    dot = dot.rename(columns=cols)
    for line in dot.itertuples():
        ## Cross ancestry Product
        dp[line[1]] += line[-1]
        dp[line[2]] += line[-1] 
        snpd[line[1]] += 1
        snpd[line[2]] += 1
        ## Within ancestry products
        dpA[line[1]] += line[3]**2
        dpA[line[2]] += line[3]**2
        snpdA[line[1]] += 1
        snpdA[line[2]] += 1        
        dpB[line[1]] += line[4]**2
        dpB[line[2]] += line[4]**2
        snpdB[line[1]] += 1
        snpdB[line[2]] += 1        
    dps=(dp,dpA,dpB)
    dens = (snpd,snpdA,snpdB)
    with open('%s.dotproduct'%(prefix),'wb') as F:
        P.dump((dens, dps,dot), F)    
    return dens, dps, dot 

def plot_decay(df, label, typ='D'):
    '''
    plot decay pattern of LD
    '''
    plt.figure()
    ss = df.loc[:, [typ, 'Distance (Bp)']]
    ax = df.plot.scatter(x='Distance (Bp)', y=typ, s=0.01, alpha=0.5,
                         color='DarkBlue')
    plt.title('%s Decay in %s'%(typ, label))
    plt.savefig('%s_%s.decay.png'%(label,typ))
    plt.close()

def plot_LDcorr(merged, labels, typ='D'):
    '''
    PLot the Ds of the two LD matrices
    '''
    X = '%s %s'%(labels[0], typ)
    Y = '%s %s'%(labels[1], typ)
    #df = merged.rename(columns={'%s_x'%(typ): X, '%s_y'%(typ):Y})
    plt.figure()
    merged.plot.scatter(x=X, y=Y, s=0.01, alpha=0.5, color='DarkBlue')
    plt.savefig('%s_vs_%s_%s.png'%(labels[0],labels[1],typ))
    plt.close()

def plot_R2vsD(mat, label):
    '''
    Plot R2 vs D per group
    '''
    try:
        df = mat.loc[:,['D','R2']]
    except:
        mat.loc[:,'R2'] = mat.R**2
        df = mat.loc[:,['D','R2']]
    plt.figure()
    df.plot.scatter(x='D', y='R2', s=0.01, alpha=0.5, color='DarkBlue')
    plt.title(label)
    plt.savefig('R2vsD_%s.png'%(label))
    plt.close()
    
def plot_DtDhist(merged, dot, labels, prefix):
    '''
    Plot histogram of the dot product
    '''
    #df = merged.rename(columns={'D_x':labels[0], 'D_y':labels[1]}).loc[
    #    :,[labels[0], labels[1]]]
    titles =['%s D'%(labels[0]), '%s D'%(labels[1]), 'Dot Product']
    ser = pd.Series(list(dot.values()))
    f, axarr = plt.subplots(3, sharex=True)
    merged.loc[:,'%s D'%(labels[0])].plot.hist(bins=100, alpha=0.5, ax=axarr[0])
    merged.loc[:,'%s D'%(labels[1])].plot.hist(bins=100, alpha=0.5, ax=axarr[1])
    ser.plot.hist(bins=100, alpha=0.5, ax=axarr[2])
    for i, ax in enumerate(axarr): 
        ax.set_yscale('log')    
        ax.set_title(titles[i])
    #plt.title('%s vs %s LD Dot product'%(labels[0], labels[1]))
    plt.savefig('%s_DtD_hist.png'%(prefix))
    plt.close()

def plot_DvsRRate(df, label, genemap):
    '''
    plot recombination rate and D by position
    '''
    #gm = '/home/jshleap/uk_biobank/sergio/LDbyEthnicity/Typed_only/'
    #gm += 'genetic_map_hg19.txt'
    genmap = pd.read_table(genemap, delim_whitespace=True, names=[
        'Chr','Position (BP)', 'Recombination Rate', 'gposition'], header=0)
    m = pd.DataFrame()
    m.loc[:,'Position (BP)'] = df.BP_A.append(df.BP_B)
    m.loc[:,'LD (D)'] = df.D
    m = m.merge(genmap, on='Position (BP)')
    m = m.loc[:, ['Position (BP)', 'LD (D)', 'Recombination Rate']]
    m = pd.DataFrame(m.loc[:,m.columns[1:]].values, 
                     index=m.loc[:,'Position (BP)'], columns=m.columns[1:])
    plt.figure()
    #m.loc[:,'LD (D)'].plot()
    #m.loc[:,'Recombination Rate'].plot(secondary_y=True, style='g')
    m.plot.scatter(x='LD (D)', y='Recombination Rate', s=0.01, alpha=0.5);
    plt.title(label)
    plt.savefig('DvsRRate_%s.png'%(label))
    plt.close()
    
def main(args):
    '''
    get the dot product of two matrices
    '''
    ## process labels
    if not args.labels:
        labels = [x[x.rfind('.')] for x in args.files]
    else:
        labels = args.labels
    ## read files
    fn1, fn2 = args.files
    df1, df2 = read_LD(fn1), read_LD(fn2)
    ## unPickle files or execute from_dataframes
    if os.path.isfile('%s.dotproduct'%(args.prefix)):
        with open('%s.dotproduct'%(args.prefix), 'rb') as L:
            dot, merged = P.load(L)
    else:
        label1, label2 = labels
        dens, dot, merged = from_dataframes(df1, df2, label1, label2, 
                                            args.prefix, typ=args.type)
    dot, dotA, dotB = dot    
    # plots
    ## Plot decay of each individual dataset
    plot_decay(df1, labels[0])
    plot_decay(df1, labels[0], typ=args.type)
    plot_decay(df2, labels[1])
    plot_decay(df2, labels[1], typ=args.type)
    ## R2 vs D per dataset
    plot_R2vsD(df1, labels[0])
    plot_R2vsD(df2, labels[1])  
    ## D vs recombination rate per dataset
    plot_DvsRRate(df1, labels[0], args.geneticMAP)
    plot_DvsRRate(df2, labels[1], args.geneticMAP)
    ## plot the correlation of the two
    plot_LDcorr(merged, labels)
    plot_LDcorr(merged, labels, typ=args.type)
    plot_DtDhist(merged, dot,labels, args.prefix)
    

if __name__ == '__main__':
    ## default gen map
    gm = '/home/jshleap/uk_biobank/sergio/LDbyEthnicity/Typed_only/'
    gm += 'genetic_map_hg19.txt'    
    ## define CLA
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prefix', help='prefix for outputs', 
                        required=True)
    parser.add_argument('-f', '--files', help='Filenames of the plink LD \
    matrices. It requires 2',  action='append', required=True)    
    parser.add_argument('-l', '--labels', help='labels for the two files',  
                        action='append', default=[], required=True)
    parser.add_argument('-g', '--geneticMAP', help='Path to genetic map',  
                        default=gm)    
    parser.add_argument('-t', '--type', help='R or R2', default='R2')    
    args = parser.parse_args()
    main(args) 
