'''
window LD heat map
'''
import pandas as pd
import seaborn as sns 
import numpy as np
import argparse

def read_LD(fn,length):
    '''
    read the LD file as outputted by plink
    '''
    df = pd.read_table(fn, delim_whitespace=True)
    ## Drop Nans
    df = df.dropna()
    ## Drop MAFs
    df = df[(df.MAF_A > 0.01) & (df.MAF_B > 0.01)]
    ## compute the distance
    df.loc[:,'Rel_dist'] = df.BP_A - df.BP_A[df.BP_A.index[0]]
    ## get groups
    count = 0
    group=[]
    orilenght=length
    for i in df.Rel_dist:
        if i == length:
            group.append(i)
            count+=1
            length = length + orilenght
        else:
            group.append(i)
    df.loc[:,'len_groups'] = group
    return df


def plot_a_group(name, df, group):
    '''
    create the heatmap of the selected group
    '''
    grouped = df.groupby('len_groups')
    df = grouped.get_group(group)
    table = df.pivot_table(df, values='D', index='SNP_A', columns='SNP_B')
    mask = np.zeros_like(table)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        ax = sns.heatmap(table, mask=mask, square=True)#,vmax=.3)    
    ax.savefig('%s_%s_heatmap.png'%(name,str(group)))

def main(prefix, fn, length, group):
    '''execute'''
    df = read_LD(fn, length)
    plot_a_group(prefix, df, group)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prefix', help='prefix for outputs', 
                        required=True)
    parser.add_argument('-g', '--group', help='the number of the group you want\
    to plot', required=True, default=0, type=int)
    parser.add_argument('-l', '--length', help='size of the window to be \
    plotted in basepairs', required=True, default=50)
    parser.add_argument('-f', '--filename', help='name of the LD file', 
                        required=True, default=50)    

    args = parser.parse_args()
    main(args.prefix, args.filename, args.length, args.group)
