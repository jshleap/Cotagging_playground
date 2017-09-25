'''
Dot product V.2
'''
from itertools import combinations, cycle, permutations, product
import pickle as P
import numpy as np
import pandas as pd
import sys, argparse, os
from collections import defaultdict
from gwaspipe import ManhattanPlot
from joblib import delayed, Parallel
try:
    import matplotlib
    matplotlib.style.use('ggplot')
    import matplotlib.pyplot as plt
except:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
from matplotlib.ticker import NullFormatter

#----------------------------------------------------------------------
def read_LD(fn, verbose=True, stack=False):
    '''
    read the LD file as outputted by plink
    '''
    if verbose:
        print('Reading file %s'%(fn))
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
    if verbose: print('\tDONE\n')
    return df


#----------------------------------------------------------------------
def from_dataframes(df1, df2, label1, label2, prefix, typ='R', verbose=True):
    '''
    try to get dot products
    '''
    if verbose: print('Processing both datasets:')
    dp = defaultdict(float)
    dpA = defaultdict(float)
    dpB = defaultdict(float)
    snpd = defaultdict(int)
    snpdA = defaultdict(int)
    snpdB = defaultdict(int)
    groups = defaultdict(list)
    #for l in df1.itertuples():
        ### SNP density per tag snp for pop1
        #snpdA[l[3]] += 1
        #snpdA[l[7]] += 1     
        ### Within ancestry products for pop 1
        #dpA[l[3]] += l[-2]**2
        #dpA[l[7]] += l[-2]**2          
    #for li in df2.itertuples():
        ### SNP density per tag snp for pop1
        #snpdB[li[3]] += 1
        #snpdB[li[7]] += 1
        ### Within ancestry products for pop 2
        #dpB[li[3]] += li[-2]**2
        #dpB[li[7]] += li[-2]**2        
    ## Merge the individual dataframes
    snps1 = pd.unique(pd.concat((df1.SNP_A, df1.SNP_B))).shape[0]
    snps2 = pd.unique(pd.concat((df2.SNP_A, df2.SNP_B))).shape[0]
    if verbose: 
        print('\tMerging dataframes ...')
        print('\t  Dataframe 1 with %d entries and %d snps' % (df1.shape[0], 
              snps1))
        print('\t  Dataframe 2 with %d entries and %d snps' % (df2.shape[0], 
              snps2))
    dot = df1.merge(df2,on=['SNP_A', 'SNP_B']).loc[:,['SNP_A', 'SNP_B', 'D_x', 
                                                      'D_y','%s_x'%(typ), 
                                                      '%s_y'%(typ)]]  
    if verbose:
        snps12 = pd.unique(pd.concat((dot.SNP_A, dot.SNP_B))).shape[0]
        print('\t  Resulting dataframe with %d entries and %d snps' % (
            dot.shape[0], snps12))
    ## include the product of the individual populations
    dot.loc[:,'DtD'] = dot.D_x * dot.D_y
    ## include the tagging within each population
    dot.loc[:,'D1'] = dot.D_x**2
    dot.loc[:,'D2'] = dot.D_y**2
    ## get the dot product and snp density for the merged set
    if verbose: print('\tComputing dot products (LDscores) ...')
    count = [0, []]
    for row in dot.itertuples():
        count[0] += 1
        count[1].append(row.SNP_A)
        count[1].append(row.SNP_B)
        ## Cross ancestry Product
        dp[row.SNP_A] += row.DtD
        dp[row.SNP_B] += row.DtD 
        snpd[row.SNP_A] += 1
        snpd[row.SNP_B] += 1
        groups[row.SNP_A].append(row.SNP_B)
        groups[row.SNP_B].append(row.SNP_A)
        ### SNP density per tag snp for pop1
        #snpdB[li[3]] += 1
        #snpdB[li[7]] += 1
        ## Within ancestry products for pop 2
        dpA[row.SNP_A] += row.D1
        dpA[row.SNP_B] += row.D1          
        ## Within ancestry products for pop 2
        dpB[row.SNP_A] += row.D2
        dpB[row.SNP_B] += row.D2  
    if verbose:
        print ('%d rows and %d snps processed' % (count[0], len(set(count[1]))))
        print ('dp has %d entries' % len(dp.keys() ))
    ## rename the columns to track ancestries
    cols = {'D_x':'%s D' % label1, 'D_y':'%s D' % label2, '%s_x' % typ:'%s %s'%(
        label1, typ), '%s_y' % typ :'%s %s' % (label2, typ), 
            'D1': '$D_{%s}^{2}$'% label1, 'D2': '$D_{%s}^{2}$'% label2}
    dot = dot.rename(columns=cols)    
    dps=(dp, dpA, dpB)
    tempA = pd.DataFrame({'SNP':list(dp.keys()), 'Cotagging':list(dp.values())})
    tempB = pd.DataFrame({'SNP':list(dpA.keys()), 'Tagging %s' % label1:list(
        dpA.values())})
    tempC = pd.DataFrame({'SNP':list(dpB.keys()), 'Tagging %s' % label2:list(
        dpB.values())})
    dtf = tempA.merge(tempB, on='SNP').merge(tempC, on='SNP')
    if verbose: print('\tDumping data in %s ...'%(prefix)) 
    dtf.to_csv('%s_taggingscores.tsv' % prefix, sep='\t', index=False)
    dens = (snpd, snpdA, snpdB)
    #with open('%s.dotproduct'%(prefix),'wb') as F:
     #   P.dump((dens, dps, dot), F)
    if verbose: print('\tDONE')
    return dens, dps, dot 

#----------------------------------------------------------------------
def plot_DtDhist(merged, dot, labels, prefix, bns):
    """
    Plot histogram of the dot product
    """
    titles =['%s D'%(labels[0]), '%s D'%(labels[1]), 'Dot Product']
    ser = pd.Series(list(dot.values()))
    f, axarr = plt.subplots(3, sharex=True)
    merged.loc[:,'%s D'%(labels[0])].plot.hist(bins=bns, alpha=0.5, ax=axarr[0])
    merged.loc[:,'%s D'%(labels[1])].plot.hist(bins=bns, alpha=0.5, ax=axarr[1])
    ser.plot.hist(bins=bns, alpha=0.5, ax=axarr[2])
    for i, ax in enumerate(axarr): 
        ax.set_yscale('log')    
        ax.set_title(titles[i])
    plt.savefig('%s_DtD_hist.png'%(prefix))
    plt.close()

#----------------------------------------------------------------------
def plot_decay(df, label, typ='D'):
    """
    Plot decay pattern of LD
    """
    plt.figure()
    ss = df.loc[:, [typ, 'Distance (Bp)']]
    df.plot(kind='scatter', x='Distance (Bp)', y=typ, s=0.3, alpha=0.8, lw = 0,
            color="#3F5D7D")
    plt.title('%s Decay in %s'%(typ, label))
    plt.savefig('%s_%s.decay.png'%(label,typ))
    plt.close()

#----------------------------------------------------------------------
def changeSize(x, y, bins=100):#,radius=0.05):
    '''
    report alpha and size depending on density round a radius
    '''
    #xhist = np.histogram(x, bins=1000)
    #yhist = np.histogram(y, bins=1000)
    #x, y = x.apply(getFreq, hist=xhist), y.apply(getFreq, hist=yhist)
    #dens = pd.concat((x.dropna(),y.dropna()),axis=1).min(axis=1)
    dens = getFreq2(x, y, bins)#pd.concat((x,y),axis=1).dropna().min(axis=1)
    alphas = norm(1/ dens, 0.3, 1)
    sizes  = norm(1/ dens, 0.01, 1)
    return alphas, sizes, dens

#----------------------------------------------------------------------
def getFreq(x, hist):
    '''
    Get the frequency of t a value x given a histogram hist
    '''
    freq, val = hist
    for i in range(len(val)-1):
        if (x > val[i]) and (x < val[i+1]) and not np.isnan(freq[i]): 
            return freq[i]

#----------------------------------------------------------------------
def single(df, lowerx, lowery, chun):
    indices = df[(df.x >= lowerx) & (df.x <= lowerx + chun) &
                 (df.y >= lowery) & (df.y <= lowery + chun)].index
    count = indices.shape[0]        
    return dict(zip(indices, cycle([count])))

#----------------------------------------------------------------------
def getFreq2(x, y, bins):
    '''
    Get the frequency of t a value x given a histogram hist
    '''
    df = pd.DataFrame({'x':x,'y':y})
    #grid = {}
    maxr = max(x.max(), y.max())
    minr = min(x.min(), y.min())
    chun = (maxr - minr) / bins
    A = np.arange(minr, maxr + chun, chun)
    combos = product(A, A)
    #dicts = 
    grid = { k: v for d in Parallel(n_jobs=-1)(delayed(single)(
        df, lowerx, lowery, chun) for lowerx, lowery in combos) 
             for k, v in d.items()}
    #    indices = df[(df.x > lowerx) & (df.x < lowerx + chun) &
     #                (df.y > lowery) & (df.y < lowery + chun)].index
        #subx = x[(x > lowerx) & (x < lowerx + chun)]
        #suby = y[(y > lowery) & (y < lowery + chun)]
        #indices = subx.index.intersection(suby.index)
        #count = indices.shape[0]
        #grid.update(dict(zip(indices, cycle([count]))))
    return pd.Series(grid)
    
        
#----------------------------------------------------------------------
def norm(array, a=0, b=1):
    '''
    normalize an array between a and b
    '''
    ## normilize 0-1
    rang = max(array) - min(array)
    A = (array - min(array)) / rang
    ## scale
    range2 = b - a
    return (A * range2) + a    


#----------------------------------------------------------------------
def ScatterWithDensities(df, col1, col2, prefix, suffix, diagonal=True, 
                         galaxy=False, corr=True, alphatunning=False, dpi=300, 
                         dens=None, bins=100):
    """
    plot the scatter (correlation) of two variables (col1, col2) and safe the
    plot in <prefix>_<suffix>.png
    """
    ## make galaxy style
    if galaxy:
        plt.style.use('dark_background')
    ## get only the required columns and drop nans
    df = df.loc[:,[col1,col2]].dropna()
    ## no labels
    #nullfmt = NullFormatter() 
    ## definitions for the axes
    left, width = 0.1, 0.7
    bottom, height = 0.08, 0.7
    bottom_h = left_h = left + width + 0.05
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.1]
    rect_histy = [left_h, bottom, 0.1, height]
    ## start with a rectangular Figure
    plt.figure(1, figsize=(11, 10.5), dpi=dpi)
    #plt.tight_layout()
    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)
    ## get alpha and size
    if not alphatunning:
        alpha, size, colors = 0.5, 0.5, 'black'
    else:
        if dens is None:
            alpha, size, density = changeSize(df.loc[:, col1], df.loc[:, col2],
                                              bins = bins)
        else:
            alpha, size, density = dens
        if not galaxy:
            ## include alpha in the color (individual alphas cannot be passed)
            colors = np.zeros((len(alpha),4))
            colors[:, 2] = 1.0 #blue
            colors[:, 3] = 1.0 if alpha is None else alpha
        else:
            colors = density
                    
    ## the scatter plot
    if galaxy:
        alpha = 1
        #colors= 
        axScatter.scatter(df.loc[:,col1], df.loc[:,col2], s=size, c=norm(
            colors,0.3, 1), cmap='plasma')#'inferno')#'autumn')
    else:
        axScatter.scatter(df.loc[:,col1], df.loc[:,col2], s=size, color=colors)
    
    ## print the correlation value
    if corr:
        color = 'white' if galaxy else 'black'
        corr = df.loc[:, [col1, col2]].corr().loc[col1,col2]
        axScatter.text(0.05, 0.95, 'Corr = %.2f'%(corr), verticalalignment='top', 
                       horizontalalignment='left', transform=axScatter.transAxes, 
                       color=color, fontsize=13)     
    ## now determine nice limits by hand:
    binwidth = 0.1
    #xymax 
    limax= np.max([np.max(df.loc[:,col1]), np.max(df.loc[:,col2])])
    #xymin = 
    limin = np.min([np.min(df.loc[:,col1]), np.min(df.loc[:,col2])])    
    #limax = (int(xymax/binwidth) + 1) * binwidth
    #limin = (int(xymin/binwidth)) * binwidth
    if corr:
        axScatter.set_xlim((limin, limax))
        axScatter.set_ylim((limin, limax))
    #bins = 100#np.arange(limin, limax + binwidth, binwidth)
    if diagonal:
        marker = 'w--' if galaxy else 'k--'
        a = np.arange(limin,limax+0.1,0.1)
        axScatter.plot(a, a, marker, linewidth=1)
    ## no labels
    #axHistx.xaxis.set_major_formatter(nullfmt)
    #axHisty.yaxis.set_major_formatter(nullfmt)
    ## Plot the histograms
    df[col1].plot.hist(ax=axHistx, bins=bins, alpha=0.5, color='white')
    df[col2].plot.hist(orientation='horizontal', ax=axHisty, bins=bins, 
                       alpha=0.5, color='white')
    ##set labels
    axScatter.set_xlabel(col1)
    axScatter.set_ylabel(col2)
    ## set limits
    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())
    ## Save figure
    plt.savefig('%s_%s.png'%(prefix, suffix))
    plt.close()    
    plt.style.use('ggplot')
    return alpha, size, density

#----------------------------------------------------------------------
def RvsR(df, prefix, bins):
    """
    Plot the correlation of R with densities
    Taken and modified from pylab examples
    """
    # Regular histogram plots
    suffix = 'Rwithhist'
    col1, col2 = df.columns[4:6]
    rdens = ScatterWithDensities(df, col1, col2, prefix, suffix, bins=bins)
    suffix = 'Dwithhist'
    col1, col2 = df.columns[2:4]
    ddens = ScatterWithDensities(df, col1, col2, prefix, suffix, bins=bins)    
    ## Galaxy plots
    #suffix = 'Rgalaxy'
    #col1, col2 = df.columns[4:6]
    #ScatterWithDensities(df, col1, col2, prefix, suffix, galaxy=True, 
                         #dens=rdens, bins=bins)
    #suffix = 'Dgalaxy'
    #col1, col2 = df.columns[2:4]
    #ScatterWithDensities(df, col1, col2, prefix, suffix, galaxy=True, 
                         #dens=ddens, bins=bins)    

#----------------------------------------------------------------------
def setcolsrows(bins):
    """
    return ncols and nrows based on the number of bins (plots)
    """
    sqrt = np.sqrt(bins)
    ncols = np.ceil(sqrt)
    nrows = np.floor(sqrt)
    return int(ncols), int(nrows)

#----------------------------------------------------------------------
def getsubsets(merged, rang, MAF_mean, curr, nor):
    """
    return a list of non-empty subsets
    """
    l = []
    for p, i in enumerate(rang):
        ss = (MAF_mean >= curr) & (MAF_mean <= curr + i)
        subset = merged[ss]
        subset.loc[:, 'mean2'] = list(nor[ss])
        if not subset.empty:
            l.append(subset)
    return l
    
        
#----------------------------------------------------------------------
def binD(df1, df2, labels, prefix):
    """
    Plot D vs D and RvsR bining on the first population and coloring in the 
    other
    """
    bins=9
    suffixes = [' %s'%(x) for x in labels]
    dot = df1.merge(df2,on=['SNP_A', 'SNP_B'], suffixes=suffixes)
    vmax = np.max((np.max(dot.loc[:, 'MAF_A%s'%(suffixes[0])]),
                  np.max(dot.loc[:, 'MAF_B%s'%(suffixes[0])])))                  
    vmin = np.min((np.min(dot.loc[:, 'MAF_A%s'%(suffixes[0])]),
                  np.min(dot.loc[:, 'MAF_B%s'%(suffixes[0])]))) 
    step = (vmax - vmin)/bins
    MAF_mean = np.mean([dot.loc[:,'MAF_A%s'%(suffixes[0])], 
                        dot.loc[:,'MAF_B%s'%(suffixes[0])]], axis=0)
    MAF_meanp2 = np.mean([dot.loc[:,'MAF_A%s'%(suffixes[1])], 
                          dot.loc[:,'MAF_B%s'%(suffixes[1])]], axis=0)
    nor = norm(MAF_meanp2, 0, 1)  
    rang = np.arange(vmin, vmax + step, step)
    subs = getsubsets(dot, rang, MAF_mean, vmin, nor)
    #ncols, nrows = setcolsrows(len(subs))
    nrows = int(np.ceil(len(subs)/3))
    fig, axes = plt.subplots(ncols=3, nrows=nrows, sharey=True, sharex=True)
    axs = axes.ravel()
    cNorm  = plt.matplotlib.colors.Normalize(vmin=0, vmax=1)
    x, y ='D%s'%(suffixes[0]), 'D%s'%(suffixes[1])
    for p, subset in enumerate(subs):
        subset.plot(kind='scatter', x=x, y=y, c='mean2', colormap='inferno', 
                    ax=axs[p])
        #subset.plot(kind='scatter', x='D%s'%(suffixes[0]), y='D%s'%(suffixes[1]
        #),
        #            c='mean2', colormap='inferno', ax=axs[p])
    plt.xlabel('D%s'%(suffixes[0]))
    plt.savefig('%s_binnedD.png'%prefix)
    plt.close()

    
#----------------------------------------------------------------------
def hexaplopts(df, prefix):
    """
    Make hexagon plots (hexbin) of the LDs
    """
    cols=df.columns
    def diagonal(df,col1, col2):
        limax= np.max([np.max(df.loc[:,col1]), np.max(df.loc[:,col2])])
        limin = np.min([np.min(df.loc[:,col1]), np.min(df.loc[:,col2])])    
        return np.arange(limin,limax+0.1,0.1)
 
    ## Start hexagonal plot for R
    plt.figure()
    ax = df.plot.hexbin(x=cols[4], y=cols[5], bins='log', cmap='inferno')
    #, gridsize=150)
    corr = df.loc[:,[cols[4], cols[5]]].corr().loc[cols[4], cols[5]]
    ax.text(0.05, 0.95, 'Corr = %.2f'%(corr), verticalalignment='top', 
            horizontalalignment='left', transform=ax.transAxes, color='white', 
            fontsize=13)  
    diag = diagonal(df,cols[4], cols[5])
    ax.plot(diag, diag, 'w--', linewidth=0.5)
    plt.savefig('%s_Rhexa.pdf'%(prefix))
    plt.close()

    ## Start hexagonal plot for R2
    x, y = '%s R2'%(cols[4].split()[0]), '%s R2'%(cols[5].split()[0])
    df2 = (df.loc[:,cols[4:6]]**2).rename(columns={cols[4]:x, cols[5]:y})
    plt.figure()
    ax = df2.plot.hexbin(x=x, y=y, bins='log', cmap='inferno')#, gridsize=150)
    corr = df2.loc[:, [x, y]].corr().loc[x,y]
    ax.text(0.05, 0.95, 'Corr = %.2f'%(corr), verticalalignment='top', 
            horizontalalignment='left', transform=ax.transAxes, color='white', 
            fontsize=13)      
    diag = diagonal(df,cols[4], cols[5])
    ax.plot(diag, diag, 'w--', linewidth=0.5)
    plt.savefig('%s_R2hexa.png'%(prefix))
    plt.close()

    ## Start hexagonal plot for Ds
    plt.figure()
    df.plot.hexbin(x=cols[2], y=cols[3], bins='log', cmap='inferno')#, gridsize=100)
    corr = df.loc[:, [cols[2], cols[3]]].corr().loc[cols[2], cols[3]]
    ax.text(0.05, 0.95, 'Corr = %.2f'%(corr), verticalalignment='top', 
            horizontalalignment='left', transform=ax.transAxes, color='white', 
            fontsize=13)     
    diag = diagonal(df,cols[2], cols[3])
    ax.plot(diag, diag, 'w--', linewidth=0.5)    
    plt.savefig('%s_Dhexa.png'%(prefix))
    plt.close()    

#----------------------------------------------------------------------
def ScoresVsScores(dots, labels, bins=100):
    """
    Plot the DtD for the cross-ancestry and the within ancestry
    """
    suffix='scores'
    cols = {0:'Cross-ancestry', 1:'Within-ancestry %s'%(labels[0]), 
            2:'Within-ancestry %s'%(labels[1])}
    df = pd.concat([pd.Series(x) for x in dots], axis=1)
    df = df.rename(columns=cols)
    for col1, col2 in combinations(df.columns,2):
        if 'Cross-ancestry' in col1:
            prefix = 'CrossVs%s'%(col2.split()[1])
        elif 'Cross-ancestry' in col2:
            prefix = 'CrossVs%s'%(col1.split()[1])
        else:
            prefix = '%sVs%s'%(col1.split()[1], col2.split()[1])
        ScatterWithDensities(df, col1, col2, prefix, suffix, bins=bins)
    return df

#----------------------------------------------------------------------
def ScoresVsDensity(dfdots, dens, labels, bins):
    """
    Plot dotproduct scores (cross-ancestry and within ancestry) in a dataframe
    vs the dictionary of corresponding snp densities
    """
    prefix='%svs%s'%(labels[0],labels[1])
    suffix = 'snpDensity'
    cols = {0:'Cross-ancestry SNP density', 1:'%s SNP density'%(labels[0]), 
            2:'%s SNP density'%(labels[1])}    
    df = pd.concat([pd.Series(x) for x in dens], axis=1)
    df = df.rename(columns=cols)
    dotcol = dfdots.columns
    dfcols = df.columns
    df = pd.concat((df,dfdots), axis=1)
    bar_width = 0.25    
    opacity = 0.4
    error_config = {'ecolor': '0.3'}    
    #plt.figure()
    for i in range(len(dotcol)):
        prefix = '%s'%(dotcol[i].split()[-1])
        D = df.loc[:,[dfcols[i], dotcol[i]]].dropna()
        #re = plt.bar(D.loc[:,dfcols[i]], D.loc[:, dotcol[i]], bar_width, 
                     #alpha=opacity, label=dotcol[i])
        ScatterWithDensities(D, dfcols[i], dotcol[i], prefix, suffix, 
                             diagonal=False, corr=False, alphatunning=False,
                             bins=bins)
        #D.plot.scatter(x=dfcols[i], y=dotcol[i])
        #print(D.head())
        #if i == 0:
            #ax = D.plot(x=dfcols[i], y=dotcol[i])
        #else:
            #D.plot(x=dfcols[i], y=dotcol[i], ax=ax)
    #plt.savefig('%s.snpDensity.png'%(prefix))
    #plt.close()            
#----------------------------------------------------------------------
def denormilize(narr, oarr):
    """
    denormalize (0, 1) -> original
    """
    rang = max(oarr) - min(oarr)
    return (narr * rang) + min(oarr)  

#----------------------------------------------------------------------
def align_yaxis(ax1, v1, ax2, v2):
    """
    adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1
    Taken from stackoverflow wrtten by drevicko
    """
    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    adjust_yaxis(ax2,(y1-y2)/2,v2)
    adjust_yaxis(ax1,(y2-y1)/2,v1)

#----------------------------------------------------------------------
def adjust_yaxis(ax,ydif,v):
    """
    shift axis ax by ydiff, maintaining point v at the same location
    Taken from stackoverflow wrtten by drevicko
    """
    inv = ax.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, ydif))
    miny, maxy = ax.get_ylim()
    miny, maxy = miny - v, maxy - v
    if -miny>maxy or (-miny==maxy and dy > 0):
        nminy = miny
        nmaxy = miny*(maxy+dy)/(miny+dy)
    else:
        nmaxy = maxy
        nminy = maxy*(miny+dy)/(maxy+dy)
    ax.set_ylim(nminy+v, nmaxy+v)
    
#----------------------------------------------------------------------
def IncludeDotsinManhattan(dots, summarystats, poplabels, prefix, 
                           chromosome='All'):
    """
    Include the within population score and the co-tagging along a manhattan
    plot
    
    :param tuple dots: a tuple with the 3 dictionaries of co-tagging scores
    :param str summarystats: filename of the summary stats (must include CHR \
    and PVAL as column names)
    :param list poplabels: List of population labels in the order of dot
    :param str prefix: prefix for outputs
    """
    ## read Summary stats
    summ = pd.read_table(summarystats, delim_whitespace=True)
    summ = summ.drop_duplicates()
    summ = summ[~pd.to_numeric(summ.BP, errors='coerce').isnull()]
    summ = summ.rename(columns={'P_BOLT_LMM_INF':'PVAL'})
    summ.loc[:, 'PVAL'] = pd.to_numeric(summ.PVAL)
    summ = summ.sort_values('PVAL')
    summ = summ[~summ.SNP.duplicated()]
    ## create the manhattan plot
    ax, df, x_labels_pos, x_labels =  ManhattanPlot(summ, None, grayscale=True, 
                                                    save=False, 
                                                    chromosome=chromosome)
    ## split the dots tupple into the cotagging, pop1, pop
    dp, dpA, dpE = dots
    ## transform them into dataframes
    #allcot = list(dp.values()) + list(dpA.values()) + list(dpE.values())
    cotag = pd.DataFrame({'SNP':list(dp.keys()), 'Co-Tagging': list(dp.values()
                                                                    )})
    scorA = pd.DataFrame({'SNP':list(dpA.keys()), '%s Score'%(label[0]): list(
        dpA.values())})
    scorB = pd.DataFrame({'SNP':list(dpE.keys()), '%s Score'%(label[1]): list(
        dpE.values())})
    ## merge dataframes
    merged = df.merge(cotag, on='SNP')
    merged = merged.merge(scorA, on='SNP')
    merged = merged.merge(scorB, on='SNP')
    #mincot = merged.loc[:,['Co-Tagging', '%s Score'%(label[0]), '%s Score'%(
    #    label[1])]].min()
    #maxcot = merged.loc[:,['Co-Tagging', '%s Score'%(label[0]), '%s Score'%(
    #        label[1])]].max()    
    ## plot the scores in the manhattan
    ax2 = ax.twinx()
    name = 'Co-Tagging'
    merged.plot(x='ind', y=name, style='b+', label=name, ax=ax2, alpha=0.5, 
                ms=1.5, secondary_y=True, legend=False)
    name = '%s Score'%(label[0])
    merged.plot(x='ind', y=name, style='c+', label=name, ax=ax2, alpha=0.5,
                ms=0.8, secondary_y=True, legend=False)
    name = '%s Score'%(label[1])
    merged.plot(x='ind', y=name, style='m+', label=name, ax=ax2, alpha=0.5,
                ms=0.8, secondary_y=True, legend=False)
    ax.set_xticks(x_labels_pos)
    ax.set_xticklabels(x_labels)
    if chromosome == 'All':
        ax.set_xlim([0, len(df)])
    ax.set_ylim([5, max(df.minuslog10pvalue)])
    ax.set_xlabel('Chromosome')
    ax.set_ylabel('-Log(P-value)')
    ax2.set_ylabel('Co-tagging score')
    #ax2.set_ylim([0,1])#[mincot, maxcot])
    #ax2.set_yticklabels(denormilize(ax2.get_yticks(), allcot))
    #sig =() / df.PVAl.max()
    align_yaxis(ax, 0, ax2, 0)
    ax.axhline(-np.log(5E-8), ls='dashed', lw=0.8, color='black')
    plt.legend(loc='best', markerscale=5, numpoints=1, fontsize=12, 
               fancybox=True)
    plt.savefig('%s.png'%(prefix))
    plt.close()

#----------------------------------------------------------------------
def main(args):
    """
    Execute commands
    """
    v = args.verbose
    ## process labels
    if not args.labels:
        labels = [x[x.rfind('.')] for x in args.files]
    else:
        labels = args.labels
        
    ## read files
    fn1, fn2 = args.files
    df1, df2 = read_LD(fn1, verbose=v), read_LD(fn2, verbose=v)
    label1, label2 = labels     
    ## unPickle files or execute from_dataframes
    if os.path.isfile('%s.dotproduct'%(args.prefix)):
        if v: print('Loading %s.dotproduct picklefile'%(args.prefix))
        with open('%s.dotproduct'%(args.prefix), 'rb') as L: 
            dens, dots, merged = P.load(L)
    else:    
        dens, dots, merged = from_dataframes(df1, df2, label1, label2,
                                             args.prefix, typ=args.type, 
                                             verbose=v)
       
    dot, dotA, dotB = dots
    snpd, snpdA, snpdB = dens
    # plots  
    ## Plot decay
    if (args.plot == 'decay') or (args.plot == 'all'):
        if v: print('Plotting decay')
        plot_decay(df1, labels[0], typ='D')
        plot_decay(df1, labels[0], typ=args.type)
        plot_decay(df2, labels[1], typ='D')
        plot_decay(df2, labels[1], typ=args.type)
        if v: print('\tDONE')
    ## plot binned version of D
    if (args.plot == 'binned') or (args.plot == 'all'):
        if v: print('Plotting binned D')
        binD(df1, df2, labels, args.prefix)
        if v: print('\tDONE')
    ## plot Hexbins for R, R2 and D
    if (args.plot == 'hexbin') or (args.plot == 'all'): 
        if v: print('Plotting hexbins')
        hexaplopts(merged, args.prefix)
        if v: print('\tDONE')
    ## plot DtD
    if (args.plot == 'dtd') or (args.plot == 'all'): 
        if v: print('Plotting dotproduct')
        plot_DtDhist(merged, dot, labels, args.prefix, args.bins)
        if v: print('\tDONE')
    ## plot RvsR
    if (args.plot == 'rvsr') or (args.plot == 'all'): 
        if v: print('Plotting R correlation')
        RvsR(merged, args.prefix, args.bins)
        if v: print('\tDONE')
    ## plot scores
    if (args.plot == 'scores') or (args.plot == 'all'): 
        if v: print('Plotting scores')
        df = ScoresVsScores(dots, labels, bins=args.bins)
        if v: print('\tDONE')
    ## plot scores vs snp density
    if (args.plot == 'density') or (args.plot == 'all'): 
        if v: print('Plotting scores and snp density')
        try:
            ScoresVsDensity(df, dens, labels, args.bins)
        except:
            cols = {0:'Cross-ancestry', 1:'Within-ancestry %s'%(labels[0]), 
                    2:'Within-ancestry %s'%(labels[1])}
            df = pd.concat([pd.Series(x) for x in dots], axis=1)
            df = df.rename(columns=cols)            
            ScoresVsDensity(df, dens, labels, args.bins)
        if v: print('\tDONE')

if __name__ == '__main__':   
    ## define CLA
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prefix', help='prefix for outputs', 
                        required=True)
    parser.add_argument('-f', '--files', help='Filenames of the plink LD \
    matrices. It requires 2',  action='append', required=True)    
    parser.add_argument('-l', '--labels', help='labels for the two files',  
                        action='append', default=[], required=True)  
    parser.add_argument('-t', '--type', help='R or R2', default='R2')
    parser.add_argument('-P', '--plot', help='plot to be done', default='all')
    parser.add_argument('-v', '--verbose', help='Turn off verbosity', 
                        action='store_false', default=True) 
    parser.add_argument('-b', '--bins', help='Bins for density and histogram',
                        type=int, default=100)       
    args = parser.parse_args()
    main(args) 