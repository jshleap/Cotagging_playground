'''
PLot R and modify size and alpha
'''
from itertools import combinations
import pandas as pd
import numpy as np
import pickle as P
import matplotlib, sys
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from scipy import stats
from matplotlib.ticker import NullFormatter


####################################################
## ARGUMENT ONE IS THE PICKLED DOT PRODUCT FILENAME#
## ARGUMENT TWO IS THE OUTPUT PREFIX               #
####################################################
def getFreq(x, hist):
    val = hist[1]
    freq = hist[0]/sum(hist[0])
    for i in range(len(val)):
        if (x > val[i]) and (x < val[i+1]): 
            return freq[i]


def isin(x,y,h,k,r):
    return ((x - h)**2 + (y - k)**2) < r**2
#
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
#    
def changeSize(x,y,radius=0.05):
    '''
    report alpha and size depending on density round a radius
    '''
    xhist = np.histogram(x, bins=100)
    yhist = np.histogram(x, bins=100)
    x, y = x.apply(getFreq, hist=xhist), y.apply(getFreq, hist=yhist)
    dens = pd.concat((x,y),axis=1).mean(axis=1)
    alphas = norm(1/(dens+0.0001), 0.3, 1)
    sizes  = norm(1/(dens+0.0001), 0.01, 1)
    return alphas, sizes

#def modifyAlpha(vx, vy):
    #'''
    #given a vector of values modify their size and alpha given the density
    #'''
    #alphas=[]
    #size=[]
    #for i in range(len(vy)):
        #if (vx[i] >= -0.5) | (vy <= 0.7) 

    #l = len(vector)
    #p = np.percentile(vector, range(0,110,10))
    #z = list(zip(p,p[1:]))
    #l = [len(vector[(vector >= i) & (vector <= j)]) for i, j in z]
    #ma, mi = 1, min(l)/max(l)
    #y = np.arange(mi, ma + (ma-mi)/100, (ma-mi)/100)
    #xA, xB = np.arange(0.2, 0.503, 0.003), np.arange(0.001, 1.00999, 0.00999)
    #slA, intA, _, _, _ = stats.linregress(xA,y)
    #alpha = lambda density: (slA * density) + intA
    #slB, intB, _, _, _ = stats.linregress(xB,y)
    #size = lambda density: (slB * density) + intB
    ##d = dict(d)
    ##z = zip(p,p[1:])
    #alphas = pd.Series(index=vector.index)
    #sizes = pd.Series(index=vector.index)
    #for i, j in z:
        #idx = vector[(vector >= i) & (vector <= j)].index
        #d = len(idx)/max(l)
        #alphas[idx] = alpha(d)
        #sizes[idx] = size(d)
    #return alphas, sizes            



with open(sys.argv[1],'rb') as F:
    dens, dot, df = P.load(F) 

cols = df.columns

nullfmt = NullFormatter()         # no labels

# definitions for the axes
left, width = 0.1, 0.65
bottom, height = 0.1, 0.65
bottom_h = left_h = left + width + 0.02

rect_scatter = [left, bottom, width, height]
rect_histx = [left, bottom_h, width, 0.2]
rect_histy = [left_h, bottom, 0.2, height]

# start with a rectangular Figure
plt.figure(1, figsize=(8, 8))

axScatter = plt.axes(rect_scatter)
axHistx = plt.axes(rect_histx)
axHisty = plt.axes(rect_histy)


# the scatter plot:
#axScatter.scatter(x, y)
#alphaX, sizeX = modifyAlpha(df.R_x)
#alphaY, sizeY = modifyAlpha(df.R_y)
#alpha = (alphaX + alphaY)/2
#size = (sizeX + sizeY)/2
df.plot.scatter(x=cols[4],y=cols[5], ax=axScatter, s=0.01, alpha=0.3)
axScatter.plot(np.arange(-1,1.1,0.1), np.arange(-1,1.1,0.1), 'k--', linewidth=1)

# now determine nice limits by hand:
binwidth = 0.25
xymax = np.max([np.max(np.fabs(df.loc[:,cols[4]])), 
                np.max(np.fabs(df.loc[:,cols[5]]))])
lim = (int(xymax/binwidth) + 1) * binwidth

axScatter.set_xlim((-lim, lim))
axScatter.set_ylim((-lim, lim))

bins = np.arange(-lim, lim + binwidth, binwidth)
df[cols[4]].plot.hist(ax=axHistx, bins=bins, alpha=0.5)
#axHistx.hist(x, bins=bins)
#axHisty.hist(y, bins=bins, orientation='horizontal')
df[cols[5]].plot.hist(orientation='horizontal', ax=axHisty, bins=bins, alpha=0.5
                      )

# no labels
axHistx.xaxis.set_major_formatter(nullfmt)
axHisty.yaxis.set_major_formatter(nullfmt)

# set limits
axHistx.set_xlim(axScatter.get_xlim())
axHisty.set_ylim(axScatter.get_ylim())

plt.savefig('%s_Rwithhist.png'%(sys.argv[2]))
plt.close()


## Start hexagonal plot for R
plt.figure()
df.plot.hexbin(x=cols[4], y=cols[5])#, gridsize=150)
plt.savefig('%s_Rhexa.png'%(sys.argv[2]))
plt.close()

## Start hexagonal plot for R2
x, y = '%s R2'%(cols[4].split()[0]), '%s R2'%(cols[5].split()[0])
df2 = (df.loc[:,cols[4:6]]**2).rename(columns={cols[4]:x, cols[5]:y})
plt.figure()
df2.plot.hexbin(x=x, y=y)#, gridsize=150)
plt.savefig('%s_R2hexa.png'%(sys.argv[2]))
plt.close()

## Start hexagonal plot for Ds
plt.figure()
df.plot.hexbin(x=cols[2], y=cols[3])#, gridsize=100)
plt.savefig('%s_Dhexa.png'%(sys.argv[2]))
plt.close()