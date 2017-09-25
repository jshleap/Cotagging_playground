'''
interpolation of cM and plot ld decay in genetic distance. Requires the 
dotprodut pickle (see dotproductv2.py) and the mapfile with position and cM
'''

from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pandas as pd
import pickle as P
import sys

## argument 1 the bim file with bath for pop1
## argument 2 dotproduct pickle with path
## argumenr 3 prefix for output
#for now hardcode the map
mapfile = '/Volumes/project/gravel/hleap_projects/UKBB/genetic_map/mapFile.txt'
##read map file
mapdf = pd.read_table(mapfile, delim_whitespace=True, header=None, names=[
    'position', 'cM'])
##read bimfile
bimdf1 = pd.read_table(sys.argv[1], delim_whitespace=True, header=None, names=[
    'chr', 'snp', 'cM','position','A1','A2'])
## create interpolating function based on map file
func = interp1d(mapdf.position, mapdf.cM)

## filter the range of the bim
pos = bimdf1.position[(bimdf1.position >  min(mapdf.position)) & (
    bimdf1.position <  max(mapdf.position))]
newCMs = func(pos)
## include interpolated values into the bim
bimdf1.loc[(bimdf1.position >  min(mapdf.position)) & (bimdf1.position <  max(
    mapdf.position)), 'cM'] = newCMs
## load the dotproduct matrices
with open(sys.argv[2], 'rb') as L: dens, dots, merged = P.load(L)
## loop over merged to get pairs, and add the cM difference to the dataframe
for t in merged.itertuples():
    sub = bimdf1[(bimdf1.snp == t.SNP_A) | (bimdf1.snp == t.SNP_B)]
    cms = sub.cM.diff().sum()
    pos = sub.position.diff().sum()/1000
    merged.loc[t.Index, 'cM'] = cms
    merged.loc[t.Index, 'position'] = cms
## plot centimorgans
Dcols = [x for x in merged.columns[-5:] if (('D' in x) | ('cM' in x))]
Ds = merged.loc[:, Dcols]
df = Ds.set_idex('cM')
plt.figure()
Ds.plot()
plt.xlabel('Distance in cM')
plt.ylabel('Score value')
plt.savefig('%s_DcM.png'%(sys.argv[3]))
plt.close()
Rcols = [x for x in merged.columns[-5:] if (('R' in x) | ('cM' in x))]
Rs = merged.loc[:, Dcols]
df = Rs.set_idex('cM')
plt.figure()
Rs.plot()
plt.xlabel('Distance in cM')
plt.ylabel('Score value')
plt.savefig('%s_RcM.png'%(sys.argv[3]))
plt.close()
## plot bp distance
Dcols = [x for x in merged.columns[-5:] if (('D' in x) | ('position' in x))]
Ds = merged.loc[:, Dcols]
df = Ds.set_idex('position')
plt.figure()
Ds.plot()
plt.xlabel('Distance in kbp')
plt.ylabel('Score value')
plt.savefig('%s_Dkbp.png'%(sys.argv[3]))
plt.close()
Rcols = [x for x in merged.columns[-5:] if (('R' in x) | ('position' in x))]
Rs = merged.loc[:, Dcols]
df = Rs.set_idex('position')
plt.figure()
Rs.plot()
plt.xlabel('Distance in kbp')
plt.ylabel('Score value')
plt.savefig('%s_Rkbp.png'%(sys.argv[3]))
plt.close()
