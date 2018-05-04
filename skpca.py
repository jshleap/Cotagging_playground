from sklearn.decomposition import IncrementalPCA as ipca #PCA
from utilities4cotagging import read_geno
import pandas as pd
import numpy as np
import argparse


def do_pca(g, n_comp, batch_size=None, chunk_size=1000):
    """
    Perform a PCA on the genetic array and return n_comp of it

    :param g: Genotype array
    :param n_comp: Number of components sought
    :return: components array
    """
    #pca = PCA(n_components=n_comp)
    pca = ipca(n_components=n_comp, batch_size=batch_size)
    for i in range(0, g.shape[0] // chunk_size):
        pca.partial_fit(g[i * chunk_size: (i + 1) * chunk_size, :])

    Xtransformed = None
    for i in range(0, g.shape[0] // chunk_size):
        Xchunk = pca.transform(g[i * chunk_size: (i + 1) * chunk_size, :])
        if Xtransformed == None:
            Xtransformed = Xchunk
        else:
            Xtransformed = np.vstack((Xtransformed, Xchunk))
    #pca = pca.fit_transform(g)
    return Xtransformed


def main(bfile, n_comps, cpus, mem, extra_covs):
    (bim, fam, g) = read_geno(bfile, 0, cpus, max_memory=mem)
    cols = ['PC%d' % (x + 1) for x in range(n_comps)]
    pca = pd.DataFrame(do_pca(g, n_comps, None, 1000), columns=cols)
    cols = pca.columns.tolist()
    pca['iid'] = fam.iid.tolist()
    pca['fid'] = fam.fid.tolist()
    if extra_covs is not None:
        extra = pd.read_table(extra_covs, delim_whitespace=True)
        if 'FID' in extra.columns.tolist():
            extra.rename(columns={'FID': 'fid', 'IID': 'iid'}, inplace=True)
        pca = pca.merge(extra, on=['fid', 'iid'])
    ordered_cols = ['fid', 'iid'] + cols
    pca = pca.loc[:, ordered_cols]
    pca.to_csv('%s.pca' % bfile, sep=' ', header=False, index=False)
    return pca

# ----------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--bfile')
    parser.add_argument('-t', '--cpus', type=int)
    parser.add_argument('-m', '--mem', type=int)
    parser.add_argument('-c', '--comps', type=int)
    parser.add_argument('-e', '--extra_covs', default=None)
    args = parser.parse_args()
    main(args.bfile, args.comps, args.cpus, args.mem, args.extra_covs)
