from sklearn.decomposition import IncrementalPCA as ipca
from sklearn.decomposition import PCA
from utilities4cotagging import *
import pandas as pd
import numpy as np
import argparse
import psutil


def do_pca(g, n_comp, batch_size=None, nchunks=None, memory=None, threads=8):
    """
    Perform a PCA on the genetic array and return n_comp of it

    :param g: Genotype array
    :param n_comp: Number of components sought
    :return: components array
    """
    memory = psutil.virtual_memory().available if memory is None else memory
    cache = Chest(available_memory=memory)
    if nchunks is None:
        pca = PCA(n_components=n_comp)
        Xtransformed = pca.fit_transform(g)
    else:
        chunk_size = g.shape[0] // nchunks
        pca = ipca(n_components=n_comp, batch_size=batch_size)
        for i in range(0, g.shape[0] // chunk_size):
            dask_options = dict(num_workers=threads, cache=cache,
                                pool=ThreadPool(threads))
            with ProgressBar(), dask.set_options(**dask_options):
                x = da.compute(g[i * chunk_size: (i + 1) * chunk_size, :])
            pca.partial_fit(x)

        Xtransformed = None
        for i in range(0, g.shape[0] // chunk_size):
            Xchunk = pca.transform(g[i * chunk_size: (i + 1) * chunk_size, :])
            if Xtransformed == None:
                Xtransformed = Xchunk
            else:
                Xtransformed = np.vstack((Xtransformed, Xchunk))
    return Xtransformed


def main(bfile, n_comps, cpus, mem, extra_covs, partial, keep):
    (bim, fam, g) = read_geno(bfile, 0, cpus, max_memory=mem)
    if keep is not None:
        keep = pd.read_table(keep, delim_whitespace=True, header=None, names=[
            'fid', 'iid'])
        idx = fam[fam.iid.isin(keep.iid.values.tolist())].i.values
        g = g[idx, :]
    cols = ['PC%d' % (x + 1) for x in range(n_comps)]
    pca = pd.DataFrame(do_pca(g, n_comps, None, partial, mem, cpus), columns=cols)
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
    parser.add_argument('-p', '--partial', default=None, type=int)
    parser.add_argument('-k', '--keep', default=None)
    args = parser.parse_args()
    main(args.bfile, args.comps, args.cpus, args.mem, args.extra_covs,
         args.partial, args.keep)
