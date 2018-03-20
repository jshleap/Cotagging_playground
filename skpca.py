from sklearn.decomposition import PCA
from utilities4cotagging import read_geno
import pandas as pd
import sys
import argparse

def do_pca(g, n_comp):
    """
    Perform a PCA on the genetic array and return n_comp of it

    :param g: Genotype array
    :param n_comp: Number of components sought
    :return: components array
    """
    pca = PCA(n_components=n_comp)
    pca = pca.fit_transform(g)
    return pca


def main(args):
    (bim, fam, g) = read_geno(args.bfile, 0, args.cpus, max_memory=args.mem)
    pca = pd.DataFrame(do_pca(g, args.comps))
    cols = pca.columns.tolist()
    pca['iid'] = fam.iid.tolist()
    pca['fid'] = fam.fid.tolist()
    ordered_cols = ['fid','iid'] + cols
    pca.loc[:, ordered_cols].to_csv('%s.pca' % args.bfile, sep=' ',
                                    header=False, index=False)

# ----------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--bfile')
    parser.add_argument('-t', '--cpus', type=int)
    parser.add_argument('-m', '--mem', type=int)
    parser.add_argument('-c', '--comps', type=int)
    args = parser.parse_args()
    main(args)
