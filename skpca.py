from sklearn.decomposition import PCA
from utilities4cotagging import read_geno
import pandas as pd
import sys

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


def main():
    (bim, fam, g) = read_geno(sys.argv[1], 0, int(sys.argv[2]),
                              max_memory=int(sys.argv[3]))
    pca = pd.DataFrame(do_pca(g, int(sys.argv[4])))
    cols = pca.columns.tolist()
    pca['iid'] = fam.iid.tolist()
    pca['fid'] = fam.fid.tolist()
    ordered_cols = ['fid','iid'] + cols
    pca.loc[:, ordered_cols].to_csv('%s.pca' % sys.argv[1], sep=' ',
                                    header=False, index=False)


main()
