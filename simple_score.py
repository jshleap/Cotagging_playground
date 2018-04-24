import pandas as pd
from utilities4cotagging import read_geno
import argparse
import numpy as np


def main(args):
    causals = pd.read_table('%s.causaleff' % args.label, delim_whitespace=True)
    (bim, fam, g) = read_geno(args.bfile, 0, args.cpus, max_memory=args.mem)
    clump = pd.read_table(args.clump, delim_whitespace=True)
    prop = int(args.clump.split('.')[0])
    sumstats = pd.read_table(args.sumstats, delim_whitespace=True)
    over_gwsig = sumstats[sumstats.P <= 1E-8]
    if over_gwsig.empty:
        tp = 0
        fp = 0
    else:
        tp = over_gwsig[over_gwsig.SNP.isin(causals.snp.tolist())].shape[0]
        fp = over_gwsig.shape[0] - tp
    pheno = pd.read_table(args.pheno, delim_whitespace=True, names=['fid','iid',
                                                                    'pheno'])
    sub = sumstats.merge(clump, on=['CHR', 'SNP', 'BP', 'P'])
    sub['i'] = bim[bim.snp.isin(sub.SNP)].i.tolist()
    fam['prs'] = g[:, sub.i.values].dot(sub.BETA).compute(num_workers=args.cpus)
    fam.to_csv('%s.prs' % args.bfile, sep='\t', header=True, index=False)
    sub_pheno = pheno[pheno.iid.isin(fam.iid)]
    r2 = np.corrcoef(fam.prs.values, sub_pheno.pheno)[1, 0] ** 2
    with open('proportions.tsv', 'a') as F:
        # output have | proportion | r2 | TP | FP | ncausal
        F.write('%d\t%f\t%d\t%d\t%d\n' % (prop, r2, tp, fp, causals.shape[0]))




# ----------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--bfile')
    parser.add_argument('-c', '--clump')
    parser.add_argument('-s', '--sumstats')
    parser.add_argument('-p', '--pheno')
    parser.add_argument('-t', '--cpus', type=int)
    parser.add_argument('-m', '--mem', type=int, default=None)
    parser.add_argument('-l', '--label', default='AFR')

    args = parser.parse_args()
    main(args)
