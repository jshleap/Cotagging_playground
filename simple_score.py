import argparse
from utilities4cotagging import *


def main(args):
    if args.mem is not None:
        memory = args.mem
    else:
        memory = psutil.virtual_memory().available
    cache = Chest(available_memory=memory)
    dask_options = dict(num_workers=args.cpus, cache=cache,
                        pool=ThreadPool(args.cpus))
    causals = pd.read_table('%s.causaleff' % args.label, delim_whitespace=True)
    (bim, fam, g) = read_geno(args.bfile, 0, args.cpus, max_memory=args.mem)
    if args.normalize:
        g = (g - g.mean(axis=0)) / g.std(axis=0)
    clump = pd.read_table(args.clump, delim_whitespace=True)
    try:
        prop = int(args.clump.split('.')[0])
    except ValueError:
        prop = int(args.clump.split('.')[0].split('_')[1])
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
    sub = sumstats.merge(clump, on=['CHR', 'SNP', 'BP'])
    sub['i'] = bim[bim.snp.isin(sub.SNP)].i.tolist()
    fam['prs'] = g[:, sub.i.values].dot(sub.BETA).compute(**dask_options)
    fam.to_csv('%s.prs' % args.bfile, sep='\t', header=True, index=False)
    sub_pheno = pheno[pheno.iid.isin(fam.iid)]
    r2 = np.corrcoef(fam.prs.values, sub_pheno.pheno)[1, 0] ** 2
    r2 = r2 * args.weight
    with open('%s.tsv' % args.prefix, 'a') as F:
        # output have | proportion | r2 | TP | FP | ncausal | label
        F.write('%d\t%f\t%d\t%d\t%d\t%s\n' % (prop, r2, tp, fp, causals.shape[0],
                                              args.label))




# ----------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-P', '--prefix', default='proportions')
    parser.add_argument('-b', '--bfile')
    parser.add_argument('-c', '--clump')
    parser.add_argument('-s', '--sumstats')
    parser.add_argument('-p', '--pheno')
    parser.add_argument('-t', '--cpus', type=int)
    parser.add_argument('-m', '--mem', type=int, default=None)
    parser.add_argument('-N', '--normalize',action='store_true', default=False)
    parser.add_argument('-l', '--label', default='AFR')
    parser.add_argument('-w', '--weight', default=1, type=float)


    args = parser.parse_args()
    main(args)
