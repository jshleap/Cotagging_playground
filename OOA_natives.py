"""
Simulate OOA with natives
"""
import msprime
import sys
import math
import argparse
import matplotlib.pylab as plt, matplotlib
import numpy as np
from utilities4cotagging import executeLine
import pandas as pd


def out_of_africa_with_native(n_natives=1, nhaps=None, recomb=None,
                              nvars=None, debug=False):
    """

    Simulate the OOA with 7 native population

    :param str recomb: recombination map. If none, will use the param values
    :param tuple nvars: number of variamts to simulate
    """
    if nhaps is None:
        nhaps = [45000] * (n_natives + 3)
    if nvars is None:
        nvars = int(1e6)
    assert n_natives > 0 # at least one native pop must be provided
    assert len(nhaps) == (n_natives + 3) # AFR, EUR, ASN
    # First we set out the maximum likelihood values of the various parameters
    # given in Table 1.
    N_A = 7300
    N_B = 2100
    N_AF = 12300
    N_EU0 = 1350  # ==> modified so is within both table 1 and 2 confidence interval
    N_AS0 = 555  # ==> modified so is within both table 1 and 2 confidence interval
    N_MX0 = 800  # From table 2
    N_O0 = 500
    # Times are provided in years, so we convert into generations.
    generation_time = 25
    T_AF = 220e3 / generation_time
    T_B = 140e3 / generation_time
    T_EU_AS = 22e3 / generation_time  # modified so is within both table 1 and 2
    T_MX = 21.6e3 / generation_time  # from table 2
    T_O = 12e3 / generation_time
    # confidence interval
    # We need to work out the starting (diploid) population sizes based on
    # the growth rates provided for these two population
    r_EU = 0.0037  # ==> modified so is within both table 1 and 2 confidence interval
    r_AS = 0.0052  # ==> modified so is within both table 1 and 2 confidence interval
    r_MX = 0.0050  # from table 2
    r_O = r_MX
    N_EU = N_EU0 / math.exp(-r_EU * T_EU_AS)
    N_AS = N_AS0 / math.exp(-r_AS * T_EU_AS)
    N_MX = N_MX0 / math.exp(-r_AS * T_MX)
    N_O = N_O0 / math.exp(-r_O * T_O)
    # Migration rates during the various epochs.
    m_AF_B = 25e-5
    m_AF_EU = 3e-5
    m_AF_AS = 1.9e-5
    m_EU_AS = 11.55e-5
    # Population IDs correspond to their indexes in the population
    # configuration array. Therefore, we have 0=YRI, 1=CEU, 2=CHB, and 3=MX
    # initially.
    population_configurations = [
                                    msprime.PopulationConfiguration(
                                        sample_size=nhaps[0],
                                        initial_size=N_AF),
                                    msprime.PopulationConfiguration(
                                        sample_size=nhaps[1], initial_size=N_EU,
                                        growth_rate=r_EU),
                                    msprime.PopulationConfiguration(
                                        sample_size=nhaps[2], initial_size=N_AS,
                                        growth_rate=r_AS),
                                    msprime.PopulationConfiguration(
                                        sample_size=nhaps[3], initial_size=N_MX,
                                        growth_rate=r_MX)
                                ]
    if n_natives > 1:
        population_configurations += [
            msprime.PopulationConfiguration( sample_size=nhaps[4 + x],
                                             initial_size=N_O, growth_rate=r_O)
            for x in range( n_natives)]
    migration_matrix = [
                           [0, m_AF_EU, m_AF_AS] + [0] * n_natives,
                           [m_AF_EU, 0, m_EU_AS] + [0] * n_natives,
                           [m_AF_AS, m_EU_AS, 0] + [0] * n_natives,
                           [0, 0, 0] + [0] * n_natives
                       ]
    demographic_events = []
    if n_natives > 1:
        migration_matrix += [[0] * (n_natives + 3)] * n_natives
        demographic_events = [msprime.MassMigration(
            time=T_O, source=x, destination=3, proportion=1.0) for x in
            range(4, 4 + n_natives)]
    demographic_events += [
        # CEU and CHB merge into B with rate changes at T_EU_AS
        msprime.MassMigration(
            time=T_MX, source=3, destination=2, proportion=1.0),
        msprime.MassMigration(
            time=T_EU_AS, source=2, destination=1, proportion=1.0),
        msprime.MigrationRateChange(time=T_EU_AS, rate=0),
        msprime.MigrationRateChange(
            time=T_EU_AS, rate=m_AF_B, matrix_index=(0, 1)),
        msprime.MigrationRateChange(
            time=T_EU_AS, rate=m_AF_B, matrix_index=(1, 0)),
        msprime.PopulationParametersChange(
            time=T_EU_AS, initial_size=N_B, growth_rate=0, population_id=1),
        # Population B merges into YRI at T_B
        msprime.MassMigration(
            time=T_B, source=1, destination=0, proportion=1.0),
        msprime.MigrationRateChange(time=T_B, rate=0),
        # Size changes to N_A at T_AF
        msprime.PopulationParametersChange(
            time=T_AF, initial_size=N_A, population_id=0)
    ]
    dp = msprime.DemographyDebugger(
        Ne=N_A,
        population_configurations=population_configurations,
        migration_matrix=migration_matrix,
        demographic_events=demographic_events)
    if debug:
        dp.print_history()
    with open('demography.txt', 'w') as fn:
        dp.print_history(output=fn)
    if recomb is not None:
        rmap = msprime.RecombinationMap.read_hapmap(recomb)
        nvars = None
        rr = None
    else:
        rmap = None
        rr = 2e-8
    settings = {
        'population_configurations': population_configurations,
        'migration_matrix': migration_matrix,
        'recombination_rate': rr,
        'demographic_events': demographic_events,
        'mutation_rate': 1.44e-8,  # according to 10.1371/journal.pgen.1004023
        'recombination_map': rmap,
        'length': nvars
    }
    return settings


def aarons():
    # Set fontsize to 10
    matplotlib.rc('font', **{'family': 'sans-serif',
                             'sans-serif': ['Helvetica'],
                             'style': 'normal',
                             'size': 10})
    # Set label tick sizes to 8
    matplotlib.rc('xtick', labelsize=8)
    matplotlib.rc('ytick', labelsize=8)

    fname = 'private_singletons.txt'

    ss = [1, 10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    singleton_counts = {}
    for ns in ss:
        singleton_counts[ns] = []

    with open(fname, 'r') as f:
        for line in f.readlines():
            ll = line.split()
            singleton_counts[int(ll[0])].append(float(ll[1]))

    means = {}
    vars = {}
    for key in ss:
        means[key] = np.mean(singleton_counts[key])
        vars[key] = np.var(singleton_counts[key])

    lower = []
    mean = [means[k] for k in ss]
    upper = []

    for k in ss:
        lower.append(means[k] - 1.96 * np.sqrt(vars[k]))
        upper.append(means[k] + 1.96 * np.sqrt(vars[k]))

    fig = plt.figure(figsize=(6, 4), dpi=300)
    plt.clf()

    ax = plt.subplot(111)
    ax.plot(ss, mean, 'b')
    ax.plot(ss, lower, 'k--')
    ax.plot(ss, upper, 'k--')

    ax.set_xlim([0, 1000])
    ax.set_ylim([0, 5000])

    ax.set_xlabel('Number sampled')
    ax.set_ylabel('Number of private singletons')

    ax.legend(['mean num. singletons', '95 $\%$ CI'], loc=5)

    plt.savefig('number_of_singletons.pdf')


def privates(nativesample, ts):
    ## find private variants
    pop3ind = ts.get_sample_size() - nativesample
    num_singles = np.zeros(nativesample)
    fs_NA_privates = np.zeros(nativesample + 1)

    for v in ts.variants():
        g = v.genotypes
        gnonNA = g[:pop3ind]
        gNA = g[pop3ind:]
        if np.sum(gnonNA) == 0 and np.sum(gNA) > 0:
            fs_NA_privates[np.sum(gNA)] += 1
            if gNA.sum() == 1:
                num_singles[np.argmax(gNA)] += 1
    return fs_NA_privates, num_singles

# TODO: include the per popyulation splitting of the beds
def make_plink(vcf_filename, plink_exe, threads=1):
    sed = "sed s'/_//'g %s > temp; mv temp %s" % (vcf_filename, vcf_filename)
    executeLine(sed)
    prefix = vcf_filename[: vcf_filename.rfind('.')]
    line = '%s --vcf %s --make- --out %s --threads %d'
    executeLine(line  % (plink_exe, vcf_filename, prefix, threads))
    df = pd.read_table('%s.bim' % prefix, delim_whitespace=True, header=None)
    df.loc[:, 1] = ['SNP%d' % x for x in range(1, df.shape[0] + 1)]
    df.to_csv('%s.bim' % prefix, sep='\t', index=False, header=False)


def main(args):
    one_native_settings = out_of_africa_with_native(n_natives=args.n_natives,
                                                    nhaps=args.nhaps,
                                                    recomb=args.recomb_map,
                                                    nvars=args.nvars,
                                                    debug=args.debug)
    ts = msprime.simulate(**one_native_settings)
    print("There are {0} total variant sites\n".format(ts.get_num_mutations()))
    ts.dump('Natives_pops_%d.hdf5' % args.n_natives, True)
    vcf_filename = "OOA_natives.vcf"
    with open(vcf_filename, "w") as vcf_file:
        ts.write_vcf(vcf_file, 2)
    if args.to_bed is not None:
        make_plink(vcf_filename, args.to_bed, args.threads)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nhaps', help='Number of haplotypes per population.',
                        nargs='+', type=int)
    parser.add_argument('--recomb_map', default=None)
    parser.add_argument('--nvars', default=1e4, type=int)
    parser.add_argument('--n_natives', help=('Number of natives population '
                                             'splitting from MX'),
                        default=6, type=int)
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--to_bed', default=None,
                        help=('Transform the vcf into plink binaries. You have '
                              'to provide the path to plink here'))
    parser.add_argument('--threads', default=1, type=int)
    args = parser.parse_args()
    main(args)