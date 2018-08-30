"""
Simulate OOA with natives
"""
import msprime
import sys
import math
import argparse
import matplotlib.pylab as plt, matplotlib
import numpy as np
from utilities4cotagging_old import executeLine
import pandas as pd


def out_of_africa_with_native(n_natives=1, nhaps=None, recomb=None,
                              nvars=None, debug=False):
    """

    Simulate the OOA with 7 native population

    :param str recomb: recombination map. If none, will use the param values
    :param tuple nvars: number of variants to simulate
    """
    if nhaps is None:
        nhaps = [45000] * (n_natives + 3)
    if nvars is None:
        nvars = int(1e6)
    assert n_natives > 0 # at least one native pop must be provided
    assert len(nhaps) == (n_natives + 3) # AFR, EUR, ASN
    # First we set out the maximum likelihood values of the various parameters
    # given in Table 1.
    N_A = 11273
    N_B = 3104
    N_AF = 23721
    N_EU0 = 2271
    N_AS0 = 924
    N_MX0 = 800 # From Table 2 Gutenkust 2009
    N_O0 = 500
    # Times are provided in years, so we convert into generations.
    # from Jouganous et al. 2017:
    generation_time = 29
    T_AF = 312e3 / generation_time
    T_B = 125e3 / generation_time
    T_EU_AS = 42.3e3 / generation_time
    T_MX = 12.2e3 / generation_time  # from table 1 Gravel 2013
    T_O = 12e3 / generation_time
    # confidence interval
    # We need to work out the starting (diploid) population sizes based on
    # the growth rates provided for these two population
    r_EU = 0.0019
    r_AS = 0.00309
    r_MX = 0.0050  # From Table 2 Gutenkust
    r_O = r_MX
    N_EU = N_EU0 / math.exp(-r_EU * T_EU_AS)
    N_AS = N_AS0 / math.exp(-r_AS * T_EU_AS)
    N_MX = N_MX0 / math.exp(-r_AS * T_MX)
    N_O = N_O0 / math.exp(-r_O * T_O)
    # Migration rates during the various epochs.
    m_AF_B = 15.8e-5
    m_AF_EU = 1.1e-5
    m_AF_AS = 0.48e-5
    m_EU_AS = 4.19e-5
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
            msprime.PopulationConfiguration(sample_size=nhaps[4 + x],
                                            initial_size=N_O, growth_rate=r_O)
            for x in range(n_natives)]
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
        # Natives grow from N_MX0 at rate r_MX at time T_MX
        msprime.PopulationParametersChange(
            time=T_MX, initial_size=N_MX0, growth_rate=r_MX, population_id=3),
        # Natives merge into asian trunk
        msprime.MassMigration(time=T_MX, source=3, destination=2, proportion=1.0
                              ),
        # As the natives merge to the asians, turn off their growth rates
        msprime.PopulationParametersChange(
            time=T_MX, initial_size=N_MX0, growth_rate=0, population_id=3),
        # Asians grow from N_AS0 at rate r_AS at time T_EU_AS
        msprime.PopulationParametersChange(
            time=T_EU_AS, initial_size=N_AS0, growth_rate=r_AS, population_id=2
        ),
        # Merge asian trunk wih european
        msprime.MassMigration(time=T_EU_AS, source=2, destination=1,
                              proportion=1.0),
        # As the Asians merge to EUR, turn off their migration rates
        msprime.MigrationRateChange(time=T_EU_AS, rate=0, matrix_index=(1, 2)),
        msprime.MigrationRateChange(time=T_EU_AS, rate=0, matrix_index=(2, 1)),
        msprime.MigrationRateChange(time=T_EU_AS, rate=0, matrix_index=(0, 2)),
        msprime.MigrationRateChange(time=T_EU_AS, rate=0, matrix_index=(2, 0)),
        # As the Asians merge to EUR, turn off their growth rates
        msprime.PopulationParametersChange(
            time=T_EU_AS, initial_size=N_AS0, growth_rate=0, population_id=2),
        # Europeans grow from N_EU0 at rate r_EU at time T_EU_AS
        msprime.PopulationParametersChange(
            time=T_EU_AS, initial_size=N_EU0, growth_rate=r_EU, population_id=1
        ),
        # Pop 1 (EUR/AS) size change at time T_B
        msprime.PopulationParametersChange(time=T_B, initial_size=N_EU0,
                                           growth_rate=0, population_id=1),
        # YRI merges with B at T_B
        msprime.MassMigration(time=T_B, source=1, destination=0, proportion=1.0
                              ),
        # Set migrations to 0
        msprime.MigrationRateChange(time=T_B, rate=0),
        # Pop 0 (AFR) size change at time T_B
        msprime.PopulationParametersChange(time=T_B, initial_size=N_B,
                                           growth_rate=0, population_id=0),
        # msprime.MigrationRateChange(time=T_B, rate=0),
        # Size changes to N_A at T_AF
        msprime.PopulationParametersChange(time=T_AF, initial_size=N_A,
                                           population_id=0)
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
    line = '%s --vcf %s --make-bed --out %s --threads %d'
    executeLine(line % (plink_exe, vcf_filename, prefix, threads))
    df = pd.read_table('%s.bim' % prefix, delim_whitespace=True, header=None)
    df.loc[:, 1] = ['SNP%d' % x for x in range(1, df.shape[0] + 1)]
    df.to_csv('%s.bim' % prefix, sep='\t', index=False, header=False)


def strip_singletons(ts, maf):
    """
    TODO: include maf filtering... done??
    modified from Jerome's
    :param ts:
    :return:
    """
    n = ts.get_sample_size()
    sites = msprime.SiteTable()
    mutations = msprime.MutationTable()
    for tree in ts.trees():
        for site in tree.sites():
            assert len(site.mutations) == 1  # Only supports infinite sites muts.
            mut = site.mutations[0]
            f = tree.get_num_leaves(mut.node) / n
            if (tree.num_samples(mut.node) > 1) and (f > maf):
                site_id = sites.add_row(
                    position=site.position,
                    ancestral_state=site.ancestral_state)
                mutations.add_row(
                    site=site_id, node=mut.node, derived_state=mut.derived_state
                )
    tables = ts.dump_tables()
    new_ts = msprime.load_tables(
        nodes=tables.nodes, edges=tables.edges, sites=sites, mutations=mutations
    )
    return new_ts


def main(args):
    one_native_settings = out_of_africa_with_native(n_natives=args.n_natives,
                                                    nhaps=args.nhaps,
                                                    recomb=args.recomb_map,
                                                    nvars=args.nvars,
                                                    debug=args.debug)
    ts = msprime.simulate(**one_native_settings)
    print("Original file contains ", ts.get_num_mutations(), "mutations")
    ts = strip_singletons(ts)
    print("New file contains ", ts.get_num_mutations(), "mutations")
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
    parser.add_argument('--maf', default=0.05, type=int)
    args = parser.parse_args()
    main(args)