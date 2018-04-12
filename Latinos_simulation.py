import argparse
import math
import gzip
import matplotlib.pylab as plt
import msprime
import pandas as pd
import os

from skpca import main as skpca
from utilities4cotagging_old import executeLine

plt.style.use('ggplot')


def make_plink(vcf_filename, plink_exe, threads=1, split=False,
               pops=['AFR', 'EUR']):
    sed = "sed s'/_//'g %s > temp; mv temp %s" % (vcf_filename, vcf_filename)
    executeLine(sed)
    prefix = vcf_filename[: vcf_filename.rfind('.')]
    line = ('%s --vcf %s --keep-allele-order --allow-no-sex --make-bed --out %s'
            ' --threads %d')
    executeLine(line % (plink_exe, vcf_filename, prefix, threads))
    df = pd.read_table('%s.bim' % prefix, delim_whitespace=True, header=None)
    df.loc[:, 1] = ['SNP%d' % x for x in range(1, df.shape[0] + 1)]
    df.to_csv('%s.bim' % prefix, sep='\t', index=False, header=False)
    if split:
        split_line = ('%s --bfile %s --keep %s.keep --keep-allele-order '
                      '--allow-no sex --make-bed --out %s --threads %d')
        x = 0
        for label, nhaps in split:
            if label in pops:
                diploid = nhaps/2
                col = ['msp%d' % i for i in range(x, diploid)]
                x += diploid
                options = {'path_or_buf': '%s.keep' % label, 'sep': ' ',
                           'header': False, 'index': False}
                pd.DataFrame({'fid': col, 'iid': col}).to_csv(**options)
                exec = split_line % (plink_exe, prefix, label, prefix, threads)
                executeLine(exec)


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

def main(nhaps=None, nvars=None, rec_map=None, maf=None, to_bed=False,
         threads=1, labels=['AFR', 'EUR', 'ASN', 'MX', 'AD'], split_out=False,
         plot_pca=True, focus_pops=['AFR', 'EUR']):
    if nhaps is None:
        nhaps = [45000] * 5
    if nvars is None:
        nvars = int(1e6)
    # First we set out the maximum likelihood values of the various parameters
    # given in Table 1.
    N_A = 11273
    N_B = 3104
    N_AF = 23721
    N_EU0 = 2271
    N_AS0 = 924
    N_MX0 = 800 # From Table 2 Gutenkust 2009
    # Times are provided in years, so we convert into generations.
    # from Jouganous et al. 2017:
    generation_time = 29
    T_AF = 312e3 / generation_time
    T_B = 125e3 / generation_time
    T_EU_AS = 42.3e3 / generation_time
    T_MX = 12.2e3 / generation_time  # from table 1 Gravel 2013
    T_AD = 18
    # confidence interval
    # We need to work out the starting (diploid) population sizes based on
    # the growth rates provided for these two population
    r_EU = 0.0019
    r_AS = 0.00309
    r_MX = 0.0050  # From Table 2 Gutenkust
    r_AD = 0.05
    N_EU = N_EU0 / math.exp(-r_EU * T_EU_AS)
    N_AS = N_AS0 / math.exp(-r_AS * T_EU_AS)
    N_MX = N_MX0 / math.exp(-r_MX * T_MX)
    N_AD0 = N_MX0 # / math.exp(-r_MX * T_AD)) / 3
    N_AD = (N_AD0 / math.exp(-r_AD * T_AD))# * 0.45
    # Migration rates during the various epochs.
    m_AF_B = 15.8e-5
    m_AF_EU = 1.1e-5
    m_AF_AS = 0.48e-5
    m_EU_AS = 4.19e-5
    m_MX_AD = 5e-5
    # Population IDs correspond to their indexes in the population
    # configuration array. Therefore, we have 0=YRI, 1=CEU, 2=CHB, and 3=MX
    # initially.
    population_configurations = [
        msprime.PopulationConfiguration(sample_size=nhaps[0], initial_size=N_AF
                                        ),
        msprime.PopulationConfiguration(sample_size=nhaps[1], initial_size=N_EU,
                                        growth_rate=r_EU),
        msprime.PopulationConfiguration(sample_size=nhaps[2], initial_size=N_AS,
                                        growth_rate=r_AS),
        msprime.PopulationConfiguration(sample_size=nhaps[3], initial_size=N_MX,
                                        growth_rate=r_MX),
        msprime.PopulationConfiguration(sample_size=nhaps[4], initial_size=N_AD,
                                        growth_rate=r_AD)
    ]
    # Migrations           AFR         EUR         ASN     MX      AD
    migration_matrix = [[   0,      m_AF_EU,    m_AF_AS,    0,      0],  # AFR
                        [m_AF_EU,       0,      m_EU_AS,    0,      0],  # EUR
                        [m_AF_AS,    m_EU_AS,       0,      0,      0],  # ASN
                        [   0,          0,          0,      0,   m_MX_AD],  # MX
                        [   0,          0,          0,   m_MX_AD,   0]]  # AD

    demographic_events = [
        # Slaves arrival
        msprime.MassMigration(time=13.758620689655173, source=4, destination=0,
                              proportion=0.098),
        # Colonials arrival
        msprime.MassMigration(time=T_AD-2, source=4, destination=1,
                              proportion=0.443),
        # Admixed fraction grow from N_AD0 at rate r_AD at time T_MX
        msprime.PopulationParametersChange(
            time=T_AD, initial_size=N_AD0, growth_rate=r_AD, population_id=4),
        # As the admixed merge to MX, turn off their migration rates
        msprime.MigrationRateChange(time=T_AD, rate=0, matrix_index=(3, 4)),
        msprime.MigrationRateChange(time=T_AD, rate=0, matrix_index=(4, 3)),
        # Admixed fraction merges with MX trunk
        msprime.MassMigration(time=T_AD, source=4, destination=3,
                              proportion=1.0),#0.459),
        # switch to standard coalescent
        #msprime.SimulationModelChange(T_AD, msprime.StandardCoalescent(1)),
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
    # dp = msprime.DemographyDebugger(
    #     Ne=N_A,
    #     population_configurations=population_configurations,
    #     migration_matrix=migration_matrix,
    #     demographic_events=demographic_events)
    # dp.print_history()
    # with open('demography.txt', 'w') as fn:
    #     dp.print_history(output=fn)
    if rec_map is not None:
        rmap = msprime.RecombinationMap.read_hapmap(rec_map)
        nvars = None
        rr = None
    else:
        rmap = None
        rr = 2e-8
    settings = {
        #'model': msprime.DiscreteTimeWrightFisher(0.25),
        'population_configurations': population_configurations,
        'migration_matrix': migration_matrix,
        'recombination_rate': rr,
        'demographic_events': demographic_events,
        'mutation_rate': 1.44e-8,  # according to 10.1371/journal.pgen.1004023
        'recombination_map': rmap,
        'length': nvars
    }
    vcf_filename = "OOA_Latino.vcf.gz"
    if not os.path.isfile('Latino.hdf5'):
        ts = msprime.simulate(**settings)
        print("Original file contains ", ts.get_num_mutations(), "mutations")
        if maf is not None:
            ts = strip_singletons(ts, maf)
        print("New file contains ", ts.get_num_mutations(), "mutations")
        ts.dump('Latino.hdf5', True)
        with open(vcf_filename, "w") as vcf_file:
            ts.write_vcf(vcf_file, 2)
    else:
        ts = msprime.load('Latino.hdf5')

    if to_bed is not None:
        if split_out:
            split = zip(labels, nhaps)
        else:
            split = False
        make_plink(vcf_filename, to_bed, threads, split, focus_pops)
        if plot_pca:
            pca = skpca(vcf_filename[: vcf_filename.rfind('.')], 2, threads,
                        None, None)
            count = 0
            for i, haps in enumerate(nhaps):
                haps = haps // 2
                pca.loc[count:(count + haps - 1), 'continent'] = labels[i]
                count += haps
            colors = iter(['k', 'b', 'y', 'r', 'g', 'c'])
            fig, ax = plt.subplots()
            for c, df in pca.groupby('continent'):
                df.plot.scatter(x='PC1', y='PC2', c=next(colors), ax=ax, label=c)
            plt.savefig('simulationPCA.pdf')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nhaps', help='Number of haplotypes per population.',
                        nargs='+', type=int)
    parser.add_argument('--labels', help='labels of population in nhaps order',
                        nargs='+')
    parser.add_argument('--rec_map', default=None)
    parser.add_argument('--nvars', default=1e4, type=int)
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--to_bed', default=None, help=(
        'Transform the vcf into plink binaries. You have to provide the path to'
        ' plink here'))
    parser.add_argument('--threads', default=1, type=int)
    parser.add_argument('--maf', default=0.01, type=int)
    parser.add_argument('--split_output', default=False, action='store_true')
    parser.add_argument('--plot_pca', default=False, action='store_true')
    parser.add_argument('--focus_pops', help='labels of population to focus on',
                        nargs='+')
    args = parser.parse_args()
    main(nhaps=args.nhaps, nvars=args.nvars, maf=args.maf, to_bed=args.to_bed,
         threads=args.threads, labels=args.labels, split_out=args.split_output,
         plot_pca=args.plot_pca, focus_pops=args.focus_pops)

