'''
Refactoring of Dr. Alicia Martin's simulate_prs.py code
https://github.com/armartin/ancestry_pipeline/blob/master/simulate_prs.py
'''
## Libraries ###################################################################
from joblib import dump, load, Parallel, delayed
import dill, sys, os, math, argparse
from collections import defaultdict
from datetime import datetime
from random import sample
from scipy import stats
from tqdm import tqdm
import msprime as ms
import pandas as pd
import numpy as np
P = dill
################################################################################
    
## Functions ###################################################################

#----------------------------------------------------------------------
def current_time():
    return(' [' + datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S') + ']')

#----------------------------------------------------------------------
def out_of_africa(prefix, nhaps, recomb):
    """
    Specify the demographic model used in these simulations based on Dr. 
    Jouganous optimization of Gravel et. al's 2011 model. Function taken 
    and modified from Dr. Alicia Martin 
    """
    if os.path.isfile(prefix+'.simulation.hdf5'):
        simulation = ms.load(prefix+'.simulation.hdf5')
        line = 'Simulation %s has been DONE! Loading %s.simulation.hdf5'
        print(line%(prefix, prefix))
    else:
        ## First we set out the maximum likelihood values of the various 
        ## parameters given in Gravel et al, 2017 Table 2 but updated to 
        ## Dr. Jouganous work
        N_A = 11273
        N_B = 3104
        N_AF = 23721
        N_EU0 = 2271
        N_AS0 = 924
        ## Times are provided in years, so we convert into generations.
        generation_time = 29 # according to doi:10.1086/302770
        T_AF = 312e3 / generation_time
        T_B = 125e3 / generation_time
        T_EU_AS = 42.3e3 / generation_time
        ## We need to work out the starting (diploid) population sizes based on
        ## the growth rates provided for these two populations
        r_EU = 0.00196
        r_AS = 0.00309
        N_EU = N_EU0 / math.exp(-r_EU * T_EU_AS)
        N_AS = N_AS0 / math.exp(-r_AS * T_EU_AS)
        ## Migration rates during the various epochs.
        m_AF_B = 15.80e-5
        m_AF_EU = 1.10e-5
        m_AF_AS = 0.48e-5
        m_EU_AS = 4.19e-5
        
        ## Population IDs correspond to their indexes in the population
        ## configuration array. Therefore, we have 0=YRI, 1=CEU and 2=CHB
        ## initially.
        population_configurations = [
            ms.PopulationConfiguration(
                sample_size=nhaps[0], initial_size=N_AF),
            ms.PopulationConfiguration(
                sample_size=nhaps[1], initial_size=N_EU, growth_rate=r_EU),
            ms.PopulationConfiguration(
                sample_size=nhaps[2], initial_size=N_AS, growth_rate=r_AS)
        ]
        ## define the migration matrix
        migration_matrix = [
            [      0, m_AF_EU, m_AF_AS],
            [m_AF_EU,       0, m_EU_AS],
            [m_AF_AS, m_EU_AS,       0],
        ]
        ## define the demographic events (mergers and splits)
        demographic_events = [
            ## CEU and CHB merge into B with rate changes at T_EU_AS
            ms.MassMigration(
                time=T_EU_AS, source=2, destination=1, proportion=1.0),
            ms.MigrationRateChange(time=T_EU_AS, rate=0),
            ms.MigrationRateChange(
                time=T_EU_AS, rate=m_AF_B, matrix_index=(0, 1)),
            ms.MigrationRateChange(
                time=T_EU_AS, rate=m_AF_B, matrix_index=(1, 0)),
            ms.PopulationParametersChange(
                time=T_EU_AS, initial_size=N_B, growth_rate=0, population_id=1),
            ## Population B merges into YRI at T_B
            ms.MassMigration(
                time=T_B, source=1, destination=0, proportion=1.0),
            ## Size changes to N_A at T_AF
            ms.PopulationParametersChange(
                time=T_AF, initial_size=N_A, population_id=0)
        ]
        ## Use the demography debugger to print out the demographic history
        ## that we have just described.
        dp = ms.DemographyDebugger(
            Ne=N_A,
            population_configurations=population_configurations,
            migration_matrix=migration_matrix,
            demographic_events=demographic_events)
        dp.print_history()
        with open('%s.demography.txt'%(prefix),'w') as fn:
            dp.print_history(output=fn)
        
        settings = {
            'population_configurations': population_configurations, 
            'migration_matrix': migration_matrix,
            'demographic_events': demographic_events,
            'mutation_rate': 1.44e-8, #according to 10.1371/journal.pgen.1004023
            'recombination_map': ms.RecombinationMap.read_hapmap(recomb)
        }
        print('Starting Simulation...\t%s'%(current_time()))
        simulation = ms.simulate(**settings)
        simulation.dump('%s.simulation.hdf5'%(prefix), True)
        print('Simulation %s DONE!\t%s'%(prefix, current_time()))
        return simulation
#----------------------------------------------------------------------
def MutInfo(simulation, nhaps, out, causal_mut_index):
    """
    Populate mut info dictionary. Intended to avoid repeating code in true_prs 
    and infer prs
    """
    causal_mutations = set()
    mut_info = {} # index -> position, afr count, eur count, eas count
    ## get populations samples and loop over them
    samp = simulation.get_samples        
    pops = enumerate(samp(population_id=x) for x in range(len(nhaps)))
    if not os.path.isfile('%s_mutInfo.pickle'%(out)):
        print('Identifying causal mutations...\t%s)'%(current_time()))
        for pop_count, pop_leaves in pops:
            ## subset the simulation into its idividual population samples 
            for tree in simulation.trees(tracked_leaves=pop_leaves):
                tracked = tree.get_num_tracked_leaves
                ## Select only causal mutations
                filtered_mut = (x for x in tree.mutations() if x.index in 
                                causal_mut_index)
                ## Store causal mutations
                for mutation in filtered_mut:
                    causal_mutations.add(mutation)
                    if pop_count == 0:
                        mut_info[mutation.index] = [mutation.position, 
                                                    tracked(mutation.node)]
                    else:
                        mut_info[mutation.index].append(tracked(mutation.node))
        with open('%s_mutInfo.pickle'%(out), 'wb') as F:
            P.dump(mut_info, F)        
    else:
        with open('%s_mutInfo.pickle'%(out), 'rb') as F:
            mut_info = P.load(F)
    mutnames = ['Pos','Node','Index']
    df = pd.DataFrame.from_records(list(causal_mutations), columns=mutnames)
    df = df.loc[:, ['Index', 'Pos']]
    mi = pd.DataFrame(mut_info, index=['Pos', 'AFR_count', 'EUR_count', 
                                       'EAS_count']).transpose()
    mi.loc[:, 'Index'] = mi.index
    df = df.merge(mi, on=['Index', 'Pos'])
    df.loc[:, 'Total'] = [simulation.get_sample_size()] * df.shape[0]    
    return causal_mutations, df

#----------------------------------------------------------------------
def true_prs(simulation, ncausal, h2, nhaps, out):
    """
    choose some number of causal alleles
    assign these alleles effect sizes
    from these effect sizes, compute polygenic risk scores for everyone
    """
    ## get total number of mutations
    num_mut = simulation.get_num_mutations()
    ## Set outfile name
    outfn = '%s_nhaps_%s_h2_%.2f_m_%d.sites.gz'
    outfn = outfn%(out, '_'.join(map(str, nhaps)), round(h2, 2), ncausal)    
    ## If already done load it
    if os.path.isfile(outfn):
        print('Sites already computed, reading\t%s'%(current_time()))
        df = pd.read_table(outfn, sep='\t')
    else:
        print('Reading all site info %s'%(current_time()))
        ## Get causal mutation indices evenly distributed
        causal_mut_index = np.linspace(0, num_mut-1, ncausal, dtype=int)
        ## go through each population's trees
        colnames = ['Index', 'Pos', 'AFR_count', 'EUR_count', 'EAS_count', 
                    'Total', 'beta']
        causal_mutations, df = MutInfo(simulation, nhaps, out, causal_mut_index)
        causal_effects = np.random.normal(loc=0, scale=h2/ncausal, size=len(
            causal_mutations))
        df.loc[:, 'beta'] = causal_effects
        df = df.loc[:, colnames]
        print('Writing all site info\t%s'%(current_time()))
        df.to_csv(outfn, sep='\t', index=False, header=True, compression='gzip')
    if os.path.isfile('%s.trueprs'%(out)):
        prs_true = P.load(open('%s.trueprs'%(out),'rb'))
    else:
        print('Computing true PRS\t%s'%(current_time()))
        filtered = enumerate((x for x in simulation.variants() if x.index in 
                              causal_mut_index))
        ## multiply vector of genotypes by beta for given variant
        prs_haps = [variant.genotypes * causal_effects[idx] for idx, variant in 
                    tqdm(filtered)]
        prs_haps = np.sum(prs_haps, axis=0)  
        prs_true = prs_haps[0::2] + prs_haps[1::2] #add to get individuals
        P.dump(prs_true, open('%s.trueprs'%(out),'wb'))
    return(prs_true)

#----------------------------------------------------------------------
def case_control(prs_true, h2, nhaps, prevalence, ncontrols, out):
    """
    get cases assuming liability threshold model
    get controls from non-cases in same ancestry
    """
    print('Defining cases/controls\t%s'%(current_time()))
    env_effect = np.random.normal(loc=0,scale=1-h2, size=int(sum(nhaps)/2))
    prs_norm = (prs_true - np.mean(prs_true)) / np.std(prs_true)
    env_norm = (env_effect - np.mean(env_effect)) / np.std(env_effect)
    total_liabi = (math.sqrt(h2) * prs_norm ) + (math.sqrt(1 - h2) * env_norm)
    np.savetxt('%s_all_liabilities.gz'%(out), total_liabi)
    eur_liability = total_liabi[nhaps[0]//2:(nhaps[0]+nhaps[1])//2]
    sorted_liability = sorted(eur_liability)
    cases = [i for (i, x) in enumerate(eur_liability) if x >= sorted_liability[
        int((1-prevalence)*len(sorted_liability))]]
    controls = set(range(nhaps[1]//2))
    func = controls.remove
    map(func, cases)
    controls = sample(controls, ncontrols)
    case_ids = [x + nhaps[0]//2 for x in cases]
    control_ids = sorted([x+nhaps[0]//2 for x in controls])
    return(case_ids, control_ids, prs_norm, env_norm)

#----------------------------------------------------------------------
def fisherPerPos(tup, num_controls, num_cases, cc_maf, p_thresh):
    """
    Helper function to parallelize fisher test
    :param tuple tup: tuple from dict.items()
    """
    cases, controls = tup[1]
    case_maf = min(cases/num_cases, (num_cases - cases / num_cases))
    control_maf = min(controls/num_controls, (num_controls - controls) /
                      num_controls)
    case_control_maf = min((cases + controls) / (num_cases + num_controls), 
                           (num_cases + num_controls - cases - controls) /
                           (num_cases + num_controls))
    if case_control_maf > cc_maf:
        contingency = [[cases, num_cases - cases], 
                       [controls, num_controls - controls]]
        (OR, p) = stats.fisher_exact(contingency) #OR, p-value
        if not np.isnan(OR) and not np.isinf(OR) and OR != 0 and p <= p_thresh:
            return tup[0], [OR, p]
            #num_var += 1   
    #progress_bar.update()
#----------------------------------------------------------------------
def run_gwas(simulation, diploid_cases, diploid_controls, p_threshold, cc_maf,
             nthreads=-1):
    """
    use cases and controls to compute OR, log(OR), and p-value for every variant
    """
    print('Running GWAS (%d cases, %d controls)\t%s'%(len(diploid_cases), 
                                                      len(diploid_controls), 
                                                      current_time()))

    #summary_stats = {} # position -> OR, p-value
    #case_control = {} # position -> ncases w mut, ncontrols w mut
    #updateCC = case_control.update
    cases = [2*x for x in diploid_cases] + [2*x+1 for x in diploid_cases]
    controls = [2*x for x in diploid_controls] + [2*x+1 for x in 
                                                  diploid_controls]
    if os.path.isfile('summary_stats.pickl'):
        print('Summary stats have been done, loading\t%s'%(current_time()))
        summary_stats = P.dump(stats, open('summary_stats.pickl','wb'))    
    else:
        print('Counting case and control mutations\t%s'%(current_time()))
        tracked_cases = simulation.trees(tracked_leaves=cases)
        tracked_controls = simulation.trees(tracked_leaves=controls)
        zipped = zip(tracked_cases, tracked_controls)
        case_control = {mutation.position: (tree_case.get_num_tracked_leaves(
            mutation.node), tree_cont.get_num_tracked_leaves(mutation.node)) 
                        for tree_case, tree_cont in zipped for mutation in 
                        tree_case.mutations()}        
        #for tree in simulation.trees(tracked_leaves=cases):
            #tracked = tree.get_num_tracked_leaves
            #updateCC({mutation.position: [tracked(mutation.node)] for mutation 
                     #in tree.mutations()})    
        #for tree in simulation.trees(tracked_leaves=cases):
            #for mutation in tree.mutations():
                #case_control[mutation.position] = [tree.get_num_tracked_leaves(mutation.node)]        
        #print('Counting control mutations\t%s'%(current_time()))
        #for tree in simulation.trees(tracked_leaves=controls):
            #tracked = tree.get_num_tracked_leaves
            #for mutation in tree.mutations():
                #case_control[mutation.position].append(tracked(mutation.node))
        # only keep sites with non-infinite or nan effect size with case and 
        # control
        # maf > .01
        #num_var = 0
        print('Computing fisher\'s exact test\t%s'%(current_time()))
        num_controls = float(len(controls))
        num_cases = float(len(cases))
        #progress_bar = tqdm.tqdm(total=len(focal_mutations))
        #num_threads = min(num_threads, len(focal_mutations))    
        ss = Parallel(n_jobs=nthreads, verbose=1)(delayed(fisherPerPos)(
            tup, num_controls, num_cases, cc_maf, p_threshold) for tup in 
                                            case_control.items())   
        summary_stats = dict(filter(lambda x: x is not None, ss))
        P.dump(stats, open('summary_stats.pickl','wb'))
    print('Done with GWAS! (%d amenable sites)\t%s'%(len(summary_stats), 
                                                     current_time()))  
    return(summary_stats, cases, controls)

#----------------------------------------------------------------------
def clump_variants(simulation, summary_stats, nhaps, r2_threshold, window_size):
    """
    perform variant clumping in a greedy fasion with p-value and r2 threshold in
    windows
    return only those variants meeting some nominal threshold
    
    1: make a dict of pos -> variant for subset of sites meeting criteria
    2: make an r2 dict of all pairs of snps meeting p-value threshold and in 
    same window
    """
    # make a list of SNPs ordered by p-value
    print('Subsetting variants to usable list\t%s'%(current_time()))
    usable_positions = {} # position -> variant (simulation indices)
    sim_pos_index = {}
    sim_muts = simulation.get_num_mutations()
    filtered = (var for var in tqdm(simulation.variants(), total=sim_muts)
                if variant.position in summary_stats)
    for variant in filtered:
        usable_positions[variant.position] = variant
        sim_pos_index[variant.position] = variant.index
    # order all snps by p-value
    ordered_positions = sorted(summary_stats.keys(), key=lambda x: 
                               summary_stats[x][-1])
    eur_subset = simulation.subset(list(range(nhaps[0], (nhaps[0]+nhaps[1]))))
    eur_index_pos = {}
    eur_pos_index = {}
    eur_muts = eur_subset.get_num_mutations()
    for mutation in tqdm(eur_subset.mutations(), total=eur_muts):
        eur_index_pos[mutation.index] = mutation.position
        eur_pos_index[mutation.position] = mutation.index
    ordered_eur_index = sorted(eur_index_pos.keys())
    ld_calc = ms.LdCalculator(eur_subset)   
    # compute LD and prune in order of significance (popping index of SNPs)
    filtered = (position for position in ordered_positions if position in 
                usable_positions)
    for position in filtered:
        r2_forward = ld_calc.get_r2_array(eur_pos_index[position], 
                                          direction=ms.FORWARD, 
                                          max_distance=125e3)
        ##identify next position in eur space
        for i in np.where(r2_forward > r2_threshold)[0]:
            usable_positions.pop(eur_index_pos[eur_pos_index[position]+i+1], 
                                 None) 
        r2_reverse = ld_calc.get_r2_array(eur_pos_index[position], 
                                          direction=ms.REVERSE, 
                                          max_distance=125e3)
        for i in np.where(r2_reverse > r2_threshold)[0]:
            usable_positions.pop(eur_index_pos[eur_pos_index[position]-i-1], 
                                 None)
    clumped_snps = set(usable_positions.keys())
    line = 'Starting SNPs: %d; SNPs after clumping: %d\t%s'
    print(line%(len(ordered_positions), len(clumped_snps), current_time()))
    return(clumped_snps, usable_positions)

#----------------------------------------------------------------------
def infer_prs(simulation, nhaps, clumped_snps, summary_stats, usable_positions, 
              h2, ncausal, out):
    """
    use clumped variants from biased gwas to compute inferred prs for everyone
    """
    
    outfn = '%s_nhaps_%s_h2_%.2f_m_%d.infer_sites.gz'
    outfn = outfn%(out, '_'.join(map(str, nhaps)), round(h2, 2), ncausal)  
    out = '%s_inferred'%(out)
    muts = simulation.get_num_mutations()
    if os.path.isfile('%s.prs'%(out)):
        print('Inferred PRSs already computed\t%s'%(current_time()))
        prs_true = P.load(open('%s.prs'%(out),'rb'))
    else:
        print('Computing inferred PRS\t%s'%(current_time()))
        filtered = (variant for variant in tqdm(simulation.variants(), 
                                                total=muts) 
                    if variant.position in usable_positions)    
        prs_haps = [variant.genotypes * math.log(
            summary_stats[variant.position][0]) for variant in filtered]
        prs_haps = np.sum(prs_haps, axis=0)
        prs_infer = prs_haps[0::2] + prs_haps[1::2] 
        P.dump(prs_infer, open('%s.prs'%(out),'rb'))
    ## If already done load it
    if os.path.isfile(outfn):
        print('Inferred sites already computed, reading\t%s'%(current_time()))
        df = pd.read_table(outfn, sep='\t')
    else:    
        colnames = ['Index', 'Pos', 'AFR_count', 'EUR_count', 'EAS_count', 
                    'Total', 'beta']        
        causal_mutations, df = MutInfo(simulation, nhaps, out, 
                                             usable_positions)    
        df.loc[:, 'beta'] = [summary_stats[pos][0] for pos in df.Pos]
        df = df.loc[:, colnames]
        print('Writing all site info\t%s'%(current_time()))    
        df.sort_values(['Index','Pos'], inplace=True)
        df.to_csv(outfn, sep='\t', index=False, header=True, compression='gzip')
    return(prs_infer)

#----------------------------------------------------------------------
def returnPop(idx, nhaps):
    """ Return AFR, EUR or EAS if index in haplotype range """
    if idx in range(nhaps[0]/2):
        return 'AFR'
    elif idx in range((nhaps[0]/2), (nhaps[0]/2) + (nhaps[1]/2)):
        return 'EUR'
    else:
        return 'EAS'   
    
#----------------------------------------------------------------------
def returnPheno(idx, cases, controls):
    """ Return AFR, EUR or EAS if index in haplotype range """    
    if idx in cases:
        return 1
    elif idx in controls:
        return 0
    else:
        return 'NA'    
#----------------------------------------------------------------------
def write_summaries(out, prs_true, prs_infer, nhaps, cases, controls, h2, 
                    ncausal, environment):
    print('Writing output!\t%s'%(current_time()))
    scaled_prs = math.sqrt(h2) * prs_true
    scaled_env = math.sqrt(1 - h2) * environment
    out_prs = '%s_nhaps_%s_h2_%.2f_m_%d.prs.gz'%(out, '_'.join(map(str, nhaps)),
                                                 round(h2, 2), ncausal)
    df = [{'Ind': ind + 1, 'Pop': returnPop(ind), 'PRS_true':prs_true[ind],
           'PRS_true_scaled':scaled_prs[ind], 'PRS_infer':prs_infer[ind], 
           'Pheno':returnPheno(ind, cases, controls),
           'Environment': scaled_env[ind]} for ind in range(len(prs_true))]
    df = pd.DataFrame(df)
    df.sort_values('Ind', inplace=True)
    df.to_csv(out_prs, index=False, compression='gzip')
    
    
#----------------------------------------------------------------------
def main(args):
    """
    Execute code
    """
    nhaps = [int(x) for x in args.nhaps.split(',')]
    recomb = args.recomb_map
    ncausal = args.ncausal
    
    # generate/load coalescent simulations
    if args.tree is None:
        (pop_config, mig_mat, demog) = out_of_africa(nhaps)
        simulation = simulate_ooa(pop_config, mig_mat, demog, recomb)
        simulation.dump('%s_nhaps_%s.hdf5'%(args.out, '_'.join(map(str, nhaps))
                                            ), True)
    else:
        simulation = ms.load(args.tree)
    print(simulation)
    print('Number of haplotypes: %s'%(','.join(map(str, nhaps))))
    print('Number of trees: %d'%(simulation.get_num_trees()))
    print('Number of mutations: %d'%(simulation.get_num_mutations()))
    print('Sequence length: %d'%(simulation.get_sequence_length()))

    
    prs_true = true_prs(simulation, args.ncausal, args.h2, nhaps, args.out)
    cc = case_control(prs_true, args.h2, nhaps, args.prevalence, args.ncontrols, 
                      args.out)
    cases_diploid, controls_diploid, prs_norm, environment = cc
    gw = run_gwas(simulation, cases_diploid, controls_diploid, args.p_threshold,
                  args.cc_maf, args.threads)
    summary_stats, cases_haploid, controls_haploid = gw
    cl = clump_variants(simulation, summary_stats, nhaps, args.r2, 
                        args.window_size)
    clumped_snps, usable_positions = cl
    prs_infer = infer_prs(simulation, nhaps, clumped_snps, summary_stats, 
                          usable_positions, args.h2, args.ncausal, args.out)
    write_summaries(args.out, prs_true, prs_infer, nhaps, cases_diploid,
                    controls_diploid, args.h2, args.ncausal, environment)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tree', help='msprime simulated tree')
    parser.add_argument('--nhaps', help='AFR,EUR,EAS', 
                        default='400000,400000,400000')
    parser.add_argument('-r', '--recomb_map', default='/Users/jshleap/' + 
                        'Playground/AdmixGWAS/Sims/genetic_map_chr20_b36' +
                        '_mod_txt')
    parser.add_argument('-c', '--ncausal', type=int, default=200)
    parser.add_argument('-n', '--ncontrols', type=int, default=10000)
    parser.add_argument('-b', '--h2', type=float, default=float(2)/3)
    parser.add_argument('-p', '--prevalence', type=float, default=0.05)
    parser.add_argument('-d', '--p_threshold', type=float, default=0.01)
    parser.add_argument('-m', '--cc_maf', type=float, default=0.01)
    parser.add_argument('-l', '--r2', type=float, default=0.5)
    parser.add_argument('-w', '--window_size', type=int, default=250e3)
    parser.add_argument('-o', '--out', default='sim0')
    parser.add_argument('-e', '--threads', default=-1, type=int)
    
    args = parser.parse_args()
    main(args)
