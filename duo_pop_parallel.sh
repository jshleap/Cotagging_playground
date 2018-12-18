#!/usr/bin/env bash
# This program will duo_pop for each chromosome, and then pool the results
# and score. It makes use of a config file, and a template for the submission
# script that has too be pre-filled with the system-specific settings.
# USAGE:
#       bash duo_pop_parallel.sh configfile


TIMEFORMAT="Time elapsed: %R"
export TIMEFORMAT
parse_config_file(){
# Parse the config file courtesy of
# https://stackoverflow.com/users/1851270/antonio
shopt -s extglob
while IFS='= ' read -r lhs rhs
do
    if [[ ! ${lhs} =~ ^\ *# && -n ${lhs} ]]; then
        rhs="${rhs%%\#*}"    # Del in line right comments
        rhs="${rhs%%*( )}"   # Del trailing spaces
        rhs="${rhs%\"*}"     # Del opening string quotes
        rhs="${rhs#\"*}"     # Del closing string quotes
        declare ${lhs}="$rhs"
    fi
done < $1
}

qtrait_simulation(){
# Implementing the qtrait simulation as a single bash function but with a
# python call
python3 - << EOF
import os
import gc
import dask
import pickle
import numpy as np
import pandas as pd
from chest import Chest
from functools import reduce
from subprocess import Popen
from pandas_plink import read_plink
from psutil import virtual_memory as vm
from dask.diagnostics import ProgressBar
from multiprocessing.pool import ThreadPool

def estimate_chunks(shape, threads, memory=None):
    avail_mem = vm().available if memory is None else memory
    usage = (reduce(np.multiply, shape) * 8 / 1E6) * threads
    n_chunks = np.ceil(usage / avail_mem).astype(int)
    with np.errstate(divide='ignore', invalid='ignore'):
        estimated = tuple(np.array(shape) / n_chunks)  # Get chunk estimation
    return tuple(int(i) for i in min(shape, tuple(estimated)))

def read_geno(bfile, freq_thresh, threads, flip=False, check=False,
              max_memory=None):
    available_memory = max_memory if max_memory is not None else vm().available
    cache = Chest(available_memory=available_memory)
    (bim, fam, g) = read_plink(bfile)   # read the files using pandas_plink
    m, n = g.shape                      # get the dimensions of the genotype
    if check:
        g_std = g.std(axis=1)
        with ProgressBar():
            print('Removing invariant sites')
            with dask.set_options(pool=ThreadPool(threads)):
                idx = (g_std != 0).compute(cache=cache)
        g = g[idx, :]
        bim = bim[idx].copy()
        del g_std, idx
        gc.collect()
    mafs = g.sum(axis=1) / (2 * n) if flip or freq_thresh > 0 else None
    if flip:
        flips = np.zeros(bim.shape[0], dtype=bool)
        flips[np.where(mafs > 0.5)[0]] = True
        bim['flip'] = flips
        vec = np.zeros(flips.shape[0])
        vec[flips] = 2
        g = abs(g.T - vec)
        del flips
        gc.collect()
    else:
        g = g.T
    if freq_thresh > 0:
        print('Filtering MAFs smaller than', freq_thresh)
        print('    Genotype matrix shape before', g.shape)
        assert freq_thresh < 0.5
        good = (mafs < (1 - float(freq_thresh))) & (mafs > float(freq_thresh))
        with ProgressBar():
            with dask.set_options(pool=ThreadPool(threads)):
                good, mafs = dask.compute(good, mafs, cache=cache)
        g = g[:, good]
        bim = bim[good]
        bim['mafs'] = mafs[good]
        print('    Genotype matrix shape after', g.shape)
        del good
        gc.collect()
    bim = bim.reset_index(drop=True)    # Get the indices in order
    bim['i'] = bim.index.tolist()
    g = g.rechunk(estimate_chunks(g.shape, threads, memory=available_memory))
    del mafs
    gc.collect()
    return bim, fam, g

def true_prs(prefix, bfile, h2, ncausal, normalize=False, bfile2=None, f_thr=0.01,
             seed=None, causaleff=None, uniform=False, usepi=False, snps=None,
             threads=1, flip=False, check=False, max_memory=None, **kwargs):
    cache = Chest(available_memory=int(max_memory))
    # set random seed
    seed = np.random.randint(10000) if seed is None else seed
    np.random.seed(seed=seed)
    if isinstance(bfile, str):
        (bim, fam, g) = read_geno(bfile, f_thr, threads, flip=flip, check=check, max_memory=max_memory)
        if bfile2 is not None:
            (bim2, fam2, G2) = read_geno(bfile2, f_thr, threads, check=check, max_memory=max_memory)
            indices = bim[bim.snp.isin(bim2.snp)].i
            g = g[:, indices.tolist()]
            bim = bim[bim.i.isin(indices)].reset_index(drop=True)
            bim['i'] = bim.index.tolist()
    else:
        bim, fam, g = kwargs['bim'], kwargs['fam'], bfile
    g = g.rechunk(estimate_chunks(g.shape, threads, max_memory))
    pi = g.var(axis=0).mean() if usepi else 1
    if normalize:
        g = (g - g.mean(axis=0)) / g.std(axis=0)
    allele = '%s.alleles' % prefix
    totalsnps = '%s.totalsnps' % prefix
    allsnps = g.shape[1]
    h2_snp = h2 / (ncausal*pi)
    std = np.sqrt(h2_snp)
    par = dict(sep=' ', index=False, header=False)
    bim.snp.to_csv(totalsnps, **par)
    bim.loc[:, ['snp', 'a0']].to_csv(allele, **par)
    if ncausal > allsnps: ncausal = allsnps
    if causaleff is not None:
        cols = ['snp', 'beta']
        causals = bim[bim.snp.isin(causaleff.snp)].copy()
        c = cols if 'beta' in bim else 'snp'
        causals = causals.merge(causaleff.reindex(columns=cols), on=c)
        bim = bim.merge(causaleff, on='snp', how='outer')
    elif uniform:
        idx = np.linspace(0, bim.shape[0] - 1, num=ncausal, dtype=int)
        causals = bim.iloc[idx].copy()
        av_dist = (np.around(causals.pos.diff().mean() / 1000)).astype(int)
    elif snps is None:
        causals = bim.sample(ncausal, replace=False, random_state=seed).copy()
    else:
        causals = bim[bim.snp.isin(snps)].copy()
    if causaleff is None:
        pre_beta = np.repeat(std, ncausal) if ncausal <= 5 else np.random.normal(
            loc=0, scale=std, size=ncausal)
        causals['beta'] = pre_beta
        causals = causals.dropna(subset=['beta'])
        assert np.allclose(sorted(causals.beta.values), sorted(pre_beta))
    nc = causals.reindex(columns=['snp', 'beta'])
    bidx = bim[bim.snp.isin(nc.snp)].index.tolist()
    bim = bim.reindex(columns=['chrom', 'snp', 'cm', 'pos', 'a0', 'a1', 'i', 'mafs', 'flip'])
    bim.loc[bidx, 'beta'] = nc.beta.values.tolist()
    idx = bim.dropna(subset=['beta']).i.values
    causals = bim.dropna(subset=['beta'])
    causalfn = '%s.causaleff' % prefix
    causals.reindex(columns=['snp', 'pos', 'a0', 'beta']).to_csv(causalfn, index=False, sep=' ')
    if 'plink' in os.environ:
        print('using plink to score')
        causals.snp.to_csv('%ssnp.extract' % prefix, index=False, header=False)
        plink = os.environ["plink"]
        p_line = '%s --bfile %s --score %s 1 3 4 header sum center --out %s ' \
                 '--keep-allele-order --allow-no-sex --extract %ssnp.extract ' \
                 '--threads %d --memory %d'
        mem = max_memory/1000000
        p_line = p_line % (plink, bfile, causalfn, prefix, prefix, threads, mem)
        p = Popen(p_line, shell=True)
        p.communicate()
        gen_eff = pd.read_table('%s.profile' % prefix, delim_whitespace=True)
        cols = {'FID': 'fid', 'IID': 'iid', 'SCORESUM': 'gen_eff'}
        gen_eff = gen_eff.rename(columns=cols).reindex(columns=['fid', 'iid', 'gen_eff'])
        gen_eff['iid'], gen_eff['fid'] = gen_eff.iid.astype(str), gen_eff.fid.astype(str)
        fam['iid'], fam['fid'] = fam.iid.astype(str), fam.fid.astype(str)
        fam = fam.merge(gen_eff, on=['fid', 'iid'])
    else:
        dask_options = dict(num_workers=threads, cache=cache, pool=ThreadPool(
            threads))
        with ProgressBar(), dask.set_options(**dask_options):
            fam['gen_eff'] = g[:, idx].dot(causals.beta).compute()
    if causaleff is not None:
        try:
            assert sorted(bim.dropna(subset=['beta']).snp) == sorted(
                causaleff.snp)
        except:
            raise
    fam.to_csv('%s.full' % prefix, sep=' ', index=False)
    return g, bim, fam, causals

def create_pheno(prefix, h2, prs_true, covs=None, force_h2=False):
    noenv = True if h2 == 1 else False
    nind = prs_true.shape[0]
    if noenv:
        env_effect = np.zeros(nind)
    else:
        va = prs_true.gen_eff.var()
        std = np.sqrt((va - (va * h2)) / h2) if force_h2 else np.sqrt(max(1 - va, 0))
        env_effect = np.random.normal(loc=0, scale=std, size=nind)
    prs_true['env_eff'] = env_effect
    prs_true['PHENO'] = prs_true.gen_eff + prs_true.env_eff
    dim1 = prs_true.shape[0]
    if covs is not None:
        cov = pd.read_table(covs, delim_whitespace=True, header=None)
        covs_names = ['Cov%d' % x for x in range(len(cov.columns) - 2)]
        columns = dict(zip(cov.columns, ['fid', 'iid'] + covs_names))
        cov = cov.rename(columns=columns)
        prs_true.merge(cov, on=['fid', 'iid'])
        assert prs_true.shape[0] == dim1
        prs_true['PHENO'] = prs_true.loc[:, ['PHENO'] + covs_names].sum(axis=1)
    realized_h2 = prs_true.gen_eff.var() / prs_true.PHENO.var()
    with open('realized_h2.txt', 'w') as F:
        F.write('Estimated heritability (Va/Vp) : %.4f' % realized_h2)
    den = prs_true.gen_eff.var() + prs_true.env_eff.var()
    est_h2 = prs_true.gen_eff.var() / den
    prs_true.to_csv('%s.prs_pheno.gz' % prefix, sep='\t', compression='gzip', index=False)
    opts = dict(sep=' ', header=False, index=False)
    prs_true.reindex(columns=['fid', 'iid', 'PHENO']).to_csv('%s.pheno' % prefix, **opts)
    return prs_true, realized_h2

def qtraits_simulation(outprefix, bfile, h2, ncausal, snps=None, noenv=False,
                       causaleff=None, bfile2=None, flip=False, freq_thresh=0.01,
                       check=False, seed=None, uniform=False, normalize=False,
                       remove_causals=False, threads=1, max_memory=None, covs=None,
                       force_h2=False, **kwargs):
    available_memory = vm().available if max_memory is None else max_memory
    if causaleff is not None:
        if isinstance(causaleff, str):
            causaleff = pd.read_table('%s' % causaleff, delim_whitespace=True)
        causaleff = causaleff.reindex(columns=['snp', 'beta'])
        assert causaleff.shape[0] == ncausal
    picklefile = '%s.pickle' % outprefix
    if not os.path.isfile(picklefile):
        opts = dict(prefix=outprefix, bfile=bfile, h2=h2, ncausal=ncausal,
                    normalize=normalize, bfile2=bfile2, seed=seed, snps=snps,
                    causaleff=causaleff, uniform=uniform, f_thr=freq_thresh,
                    flip=flip, check=check, threads=threads, max_memory=available_memory)
        opts.update(kwargs)
        g, bim, truebeta, causals = true_prs(**opts)  # Get true PRS
        with open(picklefile, 'wb') as F: pickle.dump((g, bim, truebeta, causals), F)
    else:
        g, bim, truebeta, causals = pd.read_pickle(picklefile)
    if not os.path.isfile('%s.prs_pheno.gz' % outprefix):
        pheno, realized_h2 = create_pheno(outprefix, h2, truebeta, noenv=noenv,
                                          covs=covs, force_h2=force_h2)
    else:
        pheno = pd.read_table('%s.prs_pheno.gz' % outprefix, sep='\t')
        realized_h2 = float(open('realized_h2.txt').read().strip().split()[-1])
    if not os.path.isfile('%s.causaleff' % outprefix):
        causals = bim.dropna(subset=['beta'])
        causals.to_csv('%s.causaleff' % outprefix, index=False, sep='\t')
    if remove_causals:
        bim = bim[~bim.snp.isin(causals.snp)]
        g = g[:, bim.i.values]
        bim.loc[:, 'i'] = list(range(g.shape[1]))
        bim.reset_index(drop=True, inplace=True)
    return pheno, realized_h2, (g, bim, truebeta, causals)

qtraits_simulation(train, ${pops4}, ${h2}, 100, threads=${cpus}, freq_thresh=0,
                   force_h2=True, max_memory=${membytes} covs=${covs})
EOF
}

corr()
{
  awk 'pass==1 {sx+=$3; sy+=$6; n+=1} pass==2 {mx=sx/(n-1)
  my=sy/(n-1); cov+=($3-mx)*($6-my)
  ssdx+=($3-mx)*($3-mx); ssdy+=($6-my)*($6-my);} END {
  print (cov / ( sqrt(ssdx) * sqrt(ssdy)) )^2 }' pass=1 $1 pass=2 $1
}

outp()
{
  infn=$1
  n="${infn//[!0-9]/}"
  Pop=$2
  outfn=$3
  echo -e "$n\t`corr $infn`\t$Pop" >> $outfn
}

python_merge()
{
python - << EOF
import pandas as pd
df1 = pd.read_table('pcs.txt', sep='\t')
df2 = pd.read_table('Covs.txt', sep='\t')
merged = df1.merge(df2, on=['FID','IID'])
merged.to_csv('pcs.txt', sep='\t', index=False)
EOF
}

run_gwas(){
# 1) Plink executable with path
# 2) Covariate names
# 3) chromosome
# 4) prefix
$1 --bfile current_prop --linear hide-covar --pheno train.pheno --memory 7000 \
--covar pcs.txt --covar-name $2 --chr $3 --out $4_chr${3} --keep-allele-order \
--allow-no-sex
}

compute_duo()
{
  # 1 : prefix of output
  # 2 : fraction computed
  # 3 : Path to merged plink fileset
  # 4 : flags common to plink calls
  # 5 : vector with names of populations
  # 6 : keep file of prefix
  # 7 : Covariates
  pcs='PC1 PC2 PC3 PC4'
  prefix="${1}_${2}"
  if [[ ! -f ${prefix}.clumped ]]
  then
    echo -e "\nComputing summary statistics for ${prefix}\n"
    ${plink} --bfile $3 --keep $6 --make-bed --out current_prop $4
    flashpca --bfile current_prop -n ${cpus} -m ${mem} -d 4
    if echo $7| grep -q -- '--covs'; then
        python_merge
        pcs=`cut -d$'\t' -f3- pcs.txt|head -1`
    fi
    export -f run_gwas
    echo "Running GWAS in parallel in ${chrs} chromosomes"
    time parallel --will-cite -j 75% run_gwas ${plink} "${pcs}" {} \
    ${prefix} ::: `seq ${chrs}`
    cat ${prefix}_chr*.assoc.linear > ${prefix}.assoc.linear
    rm ${prefix}_chr*.assoc.linear
    # ${plink} --bfile current_prop --linear hide-covar --pheno train.pheno \
    # --covar pcs.txt --covar-name ${pcs} --out ${prefix} $4
    # --clump-r2 0.50              LD thqreshold for clumping is default
    echo "Running Scorings"
    time ${plink} --bfile current_prop --clump ${prefix}.assoc.linear \
     --clump-p1 0.01 --pheno train.pheno --out ${prefix} $4
  else
    echo -e "${prefix} has already been done"
  fi
  if [[ ! -f ${prefix}.myscore ]]; then
    grep -w "$(awk -F' ' '{if (NR!=1) { print $3 }}' ${prefix}.clumped)" \
    ${prefix}.assoc.linear > ${prefix}.myscore
  fi
  for pop in $5
  do
    if [[ ! -f ${pop}_${prefix}.profile ]]; then
      if [[ ! -f ${pop}_test.bed ]]; then
        cp ${pop}.bed ${pop}_test.bed
        cp ${pop}.bim ${pop}_test.bim
        cp ${pop}.fam ${pop}_test.fam
      fi
      $plink --bfile ${pop}_test --score ${prefix}.myscore 2 4 7 sum center \
      --pheno train.pheno --out ${pop}_${prefix} $4
      echo "Running correlation for pop ${pop}"
      time outp ${pop}_${prefix}.profile ${pop} ${1}.tsv
    fi
  done
}

gen_merge_list()
{
  # only one input, the string with the current populations
  for p in $1
    do
      echo "${genos}/${p}" >> merge.list
    done
}

gen_keeps_n_covs()
{
  count=0
  if [[ ! -f 'Covs.txt' ]];then
  for p in EUR ASN AFR AD;do
    cut -f1,2 ${genos}/${p}.fam| tee ${p}.keep | \
    sed "s/$/ ${count}/" >> Covs.txt
    count=$(( ${count} + 1 ))
  done
  fi
  if [[ "$covs" == TRUE ]]
  then
  covs='--covs Covs.txt'
  fi
}

merge_filesets(){
if [[ ! -f ${genos}/EURnASNnAFRnAD.bed ]]
    then
      echo -e "\n\nGenerating merged filesets"
      echo -e "${genos}/EUR\n${genos}/ASN\n${genos}/AFR\n${genos}/AD" > merge.list
      comm -12 <(comm -12 <(comm -12 <(sort ${genos}/EUR.bim) \
      <(sort ${genos}/ASN.bim)) <(sort ${genos}/AFR.bim)) \
      <(sort ${genos}/AD.bim) > merged.totalsnps
      ${plink} --merge-list merge.list --extract merged.totalsnps --make-bed \
      --out ${genos}/EURnASNnAFRnAD ${common_plink}
      cat ${genos}/EUR.fam ${genos}/${target}.fam > duo.keep
      ${plink} --bfile ${genos}/EURnASNnAFRnAD --keep duo.keep --make-bed \
      --out ${genos}/EURn${target} ${common_plink}
fi
}

generate_pheno(){
if [[ ! -f train.pheno ]]; then
    echo -e "\n\nGenerating phenotypes\n"
    export plink
    qtrait_simulation
    else
      echo -e "\n\nPhenotypes already present... moving on\n"
fi
}

get_initial(){
sort -R ${genos}/EUR.keep| head -n ${init} > initial.keep
if [ ! -f ${target}.test ]; then
    echo -e "\n\nGenerating keep files"
    comm -23 <(sort ${genos}/EUR.keep) <(sort initial.keep) > EUR.rest
    # split train/test in EUR
    sort -R EUR.rest| head -n ${sample} > EUR.train
    comm -23 <(sort EUR.rest) <(sort EUR.train) > EUR.test
    # split train/test in target
    sort -R ${genos}/${target}.keep| head -n ${sample} > ${target}.train
    comm -23 <(sort ${genos}/${target}.keep) <(sort ${target}.train) > ${target}.test
    else
      echo -e "\n\nTrain/test Keep files already present\n"
fi
}

gen_test(){
if [[ ! -f EUR_test.bed ]]; then
    echo -e "\n\nGenerating test filesets"
    # create the test filesets
    $plink --bfile ${genos}/${target} --keep ${target}.test --make-bed \
    --out ${target}_test ${common_plink}
    $plink --bfile ${genos}/EUR --keep EUR.test --make-bed --out EUR_test \
    ${common_plink}
fi
}

make_train_subset(){
if [[ ! -f train.txt ]]
    then
        cat ${target}.train EUR.train > train.keep
        cat train.keep initial.keep | sort | uniq > train.txt
fi
}

proportions(){
prop=NONE
const=NONE
sequence=`seq 0 ${step} ${sample}`
echo -e "\n\nStarting Original"
if [[ -f proportions.tsv ]]
    then
        pre=`cut -f1 proportions.tsv`
        sequ=`echo ${pre[@]} ${sequence[@]}| tr ' ' '\n'| sort| uniq -u`
    else
        sequ=${sequence}
fi

for i in ${sequ}
do
    eur=$(( sample - i ))
    echo -e "\n\nProcesing $eur european and $i $target"
    t=`bc <<< "(${eur} == 0)"`
    if [[ $eur = $sample ]]; then
      head -n ${eur} EUR.train > ${i}.keep
      #cat EUR.train > ${i}.keep
      #cp EUR.train constant_${i}.keep
    elif [ ${t} -ne 1 ]; then
      head -n ${eur} EUR.train > ${i}.keep
      head -n ${i} ${target}.train >> ${i}.keep
      cp EUR.train constant_${i}.keep
    else
      head -n ${i} ${target}.train >> ${i}.keep
    fi
    # Compute sumstats and clump for proportions
    echo "Running compute_duo proportions ${i} ${all} "${common_plink}" \
    "${target} ${others}" ${i}.keep "${covs}""
    time compute_duo proportions ${i} ${all} "${common_plink}" \
    "${target} ${others}" ${i}.keep "${covs}"
done
}

init(){
# constant initial source add mixing
if [[ -f init.tsv ]]
    then
        pre=`cut -f1 init.tsv`
        sequ=`echo ${pre[@]} ${sequence[@]}| tr ' ' '\n'| sort| uniq -u`
    else
        sequ=${sequence}
fi
echo -e "\n\nStarting constant initial source add mixing"
for i in ${sequ}
do
    eur=$(( sample - i ))
    echo -e "\n\nProcesing $eur european and $i $target with start of $init"
    cat initial.keep > init_${i}.keep
    if [[ ! $i = 0 ]]; then head -n $i ${target}.train >> init_${i}.keep; fi
    if [[ ! $eur = 0  ]]; then head -n $eur EUR.train >> init_${i}.keep; fi
   echo "compute_duo init ${i} ${all} "${common_plink}" "${target} ${others}" \
   init_${i}.keep "${covs}""
   time compute_duo init ${i} ${all} "${common_plink}" "${target} ${others}" \
   init_${i}.keep "${covs}"
done
}

cost(){
# do the cost derived
sequence=`seq 0 10`
if [[ -f cost.tsv ]]
    then
        pre=`cut -f1 cost.tsv`
        sequ=`echo ${pre[@]} ${sequence[@]}| tr ' ' '\n'| sort| uniq -u`
    else
        sequ=${sequence}
fi
echo -e "\n\nStarting cost"
for j in ${sequ}
do
    eu=`bc <<< "scale = 1; $j/10"`
    ad=`bc <<< "scale = 1; 1 - ($j/10)"`
    n=`bc <<< "scale = 1; $sample / (($ad * 2) + $eu)"`
    n=`bc <<< "$n/1"`
    eu=`bc <<< "($n * $eu)/1"`
    ad=`bc <<< "($n * $ad)/1"`
    echo -e "\n\nProcesing $eu european and $ad admixed"
    if [[ ! $ad = 0  ]]; then
        sort -R ${target}.train| head -n $ad > frac_${j}.keep
    fi
    if [[ ! $eu = 0  ]]; then
        sort -R EUR.train| head -n $eu >> frac_${j}.keep
    fi
    # Perform associations and clumping
    echo "compute_duo cost ${j} ${all} "${common_plink}" "${target} ${others}" \
    frac_${j}.keep "${covs}""
    time compute_duo cost ${j} ${all} "${common_plink}" "${target} ${others}" \
    frac_${j}.keep "${covs}"
done
}

execute(){
# Get the config file
#parse_config_file $1
source $1
cwd=$PWD
membytes=$(( mem * 1000000 ))
echo "Performing Rawlsian analysis of two Populations with target ${target}"
step=$(( sample/10 ))
others=`echo 'EUR ASN AFR AD' | sed -e "s/$target //"`
common_plink="--keep-allele-order --allow-no-sex --threads ${cpus} --memory ${mem}"
pops4=${genos}/EURnASNnAFRnAD
all=${genos}/EURn${target}
echo "Running gen_keeps_n_covs"
time gen_keeps_n_covs
echo "Running merge_filesets"
time merge_filesets
echo "Running generate_pheno"
time generate_pheno
echo "Running get_initial"
time get_initial
echo "Running gen_test"
time gen_test
echo "Running make_train_subset"
time make_train_subset
echo "Running proportions"
time proportions
echo "Running init"
time init
echo "Running cost"
cost
}
#--------------------------------------Execution-------------------------------------
TIMEFORMAT="Time elapsed in the full pipeline: %R"
export TIMEFORMAT
time execute $1
