#!/usr/bin/env bash

# Run simulation and score fractions
set -e
source ~/.bashrc
main_source=$1  # main source genotypes path
main_target=$2	# main target genotypes path
main_target2=$9	# main second (untyped) target genotypes path
genos=$3 	# path 2 train genotypes path
code=$4 	# path to codes
plink=$5	# path to plink exe
cpus=$6 	# number of cpus to use
mem=$7 		# max memory to use
sample=$8 	# sample size to keep
covs=${10}  # include covariates
#pt=$7
#lt=$8

# for clumping clump-p1 default is 0.0001, clump-p2 0.01, clump-r2 0.5
if [ "$covs" == TRUE ]
  then python3 ${code}/skpca.py -b ${genos}/EURnAD -t ${cpus} -m ${mem} -c 2
  python3 -c "import pandas as pd; df=pd.read_table('EURnAD.pca', delim_whitespace=True, header=None).loc[:, [0,1,3]].to_csv('covariates.tsv', sep='\t')"
  cov='--covs EURnAD.pca'
fi
python3 ${code}/qtraitsimulation.py -p EUR -m 100 -b 0.8 -f 0 -B ${main_source} -2 ${main_target} -t ${cpus} #--normalize $cov
python3 ${code}/qtraitsimulation.py -p AD -m 100 -b 0.8 -f 0 -B ${main_target} -2 ${main_source} -t ${cpus} --causal_eff EUR.causaleff #--normalize $cov
python3 ${code}/qtraitsimulation.py -p AFR -m 100 -b 0.8 -f 0 -B ${main_target2} -2 ${main_source} -t ${cpus} --causal_eff EUR.causaleff #--normalize $cov
cat EUR.pheno AD.pheno > train.pheno
all=${genos}/EURnAD
step=$(( sample/10 ))
cat ${genos}/EUR.train > constant.keep
python3 ${code}/skpca.py -b $all -t ${cpus} -m ${mem} -c 1

for i in `seq 0 $step $sample`
do 
    if [[ ! $i = 0 ]]; then head -n $i ${genos}/AD.train > ${i}.keep
    head -n $i ${genos}/AD.train >> constant.keep
    fi
    eur=$(( sample - i ))
    if [[ ! $eur = 0  ]]; then head -n $eur ${genos}/EUR.train >> ${i}.keep; fi
    #$plink --bfile ${genos}/EURnAD --pheno train.pheno --keep ${i}.keep --keep-allele-order --allow-no-sex --make-bed --out ${i} --threads ${cpus} --memory $(( mem/1000000 ))
    #smartpca.perl -i ${i}.bed -a ${i}.bim -b ${i}.fam -k 1 -o ${i}.pca -p ${i}.plot -e ${i}.eval -l ${i}.log -m 0 -q YES
    #awk '{$1=$1};1' ${i}.pca.evec| tr '  :' '\t'| cut -d$'\t' -f1,2,3| tr '\t' ' '|sed '1d' > ${i}.eigvec
    $plink --bfile ${all} --keep ${i}.keep --keep-allele-order --allow-no-sex --linear hide-covar --covar ${all}.pca --out ${i} --threads ${cpus} --memory $(( mem/1000000 ))
    $plink --bfile ${all} --keep ${i}.keep --keep-allele-order --allow-no-sex --clump ${i}.assoc.linear --out ${i}
    # Do the constant estimations
    $plink --bfile ${all} --keep constant.keep --keep-allele-order --allow-no-sex --linear hide-covar --pheno train.pheno --covar ${all}.pca --out constant_${i} --threads ${cpus} --memory $(( mem/1000000 ))
    $plink --bfile ${all} --keep constant.keep --keep-allele-order --allow-no-sex --clump ${i}.assoc.linear --out constant_${i}
    # Score original
    python3 ${code}/simple_score.py -b ${genos}/AD_test -c ${i}.clumped -s ${i}.assoc.linear -t ${cpus} -p train.pheno -l AD #-N
    python3 ${code}/simple_score.py -b ${genos}/EUR_test -c ${i}.clumped -s ${i}.assoc.linear -t ${cpus} -p train.pheno -l EUR #-N
    python3 ${code}/simple_score.py -b ${main_target2} -c ${i}.clumped -s ${i}.assoc.linear -t ${cpus} -p AFR.pheno -l AFR #-N
    # Score constant
    python3 ${code}/simple_score.py -b ${genos}/AD_test -c ${i}.clumped -s constant_${i}.assoc.linear -t ${cpus} -p train.pheno -l AD -P constant
    python3 ${code}/simple_score.py -b ${genos}/EUR_test -c ${i}.clumped -s constant_${i}.assoc.linear -t ${cpus} -p train.pheno -l EUR -P constant
    python3 ${code}/simple_score.py -b ${main_target2} -c ${i}.clumped -s constant_${i}.assoc.linear -t ${cpus} -p AFR.pheno -l AFR -P constant
done