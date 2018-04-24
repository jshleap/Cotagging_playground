#!/usr/bin/env bash

# Run simulation and score fractions
set -e
source ~/.bashrc
main_source=$1  # main source genotypes path
main_target=$2	# main target genotypes path
genos=$3 	# path 2 train genotypes path
code=$4 	# path to codes	
plink=$5	# path to plink exe
cpus=$6 	# number of cpus to use
mem=$7 		# max memory to use
sample=$8 	# sample size to keep
 
#pt=$7
#lt=$8

# for clumping clump-p1 default is 0.0001, clump-p2 0.01, clump-r2 0.5

python3 ${code}/qtraitsimulation.py -p EUR -m 100 -b 0.8 -f 0 -B ${main_source} -2 ${main_target} -t ${cpus} --normalize
python3 ${code}/qtraitsimulation.py -p AD -m 100 -b 0.8 -f 0 -B ${main_target} -2 ${main_source} -t ${cpus} --causal_eff EUR.causaleff --normalize
cat EUR.pheno AD.pheno > train.pheno

step=$(( sample/10 ))
for i in `seq 0 $step $sample`
do 
    if [[ ! $i = 0 ]]; then head -n $i ${genos}/AD.train > ${i}.keep;fi
    eur=$(( sample - i ))
    if [[ ! $eur = 0  ]]; then head -n $eur ${genos}/EUR.train >> ${i}.keep;fi
    $plink --bfile ${genos}/EURnAD --pheno train.pheno --keep ${i}.keep --keep-allele-order --allow-no-sex --make-bed --out ${i} --threads ${cpus} --memory $(( mem/1000000 ))
    #smartpca.perl -i ${i}.bed -a ${i}.bim -b ${i}.fam -k 1 -o ${i}.pca -p ${i}.plot -e ${i}.eval -l ${i}.log -m 0 -q YES
    python3 ${code}/skpca.py -b ${i} -t ${cpus} -m ${mem} -c 1
    #awk '{$1=$1};1' ${i}.pca.evec| tr '  :' '\t'| cut -d$'\t' -f1,2,3| tr '\t' ' '|sed '1d' > ${i}.eigvec
    $plink --bfile ${i} --keep-allele-order --allow-no-sex --linear hide-covar standard-beta --covar ${i}.pca --out ${i} --threads ${cpus} --memory $(( mem/1000000 ))
    $plink --bfile ${i} --keep-allele-order --allow-no-sex --clump ${i}.assoc.linear --out ${i}
    python3 ${code}/simple_score.py -b ${genos}/AD_test -c ${i}.clumped -s ${i}.assoc.linear -t ${cpus} -p train.pheno -l AD

done