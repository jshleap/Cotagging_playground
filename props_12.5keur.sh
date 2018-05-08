#!/bin/sh -xv
set -e
source ~/.bashrc
cwd=$PWD
cpus=16
mem=27000
membytes=2700000000
genos=`readlink -e $cwd/..`
# I will assume that this script is going to be launched in the genos folder.
# For now, in only works with EUR, AD and AFR as Source, Target and untyped
code=$1
plink=$2
init=12500
sample=$3
covs=$4
#TODO: make functions
if [ "$covs" == TRUE ]
  then python3 ${code}/skpca.py -b ${genos}/EURnAD -t ${cpus} -m ${mem} -c 2
  python3 -c "import pandas as pd; df=pd.read_table('EURnAD.pca', delim_whitespace=True, header=None).loc[:, [0,1,3]].to_csv('covariates.tsv', sep='\t')"
  cov='--covs EURnAD.pca'
fi
# generate the phenos
python3 ${code}/qtraitsimulation.py -p EUR -m 100 -b 0.8 -f 0 -B ${genos}/EUR -2 ${genos}/AD -t ${cpus}
python3 ${code}/qtraitsimulation.py -p AD -m 100 -b 0.8 -f 0 -B ${genos}/AD -2 ${genos}/EUR -t ${cpus} --causal_eff EUR.causaleff #--normalize $cov
python3 ${code}/qtraitsimulation.py -p AFR -m 100 -b 0.8 -f 0 -B ${genos}/AFR -2 ${genos}/EUR -t ${cpus} --causal_eff EUR.causaleff #--normalize $cov
cat EUR.pheno AD.pheno > train.pheno
# get the initial 12.5k
sort -R ${genos}/EUR.keep| head -n ${init} > initial.keep
comm -23 <(sort ${genos}/EUR.keep) <(sort initial.keep) > EUR.rest
# split train/test in EUR
sort -R EUR.rest| head -n ${sample} > EUR.train
comm -23 <(sort EUR.rest) <(sort EUR.train) > EUR.test
# split train/test in AD
sort -R ${genos}/AD.keep| head -n ${sample} > AD.train
comm -23 <(sort ${genos}/AD.keep) <(sort AD.train) > AD.test
$plink --bfile ${genos}/AD --keep AD.test --keep-allele-order --allow-no-sex --make-bed --out AD_test --threads ${cpus} --memory $mem
$plink --bfile ${genos}/EUR --keep EUR.test --keep-allele-order --allow-no-sex --make-bed --out EUR_test --threads ${cpus} --memory $mem
all=${genos}/EURnAD
#make train subset
cat AD.train EUR.train > train.keep
$plink --bfile ${all} --keep train.keep --keep-allele-order --allow-no-sex --make-bed --out train
# compute pca for this subset
python3 ${code}/skpca.py -b train -t ${cpus} -m ${mem} -c 1
#$plink --bfile train --keep ${i}.keep --keep-allele-order --allow-no-sex --pca 1 --out ${i} --threads ${cpus} --memory $mem

step=$(( sample/10 ))

# do Original
for i in `seq 0 $step $sample`
do
    eur=$(( sample - i ))
    if [[ $eur = $sample ]]
        then
            cat EUR.train > ${i}.keep
            cp EUR.train constant_${i}.keep
        else
            head -n $eur EUR.train> ${i}.keep
            head -n $i AD.train >> ${i}.keep
            cp EUR.train constant_${i}.keep
            head -n $i AD.train >> constant_${i}.keep
    fi
    # compute pca for this subset
    #$plink --bfile ${all} --keep ${i}.keep --keep-allele-order --allow-no-sex --pca 1 --out ${i} --threads ${cpus} --memory $mem
    # Compute sumstats and clump for proportions
    $plink --bfile ${all} --keep ${i}.keep --keep-allele-order --allow-no-sex --linear hide-covar --pheno train.pheno --covar train.pca --out ${i} --threads ${cpus} --memory $mem #${i}.eigenvec
    $plink --bfile ${all} --keep ${i}.keep --keep-allele-order --allow-no-sex --clump ${i}.assoc.linear --pheno train.pheno --out ${i}
    # Do the constant estimations
    $plink --bfile ${all} --keep constant_${i}.keep --keep-allele-order --allow-no-sex --linear hide-covar --pheno train.pheno --covar  ${i}.eigenvec --out constant_${i} --threads ${cpus} --memory $mem
    $plink --bfile ${all} --keep constant_${i}.keep --keep-allele-order --allow-no-sex --clump constant_${i}.assoc.linear --pheno train.pheno --out constant_${i}
    # Score original
    python3 ${code}/simple_score.py -b AD_test -c ${i}.clumped -s ${i}.assoc.linear -t ${cpus} -p train.pheno -l AD -m $membytes
    python3 ${code}/simple_score.py -b EUR_test -c ${i}.clumped -s ${i}.assoc.linear -t ${cpus} -p train.pheno -l EUR -m $membytes
    python3 ${code}/simple_score.py -b ${genos}/AFR -c ${i}.clumped -s ${i}.assoc.linear -t ${cpus} -p AFR.pheno -l AFR -m $membytes
    # Score constant
    python3 ${code}/simple_score.py -b AD_test -c constant_${i}.clumped -s constant_${i}.assoc.linear -t ${cpus} -p train.pheno -l AD -P constant -m $mem
    python3 ${code}/simple_score.py -b EUR_test -c constant_${i}.clumped -s constant_${i}.assoc.linear -t ${cpus} -p train.pheno -l EUR -P constant -m $mem
    python3 ${code}/simple_score.py -b ${genos}/AFR -c constant_${i}.clumped -s constant_${i}.assoc.linear -t ${cpus} -p AFR.pheno -l AFR -P constant -m $mem
done


# constant initial source add mixing
for i in `seq 0 $step $sample`
do
    cat initial.keep > ${i}.keep
    if [[ ! $i = 0 ]]; then head -n $i AD.train >> ${i}.keep; fi
    eur=$(( sample - i ))
    if [[ ! $eur = 0  ]]; then head -n $eur EUR.train >> ${i}.keep; fi
    # compute pca for this subset
    #$plink --bfile ${all} --keep ${i}.keep --keep-allele-order --allow-no-sex --pca 1 --out ${i} --threads ${cpus} --memory $mem
    # Perform associations and clumping
    $plink --bfile train --keep ${i}.keep --keep-allele-order --allow-no-sex --linear hide-covar --pheno train.pheno --covar train.pca --out ${i} --threads ${cpus} --memory $mem
    $plink --bfile train --keep ${i}.keep --keep-allele-order --allow-no-sex --clump ${i}.assoc.linear --pheno train.pheno --out ${i}
    # Score original
    python3 ${code}/simple_score.py -b AD_test -c ${i}.clumped -s ${i}.assoc.linear -t ${cpus} -p train.pheno -l AD -m $membytes -P init12k
    python3 ${code}/simple_score.py -b EUR_test -c ${i}.clumped -s ${i}.assoc.linear -t ${cpus} -p train.pheno -l EUR -m $membytes -P init12k
    python3 ${code}/simple_score.py -b ${genos}/AFR -c ${i}.clumped -s ${i}.assoc.linear -t ${cpus} -p AFR.pheno -l AFR -m $membytes -P init12k
done

# do the cost derived
for j in `seq 0 10`
do
    eu=`bc <<< "scale = 1; $j/10"`
    ad=`bc <<< "scale = 1; 1 - ($j/10)"`
    n=`bc <<< "scale = 1; $sample / (($ad * 2) + $eu)"`
    n=`bc <<< "$n/1"`
    eu=`bc <<< "($n * $eu)/1"`
    ad=`bc <<< "($n * $ad)/1"`
    if [[ ! $ad = 0  ]]; then
        sort -R AD.train| head -n $ad > frac_${j}.keep
    fi
    if [[ ! $eu = 0  ]]; then
        sort -R EUR.train| head -n $eu >> frac_${j}.keep
    fi
        # Perform associations and clumping
    $plink --bfile ${all} --keep frac_${j}.keep --keep-allele-order --allow-no-sex --linear hide-covar --pheno train.pheno --covar train.pca --out cost_${j} --threads ${cpus} --memory $mem
    $plink --bfile ${all} --keep frac_${j}.keep --keep-allele-order --allow-no-sex --clump cost_${j}.assoc.linear --pheno train.pheno --out cost_${j}
    # Score original
    python3 ${code}/simple_score.py -b AD_test -c cost_${j}.clumped -s cost_${j}.assoc.linear -t ${cpus} -p train.pheno -l AD -m $membytes -P cost
    python3 ${code}/simple_score.py -b EUR_test -c cost_${j}.clumped -s cost_${j}.assoc.linear -t ${cpus} -p train.pheno -l EUR -m $membytes -P cost
    python3 ${code}/simple_score.py -b ${genos}/AFR -c cost_${j}.clumped -s cost_${j}.assoc.linear -t ${cpus} -p AFR.pheno -l AFR -m $membytes -P cost
done