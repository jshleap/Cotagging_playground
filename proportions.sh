set -e
source ~/.bashrc
cwd=$PWD
cpus=16
mem=37000
membytes=3000000000
genos=`readlink -e $cwd/..`
# I will assume that this script is going to be launched in the genos folder.
# For now, in only works with EUR, AD and AFR as Source, Target and untyped
code=$1
plink=$2
sample=$3
covs=$4
if [ "$covs" == TRUE ]
  then python3 ${code}/skpca.py -b ${genos}/EURnAD -t ${cpus} -m ${mem} -c 2
  python3 -c "import pandas as pd; df=pd.read_table('EURnAD.pca', delim_whitespace=True, header=None).loc[:, [0,1,3]].to_csv('covariates.tsv', sep='\t')"
  cov='--covs EURnAD.pca'
fi
sort -R ${genos}/EUR.keep| head -n ${sample} > EUR.train
comm -23 <(sort ${genos}/EUR.keep) <(sort EUR.train) > EUR.test
sort -R ${genos}/AD.keep| head -n ${sample} > AD.train
comm -23 <(sort ${genos}/AD.keep) <(sort AD.train) > AD.test

python3 ${code}/qtraitsimulation.py -p EUR -m 100 -b 0.8 -f 0 -B ${genos}/EUR -2 ${genos}/AD -t ${cpus}
python3 ${code}/qtraitsimulation.py -p AD -m 100 -b 0.8 -f 0 -B ${genos}/AD -2 ${genos}/EUR -t ${cpus} --causal_eff EUR.causaleff #--normalize $cov
python3 ${code}/qtraitsimulation.py -p AFR -m 100 -b 0.8 -f 0 -B ${genos}/AFR -2 ${genos}/EUR -t ${cpus} --causal_eff EUR.causaleff #--normalize $cov
cat EUR.pheno AD.pheno > train.pheno
all=${genos}/EURnAD
step=$(( sample/10 ))
#if [ ! -f ${all}.pca ]; then
#    python3 ${code}/skpca.py -b $all -t ${cpus} -m ${mem} -c 1
#fi

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
    $plink --bfile ${all} --keep ${i}.keep --keep-allele-order --allow-no-sex --pca 1 --out ${i} --threads ${cpus} --memory $mem
    # Compute sumstats and clump for proportions
    $plink --bfile ${all} --keep ${i}.keep --keep-allele-order --allow-no-sex --linear hide-covar --pheno train.pheno --covar ${i}.eigenvec --out ${i} --threads ${cpus} --memory $mem
    $plink --bfile ${all} --keep ${i}.keep --keep-allele-order --allow-no-sex --clump ${i}.assoc.linear --pheno train.pheno --out ${i}
    # Do the constant estimations
    $plink --bfile ${all} --keep constant_${i}.keep --keep-allele-order --allow-no-sex --linear hide-covar --pheno train.pheno --covar  ${i}.eigenvec --out constant_${i} --threads ${cpus} --memory $mem
    $plink --bfile ${all} --keep constant_${i}.keep --keep-allele-order --allow-no-sex --clump constant_${i}.assoc.linear --pheno train.pheno --out constant_${i}
    # Score original
    python3 ${code}/simple_score.py -b ${genos}/AD_test -c ${i}.clumped -s ${i}.assoc.linear -t ${cpus} -p train.pheno -l AD -m $membytes
    python3 ${code}/simple_score.py -b ${genos}/EUR_test -c ${i}.clumped -s ${i}.assoc.linear -t ${cpus} -p train.pheno -l EUR -m $membytes
    python3 ${code}/simple_score.py -b ${genos}/AFR -c ${i}.clumped -s ${i}.assoc.linear -t ${cpus} -p AFR.pheno -l AFR -m $membytes
    # Score constant
    python3 ${code}/simple_score.py -b ${genos}/AD_test -c constant_${i}.clumped -s constant_${i}.assoc.linear -t ${cpus} -p train.pheno -l AD -P constant -m $mem
    python3 ${code}/simple_score.py -b ${genos}/EUR_test -c constant_${i}.clumped -s constant_${i}.assoc.linear -t ${cpus} -p train.pheno -l EUR -P constant -m $mem
    python3 ${code}/simple_score.py -b ${genos}/AFR -c constant_${i}.clumped -s constant_${i}.assoc.linear -t ${cpus} -p AFR.pheno -l AFR -P constant -m $mem
done
