#!/bin/sh -xv
set -e
source ~/.bashrc
cwd=$PWD
cpus=16
mem=27000
membytes=$(( mem * 1000000 ))
genos=$1
# I will assume that this script is going to be launched in the genos folder.
# For now, in only works with EUR, AD and AFR as Source, Target and untyped
code=$2
plink=$3
init=12500
sample=$4
target=$5
covs=$6
#TODO: make functions
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
  echo -e "$n\t`corr $infn`\t$Pop\t$run" >> $outfn
}

if [ "$covs" == TRUE ]
  then
    cut -d' ' -f1,2 ${genos}/${target}.fam|sed 's/$/ 1/' > Covs.txt
    cut -d' ' -f1,2 ${genos}/EUR.fam|sed 's/$/ 0/' >> Covs.txt
    covs='--covs Covs.txt'
#    python3 ${code}/skpca.py -b ${genos}/EURnAD -t ${cpus} -m ${mem} -c 2
#    python3 -c "import pandas as pd; df=pd.read_table('EURnAD.pca', delim_whitespace=True, header=None).loc[:, [0,1,3]].to_csv('covariates.tsv', sep='\t')"
#    cov='--covs EURnAD.pca'
fi

# generate the phenos
if [ ! -f train.pheno ]; then
    echo -e "\n\nGenerating phenotypes\n"
    python3 ${code}/qtraitsimulation.py -p EUR -m 100 -b 0.8 -f 0 -B ${genos}/EUR -2 ${genos}/${target} -t ${cpus} -M $membytes $covs
    python3 ${code}/qtraitsimulation.py -p ${target} -m 100 -b 0.8 -f 0 -B ${genos}/${target} -2 ${genos}/EUR -t ${cpus} --causal_eff EUR.causaleff -M $membytes $covs
    python3 ${code}/qtraitsimulation.py -p AFR -m 100 -b 0.8 -f 0 -B ${genos}/AFR -2 ${genos}/EUR -t ${cpus} --causal_eff EUR.causaleff -M $membytes
    cat EUR.pheno ${target}.pheno > train.pheno
fi

if [ ! -f ${target}.test ]; then
    echo -e "\nGenerating keep files"
    # get the initial 12.5k
    sort -R ${genos}/EUR.keep| head -n ${init} > initial.keep
    comm -23 <(sort ${genos}/EUR.keep) <(sort initial.keep) > EUR.rest
    # split train/test in EUR
    sort -R EUR.rest| head -n ${sample} > EUR.train
    comm -23 <(sort EUR.rest) <(sort EUR.train) > EUR.test
    # split train/test in AD
    sort -R ${genos}/${target}.keep| head -n ${sample} > ${target}.train
    comm -23 <(sort ${genos}/${target}.keep) <(sort ${target}.train) > ${target}.test
fi

if [ ! -f EUR_test.bed ]; then
    echo -e "\n\nGenerating test filesets"
    # create the test filesets
    $plink --bfile ${genos}/${target} --keep ${target}.test --keep-allele-order --allow-no-sex --make-bed --out ${target}_test --threads ${cpus} --memory $mem
    $plink --bfile ${genos}/EUR --keep EUR.test --keep-allele-order --allow-no-sex --make-bed --out EUR_test --threads ${cpus} --memory $mem
fi

if [ ! -f ${genos}/EURn${target} ]
    then
        $plink --bfile ${genos}/EUR --bmerge ${genos}/${target} --keep-allele-order --allow-no-sex --make-bed --out ${genos}/EURn${target} --threads ${cpus} --memory $mem
fi
all=${genos}/EURn${target}
#make train subset
cat ${target}.train EUR.train > train.keep

if [ ! -f train.bed  ]; then
$plink --bfile ${all} --keep train.keep --keep-allele-order --allow-no-sex --make-bed --out train --memory $mem
fi

# compute pca for this subset
if [ ! -f train.pca  ]; then
    python3 ${code}/skpca.py -b train -t ${cpus} -m ${mem} -c 4
fi
#$plink --bfile train --keep ${i}.keep --keep-allele-order --allow-no-sex --pca 1 --out ${i} --threads ${cpus} --memory $mem

step=$(( sample/10 ))

# do Original
prop=NONE
const=NONE

echo -e "\n\nStarting Original"
if [ -f constant.tsv ]
    then
        const1=(`tail -n 1 constant.tsv`)
        const=${const1[0]}
fi
if [ -f proportions.tsv ]
    then
        prop1=(`tail -n 1 proportions.tsv`)
        prop=${prop1[0]}
fi
for i in `seq 0 $step $sample`
do
    eur=$(( sample - i ))
    echo -e "\n\nProcesing $eur european and $i admixed"
    if [[ $eur = $sample ]]
        then
            cat EUR.train > ${i}.keep
            cp EUR.train constant_${i}.keep
        else
            head -n $eur EUR.train> ${i}.keep
            head -n $i ${target}.train >> ${i}.keep
            cp EUR.train constant_${i}.keep
            head -n $i ${target}.train >> constant_${i}.keep
    fi
    # compute pca for this subset
    #$plink --bfile ${all} --keep ${i}.keep --keep-allele-order --allow-no-sex --pca 1 --out ${i} --threads ${cpus} --memory $mem
    # Compute sumstats and clump for proportions
    #${i}.eigenvec
    if [ ! -f props_${i}.clumped ]
        then
            $plink --bfile ${all} --keep ${i}.keep --keep-allele-order --allow-no-sex --linear hide-covar --pheno train.pheno --covar train.pca --out props_${i} --threads ${cpus} --memory $mem
            $plink --bfile ${all} --keep ${i}.keep --keep-allele-order --allow-no-sex --clump props_${i}.assoc.linear --pheno train.pheno --out props_${i} --memory $mem
            # Do the constant estimations
            $plink --bfile ${all} --keep constant_${i}.keep --keep-allele-order --allow-no-sex --linear hide-covar --pheno train.pheno --covar  train.pca --out constant_${i} --threads ${cpus} --memory $mem
            $plink --bfile ${all} --keep constant_${i}.keep --keep-allele-order --allow-no-sex --clump constant_${i}.assoc.linear --pheno train.pheno --out constant_${i} --memory $mem
    fi
    # Score original
#    if [[ $prop == None || $prop -lt $i ]]
#        then
    grep -w "$(awk -F' ' '{if (NR!=1) { print $3 }}' props_${i}.clumped)" props_${i}.assoc.linear > props_${i}.myscore
    $plink --bfile ${target}_test --score props_${i}.myscore 2 4 7 sum center --pheno train.pheno --keep-allele-order --allow-no-sex --out ${target}_${i}_props --threads ${cpus} --memory $mem
    outp ${target}_${i}_props.profile ${target} proportions.tsv
    $plink --bfile EUR_test --score props_${i}.myscore 2 4 7 sum center --pheno train.pheno --keep-allele-order --allow-no-sex --out EUR_${i}_props --threads ${cpus} --memory $mem
    outp EUR_${i}_props.profile EUR proportions.tsv
    $plink --bfile ${genos}/AFR --score props_${i}.myscore 2 4 7 sum center --pheno AFR.pheno --keep-allele-order --allow-no-sex --out AFR_${i}_props --threads ${cpus} --memory $mem
    outp AFR_${i}_props.profile AFR proportions.tsv
#            python3 ${code}/simple_score.py -b ${target}_test -c props_${i}.clumped -s props_${i}.assoc.linear -t ${cpus} -p train.pheno -l ${target} -m $membytes
#            python3 ${code}/simple_score.py -b EUR_test -c props_${i}.clumped -s props_${i}.assoc.linear -t ${cpus} -p train.pheno -l EUR -m $membytes
#            python3 ${code}/simple_score.py -b ${genos}/AFR -c props_${i}.clumped -s props_${i}.assoc.linear -t ${cpus} -p AFR.pheno -l AFR -m $membytes
##        else
##            echo "Original $i done with line $prop1"
#    fi
    # Score constant
#    if [[ $const == None || $const -lt $i ]]
#        then
    grep -w "$(awk -F' ' '{if (NR!=1) { print $3 }}' ${i}.clumped)" constant_${i}.assoc.linear > constant_${i}.myscore
    $plink --bfile ${target}_test --score constant_${i}.myscore 2 4 7 sum center --pheno train.pheno --keep-allele-order --allow-no-sex --out ${target}_${i}_constant --threads ${cpus} --memory $mem
    outp ${target}_${i}_constant.profile ${target} constant.tsv
    $plink --bfile EUR_test --score constant_${i}.myscore 2 4 7 sum center --pheno train.pheno --keep-allele-order --allow-no-sex --out EUR_${i}_constant --threads ${cpus} --memory $mem
    outp EUR_${i}_constant.profile EUR constant.tsv
    $plink --bfile ${genos}/AFR --score constant_${i}.myscore 2 4 7 sum center --pheno AFR.pheno --keep-allele-order --allow-no-sex --out AFR_${i}_constant --threads ${cpus} --memory $mem
    outp AFR_${i}_constant.profile AFR constant.tsv

#            python3 ${code}/simple_score.py -b ${target}_test -c constant_${i}.clumped -s constant_${i}.assoc.linear -t ${cpus} -p train.pheno -l ${target} -P constant -m $membytes
#            python3 ${code}/simple_score.py -b EUR_test -c constant_${i}.clumped -s constant_${i}.assoc.linear -t ${cpus} -p train.pheno -l EUR -P constant -m $membytes
#            python3 ${code}/simple_score.py -b ${genos}/AFR -c constant_${i}.clumped -s constant_${i}.assoc.linear -t ${cpus} -p AFR.pheno -l AFR -P constant -m $membytes
#        else
#            echo "Constant $i done with line $const1"
#    fi
done


# constant initial source add mixing
if [ -f init12k.tsv ]
    then
        lines=`wc -l init12k.tsv`
        echo -e "\n\ninit12k has been done, wc is $lines"
    else
        echo -e "\n\nStarting constant initial source add mixing"
        for i in `seq 0 $step $sample`
        do
            cat initial.keep > init_${i}.keep
            if [[ ! $i = 0 ]]; then head -n $i ${target}.train >> init_${i}.keep; fi
            eur=$(( sample - i ))
            echo -e "\n\nProcesing $eur european and $i admixed with start of $init"
            if [[ ! $eur = 0  ]]; then head -n $eur EUR.train >> init_${i}.keep; fi
            # compute pca for this subset
            #$plink --bfile ${all} --keep ${i}.keep --keep-allele-order --allow-no-sex --pca 1 --out ${i} --threads ${cpus} --memory $mem
            # Perform associations and clumping
            $plink --bfile train --keep init_${i}.keep --keep-allele-order --allow-no-sex --linear hide-covar --pheno train.pheno --covar train.pca --out init_${i} --threads ${cpus} --memory $mem
            $plink --bfile train --keep init_${i}.keep --keep-allele-order --allow-no-sex --clump init_${i}.assoc.linear --pheno train.pheno --out init_${i} --memory $mem
            grep -w "$(awk -F' ' '{if (NR!=1) { print $3 }}' init_${i}.clumped)" init_${i}.assoc.linear > init_${i}.myscore
            # Score original
            $plink --bfile ${target}_test --score init_${i}.myscore 2 4 7 sum center --pheno train.pheno --keep-allele-order --allow-no-sex --out ${target}_init_${i} --threads ${cpus} --memory $mem
            outp ${target}_init_${i}.profile ${target} init12k.tsv
            $plink --bfile EUR_test --score init_${i}.myscore 2 4 7 sum center --pheno train.pheno --keep-allele-order --allow-no-sex --out EUR_init_${i} --threads ${cpus} --memory $mem
            outp EUR_init_${i}.profile EUR init12k.tsv
            $plink --bfile ${genos}/AFR --score init_${i}.myscore 2 4 7 sum center --pheno AFR.pheno --keep-allele-order --allow-no-sex --out AFR_init_${i} --threads ${cpus} --memory $mem
            outp AFR_init_${i}.profile AFR init12k.tsv

            # python3 ${code}/simple_score.py -b ${target}_test -c ${i}.clumped -s ${i}.assoc.linear -t ${cpus} -p train.pheno -l ${target} -m $membytes -P init12k
            # python3 ${code}/simple_score.py -b EUR_test -c ${i}.clumped -s ${i}.assoc.linear -t ${cpus} -p train.pheno -l EUR -m $membytes -P init12k
            # python3 ${code}/simple_score.py -b ${genos}/AFR -c ${i}.clumped -s ${i}.assoc.linear -t ${cpus} -p AFR.pheno -l AFR -m $membytes -P init12k
        done
fi
# do the cost derived
if [ -f cost.tsv ]
    then
        lines2=`wc -l cost.tsv`
        echo -e "\n\nCost has been done, wc is $lines2"
    else
        echo -e "\n\nStarting cost"
        for j in `seq 0 10`
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
            $plink --bfile ${all} --keep frac_${j}.keep --keep-allele-order --allow-no-sex --linear hide-covar --pheno train.pheno --covar train.pca --out cost_${j} --threads ${cpus} --memory $mem
            $plink --bfile ${all} --keep frac_${j}.keep --keep-allele-order --allow-no-sex --clump cost_${j}.assoc.linear --pheno train.pheno --out cost_${j} --memory $mem
            # Score cost
            grep -w "$(awk -F' ' '{if (NR!=1) { print $3 }}' cost_${j}.clumped)" cost_${j}.assoc.linear > cost_${j}.myscore
            $plink --bfile ${target}_test --score cost_${j}.myscore 2 4 7 sum center --pheno train.pheno --keep-allele-order --allow-no-sex --out ${target}_${j}_cost --threads ${cpus} --memory $mem
            outp ${target}_${j}_cost.profile ${target} cost.tsv
            $plink --bfile EUR_test --score cost_${j}.myscore 2 4 7 sum center --pheno train.pheno --keep-allele-order --allow-no-sex --out EUR_${j}_cost --threads ${cpus} --memory $mem
            outp EUR_${j}_cost.profile EUR cost.tsv
            $plink --bfile ${genos}/AFR --score cost_${j}.myscore 2 4 7 sum center --pheno AFR.pheno --keep-allele-order --allow-no-sex --out AFR_${j}_cost --threads ${cpus} --memory $mem
            outp AFR_${j}_cost.profile AFR cost.tsv
#            python3 ${code}/simple_score.py -b ${target}_test -c cost_${j}.clumped -s cost_${j}.assoc.linear -t ${cpus} -p train.pheno -l ${target} -m $membytes -P cost
#            python3 ${code}/simple_score.py -b EUR_test -c cost_${j}.clumped -s cost_${j}.assoc.linear -t ${cpus} -p train.pheno -l EUR -m $membytes -P cost
#            python3 ${code}/simple_score.py -b ${genos}/AFR -c cost_${j}.clumped -s cost_${j}.assoc.linear -t ${cpus} -p AFR.pheno -l AFR -m $membytes -P cost
        done
fi