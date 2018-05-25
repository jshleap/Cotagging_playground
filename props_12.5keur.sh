#!/bin/sh -xv
set -e
source ~/.bashrc
cwd=$PWD
cpus=16
mem=37000
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
  echo -e "$n\t`corr $infn`\t$Pop" >> $outfn
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
    export plink
    python3 ${code}/qtraitsimulation.py -p EUR -m 100 -b 0.5 -f 0 -B ${genos}/EUR -2 ${genos}/${target} -t ${cpus} --force_h2 -M $membytes $covs
    python3 ${code}/qtraitsimulation.py -p ${target} -m 100 -b 0.5 -f 0 -B ${genos}/${target} -2 ${genos}/EUR -t ${cpus} --force_h2 --causal_eff EUR.causaleff -M $membytes $covs
    python3 ${code}/qtraitsimulation.py -p AFR -m 100 -b 0.5 -f 0 -B ${genos}/AFR -2 ${genos}/EUR -t ${cpus} --force_h2 --causal_eff EUR.causaleff -M $membytes
    python3 ${code}/qtraitsimulation.py -p AD -m 100 -b 0.5 -f 0 -B ${genos}/AD -2 ${genos}/EUR -t ${cpus} --force_h2 --causal_eff EUR.causaleff -M $membytes
    cat EUR.pheno ${target}.pheno > train.pheno
    else
      echo -e "\n\nPhenotypes already present... moving on\n"
fi

if [ ! -f ${target}.test ]; then
    echo -e "\n\nGenerating keep files"
    # get the initial 12.5k
    sort -R ${genos}/EUR.keep| head -n ${init} > initial.keep
    comm -23 <(sort ${genos}/EUR.keep) <(sort initial.keep) > EUR.rest
    # split train/test in EUR
    sort -R EUR.rest| head -n ${sample} > EUR.train
    comm -23 <(sort EUR.rest) <(sort EUR.train) > EUR.test
    # split train/test in AD
    sort -R ${genos}/${target}.keep| head -n ${sample} > ${target}.train
    comm -23 <(sort ${genos}/${target}.keep) <(sort ${target}.train) > ${target}.test
    else
      echo -e "\n\nKeep files already present\n"
fi

if [ ! -f EUR_test.bed ]; then
    echo -e "\n\nGenerating test filesets"
    # create the test filesets
    $plink --bfile ${genos}/${target} --keep ${target}.test --keep-allele-order --allow-no-sex --make-bed --out ${target}_test --threads ${cpus} --memory $mem
    $plink --bfile ${genos}/EUR --keep EUR.test --keep-allele-order --allow-no-sex --make-bed --out EUR_test --threads ${cpus} --memory $mem
fi

if [ ! -f ${genos}/EURn${target}.bed ]
    then
        echo -e "\n\nGenerating merged filesets"
        comm -12 <(sort EUR.totalsnps) <(sort ${target}.totalsnps) > merged.totalsnps
        $plink --bfile ${genos}/EUR --bmerge ${genos}/${target} --keep-allele-order --allow-no-sex --extract merged.totalsnps --make-bed --out ${genos}/EURn${target} --threads ${cpus} --memory $mem
fi
all=${genos}/EURn${target}
#if [ ! -f train.bed  ]; then
#    $plink --bfile ${all}  --keep train.keep --keep-allele-order --allow-no-sex --make-bed --out train --memory $mem
#fi

# compute pca for this subset
#if [ ! -f train.pca  ]; then
#    python3 ${code}/skpca.py -b ${all} -t ${cpus} -m ${mem} -c 4
#fi
#make train subset
if [ ! -f train.txt ]
    then
        cat ${target}.train EUR.train > train.keep
        cat train.keep initial.keep | sort | uniq > train.txt
fi

if [ ! -f train.eigenvec  ]
    then
        $plink --bfile ${all} --keep train.txt --keep-allele-order --allow-no-sex --pca 4 --out train --threads ${cpus} --memory $mem
fi

step=$(( sample/10 ))

# do Original
prop=NONE
const=NONE
sequence=`seq 0 $step $sample`
echo -e "\n\nStarting Original"
if [ -f constant.tsv ]
    then
        pre=`cut -f1 constant.tsv`
        sequ=`echo ${pre[@]} ${sequence[@]}| tr ' ' '\n'| sort| uniq -u`
        # const1=(`tail -n 1 constant.tsv`)
        # const=${const1[0]}
    else
        sequ=$sequence
fi
# if [ -f proportions.tsv ]
#    then
#        prop1=(`tail -n 1 proportions.tsv`)
#        prop=${prop1[0]}
# fi

for i in ${sequ}
do
    eur=$(( sample - i ))
    echo -e "\n\nProcesing $eur european and $i $target"
    if [[ $eur = $sample ]]
        then
            cat EUR.train > ${i}.keep
            cp EUR.train constant_${i}.keep
        else
            head -n $eur EUR.train > ${i}.keep
            head -n $i ${target}.train >> ${i}.keep
            cp EUR.train constant_${i}.keep
            head -n $i ${target}.train >> constant_${i}.keep
    fi
    # Compute sumstats and clump for proportions
    # ${i}.eigenvec
    if [ ! -f props_${i}.clumped ]
        then
            # $plink --bfile ${all} --keep ${i}.keep --keep-allele-order --allow-no-sex --pca 4 --out props_${i} --threads ${cpus} --memory $mem
            # python3 ${code}/skpca.py -b ${all} -t ${cpus} -m ${mem} -c 4 --keep ${i}.keep -p 3
            $plink --bfile ${all} --keep ${i}.keep --keep-allele-order --allow-no-sex --linear hide-covar --pheno train.pheno --covar train.eigenvec --out props_${i} --threads ${cpus} --memory $mem
            $plink --bfile ${all} --keep ${i}.keep --keep-allele-order --allow-no-sex --clump props_${i}.assoc.linear --clump-p1 0.01 --pheno train.pheno --out props_${i} --memory $mem
            # Do the constant estimations
            # $plink --bfile ${all} --keep ${i}.keep --keep-allele-order --allow-no-sex --pca 4 --out constant_${i} --threads ${cpus} --memory $mem
            $plink --bfile ${all} --keep constant_${i}.keep --keep-allele-order --allow-no-sex --linear hide-covar --pheno train.pheno --covar train.eigenvec --out constant_${i} --threads ${cpus} --memory $mem
            $plink --bfile ${all} --keep constant_${i}.keep --keep-allele-order --allow-no-sex --clump constant_${i}.assoc.linear --clump-p1 0.01 --pheno train.pheno --out constant_${i} --memory $mem
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
    $plink --bfile ${genos}/AD --score props_${i}.myscore 2 4 7 sum center --pheno AD.pheno --keep-allele-order --allow-no-sex --out AD${i}_props --threads ${cpus} --memory $mem
    outp AD_${i}_props.profile AD proportions.tsv
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
    $plink --bfile ${genos}/AD --score constant_${i}.myscore 2 4 7 sum center --pheno AD.pheno --keep-allele-order --allow-no-sex --out AD_${i}_constant --threads ${cpus} --memory $mem
    outp AD_${i}_constant.profile AD constant.tsv

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
        pre=`cut -f1 init12k.tsv`
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
    # compute pca for this subset
    #$plink --bfile ${all} --keep init_${i}.keep --keep-allele-order --allow-no-sex --pca 4 --out init_${i} --threads ${cpus} --memory $mem
    # Perform associations and clumping
    $plink --bfile ${all} --keep init_${i}.keep --keep-allele-order --allow-no-sex --linear hide-covar --pheno train.pheno --covar train.eigenvec --out init_${i} --threads ${cpus} --memory $mem
    $plink --bfile ${all} --keep init_${i}.keep --keep-allele-order --allow-no-sex --clump init_${i}.assoc.linear --clump-p1 0.01 --pheno train.pheno --out init_${i} --memory $mem
    grep -w "$(awk -F' ' '{if (NR!=1) { print $3 }}' init_${i}.clumped)" init_${i}.assoc.linear > init_${i}.myscore
    # Score original
    $plink --bfile ${target}_test --score init_${i}.myscore 2 4 7 sum center --pheno train.pheno --keep-allele-order --allow-no-sex --out ${target}_init_${i} --threads ${cpus} --memory $mem
    outp ${target}_init_${i}.profile ${target} init12k.tsv
    $plink --bfile EUR_test --score init_${i}.myscore 2 4 7 sum center --pheno train.pheno --keep-allele-order --allow-no-sex --out EUR_init_${i} --threads ${cpus} --memory $mem
    outp EUR_init_${i}.profile EUR init12k.tsv
    $plink --bfile ${genos}/AFR --score init_${i}.myscore 2 4 7 sum center --pheno AFR.pheno --keep-allele-order --allow-no-sex --out AFR_init_${i} --threads ${cpus} --memory $mem
    outp AFR_init_${i}.profile AFR init12k.tsv
    $plink --bfile ${genos}/AD --score init_${i}.myscore 2 4 7 sum center --pheno AD.pheno --keep-allele-order --allow-no-sex --out AD_init_${i} --threads ${cpus} --memory $mem
    outp AD_init_${i}.profile AD init12k.tsv

    # python3 ${code}/simple_score.py -b ${target}_test -c ${i}.clumped -s ${i}.assoc.linear -t ${cpus} -p train.pheno -l ${target} -m $membytes -P init12k
    # python3 ${code}/simple_score.py -b EUR_test -c ${i}.clumped -s ${i}.assoc.linear -t ${cpus} -p train.pheno -l EUR -m $membytes -P init12k
    # python3 ${code}/simple_score.py -b ${genos}/AFR -c ${i}.clumped -s ${i}.assoc.linear -t ${cpus} -p AFR.pheno -l AFR -m $membytes -P init12k
done

# do the cost derived
sequence=`seq 0 10`
if [ -f cost.tsv ]
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
    # $plink --bfile ${all} --keep frac_${j}.keep --keep-allele-order --allow-no-sex --pca 4 --out frac_${j} --threads ${cpus} --memory $mem
    $plink --bfile ${all} --keep frac_${j}.keep --keep-allele-order --allow-no-sex --linear hide-covar --pheno train.pheno --covar train.eigenvec --out cost_${j} --threads ${cpus} --memory $mem
    $plink --bfile ${all} --keep frac_${j}.keep --keep-allele-order --allow-no-sex --clump cost_${j}.assoc.linear --clump-p1 0.01 --pheno train.pheno --out cost_${j} --memory $mem
    # Score cost
    grep -w "$(awk -F' ' '{if (NR!=1) { print $3 }}' cost_${j}.clumped)" cost_${j}.assoc.linear > cost_${j}.myscore
    $plink --bfile ${target}_test --score cost_${j}.myscore 2 4 7 sum center --pheno train.pheno --keep-allele-order --allow-no-sex --out ${target}_${j}_cost --threads ${cpus} --memory $mem
    outp ${target}_${j}_cost.profile ${target} cost.tsv
    $plink --bfile EUR_test --score cost_${j}.myscore 2 4 7 sum center --pheno train.pheno --keep-allele-order --allow-no-sex --out EUR_${j}_cost --threads ${cpus} --memory $mem
    outp EUR_${j}_cost.profile EUR cost.tsv
    $plink --bfile ${genos}/AFR --score cost_${j}.myscore 2 4 7 sum center --pheno AFR.pheno --keep-allele-order --allow-no-sex --out AFR_${j}_cost --threads ${cpus} --memory $mem
    outp AFR_${j}_cost.profile AFR cost.tsv
    $plink --bfile ${genos}/AD --score cost_${j}.myscore 2 4 7 sum center --pheno AD.pheno --keep-allele-order --allow-no-sex --out AD_${j}_cost --threads ${cpus} --memory $mem
    outp AD_${j}_cost.profile AD cost.tsv
done
