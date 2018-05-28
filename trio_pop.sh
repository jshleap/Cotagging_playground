#!/usr/bin/env bash

#!/bin/sh -xv
set -e
source ~/.bashrc
cwd=$PWD
cpus=16
mem=37000
membytes=$(( mem * 1000000 ))
genos=$1
code=$2
plink=$3
init=12500
sample=$4
pop1=$5
pop2=$6
pop3=$7
covs=$8

corr()
{
  awk 'pass==1 {sx+=$3; sy+=$6; n+=1} pass==2 {mx=sx/(n-1)
  my=sy/(n-1); cov+=($3-mx)*($6-my)
  ssdx+=($3-mx)*($3-mx); ssdy+=($6-my)*($6-my);} END {
  print (cov / ( sqrt(ssdx) * sqrt(ssdy)) )^2 }' pass=1 $1 pass=2 $1
}
outp()
{
  echo -e "$3\t`corr $1`\t$2" >> $4
}

perfrac()
{
  echo -e "${pop1}\t${pop2}\t${pop3}\t`corr $1`\t`corr $2`\t`corr $3`" >> $4
}

do_covs()
{
  echo 'Computing covariates'
  c=0
  for i in $1
  do
    cut -d' ' -f1,2 ${genos}/${i}.fam|sed "s/$/ $c/" >> Covs.txt
    c=$(( c + 1 ))
  done
  covs='--covs Covs.txt'
}

split_train_test()
{
  echo 'Splitting test and training sets'
  c=0
  for i in $1
  do
    echo -e "\tProcessing $i"
    sort -R ${genos}/${i}.keep| head -n ${sample} > ${i}.train
    comm -23 <(sort ${genos}/${i}.keep) <(sort ${i}.train) > ${i}.test
  done
}

gen_tests()
{
  echo 'Splitting test and training sets'
  c=0
  for i in $1
  do
    ${plink} --bfile ${genos}/${i} --keep ${i}.test --make-bed --out ${i}_test ${common_plink}
  done
}


pops="${pop1} ${pop2} ${pop3}"

common_plink="--keep-allele-order --allow-no-sex --threads ${cpus} --memory ${mem}"
common_pheno="-m 100 -b 0.5 -f 0 -t ${cpus} --force_h2 -M ${membytes} ${covs}"

if [ "$covs" == TRUE ]
  then
    do_covs "${pops}"
fi

all=`echo ${pops} | tr ' ' 'n'`
if [ ! -f ${genos}/${all}.bed ]
    then
      echo -e "\n\nGenerating merged fileset"
      echo -e "${genos}/${pop1}\n${genos}/${pop2}\n${genos}/${pop3}" > merge.list
      comm -12 <(comm -12 <(sort ${genos}/${pop1}.bim) <(sort ${genos}/${pop2}.bim)) <(sort ${genos}/$pop3.bim) > merged.totalsnps
      ${plink} --merge-list merge.list --extract merged.totalsnps --make-bed --out ${genos}/${all} ${common_plink}
    else
      echo -e "\n\nMerged fileset found!\n"
fi
all="${genos}/${all}"
# generate the phenos
if [ ! -f train.pheno ]; then
    echo -e "\n\nGenerating phenotypes\n"
    export plink
    python3 ${code}/qtraitsimulation.py -p train -B ${genos}/${all} ${common_pheno}
    cp all.totalsnps merged.totalsnps

#    python3 ${code}/qtraitsimulation.py -p ${pop2} -B ${genos}/${pop2} -2 ${genos}/${pop1} --causal_eff ${pop1}.causaleff ${common_pheno}
#    python3 ${code}/qtraitsimulation.py -p ${pop3} -B ${genos}/${pop3} -2 ${genos}/${pop1} --causal_eff ${pop1}.causaleff ${common_pheno}
#    cat ${pop1}.pheno ${pop2}.pheno ${pop3}.pheno > train.pheno
    else
      echo -e "\n\nPhenotypes already present... moving on\n"
fi

if [ ! -f ${pop3}.test ]; then
    echo -e "\n\nGenerating keep files"
    split_train_test "${pops}"
    else
      echo -e "\n\nKeep files already present\n"
fi

if [ ! -f ${pop3}_test.bed ]; then
    echo -e "\n\nGenerating test filesets"
    gen_tests "${pops}"
    else
      echo -e "\n\nTest filesets already present... moving on\n"
fi

#all=`echo ${pops} | tr ' ' 'n'`
#if [ ! -f ${all}.bed ]
#    then
#      echo -e "\n\nGenerating merged fileset"
#      echo -e "${genos}/${pop1}\n${genos}/${pop2}\n${genos}/${pop3}" > merge.list
#      comm -12 <(comm -12 <(sort ${pop1}.totalsnps) <(sort ${pop2}.totalsnps)) <(sort $pop3.totalsnps) > merged.totalsnps
#      ${plink} --merge-list merge.list --extract merged.totalsnps --make-bed --out ${all} ${common_plink}
#    else
#      echo -e "\n\nMerged fileset found!\n"
#fi

#make train subset
if [ ! -f train.txt ]
    then
        cat ${pop1}.train ${pop2}.train ${pop3}.train > train.keep
fi

if [ ! -f train.eigenvec  ]
    then
        #cut -f1,2,5,6,7,8 ${genos}/pca_proj_mydata.sscore| tail -n +2 > train.eigenvec
        cut -f1,2,5,6,7,8 ${genos}/pca_proj_mydata.sscore > train.eigenvec
        #$plink --bfile ${all} --pca 4 --out train ${common_plink}
fi

step=$(( sample/10 ))
sequence=`seq 0 $step $sample`
python -c "import numpy as np;from itertools import product;open('trios.txt','w').write('\n'.join([' '.join([str(np.round(y,2)) for y in x]) for x in product(np.arange(0,1,0.1), np.arange(0,1,0.1), np.arange(0,1,0.1)) if sum(x) == 1]))"
while read p
  do
    read eu as af <<<${p}
    if [[ ! ${as} = 0  ]]; then
        sort -R ${pop2}.train| head -n `bc <<< "(${as} * ${sample})/1"` > trio_frac.keep
    fi
    if [[ ! ${eu} = 0  ]]; then
        sort -R ${pop1}.train| head -n `bc <<< "(${eu} * ${sample})/1"` >> trio_frac.keep
    fi
    if [[ ! ${af} = 0  ]]; then
        sort -R ${pop3}.train| head -n `bc <<< "(${af} * ${sample})/1"` >> trio_frac.keep
    fi
    $plink --bfile ${all} --keep trio_frac.keep --linear hide-covar --pheno train.pheno --covar train.eigenvec --covar-name PC1_AVG --vif 100 --out trio ${common_plink}
    $plink --bfile ${all} --keep trio_frac.keep --clump trio.assoc.linear --clump-p1 0.01 --pheno train.pheno --out trio ${common_plink}
    awk -F' ' '{if (NR!=1) { print $3 }}' trio.clumped | xargs -n 100 -I {} grep {} trio.assoc.linear > trio.myscore
    sort -u trio.myscore > temp.txt && mv temp.txt trio.myscore
    for pop in $pops
    do
      $plink --bfile ${pop}_test --score trio.myscore 2 4 7 sum center --pheno train.pheno --out ${pop}_trio ${common_plink}
    done
    outp ${pop1}_trio.profile ${pop1} ${eu} trio.tsv
    outp ${pop2}_trio.profile ${pop2} ${as} trio.tsv
    outp ${pop3}_trio.profile ${pop3} ${af} trio.tsv
    perfrac ${pop1}_trio.profile ${pop2}_trio.profile ${pop3}_trio.profile trio_df.tsv
  done <trios.txt