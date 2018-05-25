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
  infn=$1
  n="${infn//[!0-9]/}"
  Pop=$2
  outfn=$3
  echo -e "$n\t`corr $infn`\t$Pop" >> $outfn
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
    do_covs ${pops}
fi

# generate the phenos
if [ ! -f train.pheno ]; then
    echo -e "\n\nGenerating phenotypes\n"
    export plink
    python3 ${code}/qtraitsimulation.py -p ${pop1} -B ${genos}/${pop1} -2 ${genos}/${pop2} ${common_pheno}
    python3 ${code}/qtraitsimulation.py -p ${pop2} -B ${genos}/${pop2} -2 ${genos}/${pop1} --causal_eff ${pop1}.causaleff ${common_pheno}
    python3 ${code}/qtraitsimulation.py -p ${pop3} -B ${genos}/${pop3} -2 ${genos}/${pop1} --causal_eff ${pop1}.causaleff ${common_pheno}
    cat ${pop1}.pheno ${pop2}.pheno ${pop3}.pheno > train.pheno
    else
      echo -e "\n\nPhenotypes already present... moving on\n"
fi

if [ ! -f ${pop3}.test ]; then
    echo -e "\n\nGenerating keep files"
    split_train_test ${pops}
    else
      echo -e "\n\nKeep files already present\n"
fi

if [ ! -f ${pop3}_test.bed ]; then
    echo -e "\n\nGenerating test filesets"
    gen_tests ${pops}
    else
      echo -e "\n\nTest filesets already present... moving on\n"
fi

all=`echo ${pops} | tr ' ' 'n'`
if [ ! -f ${all}.bed ]
    then
      echo -e "\n\nGenerating merged fileset"
      echo -e "${genos}/${pop1}\n${genos}/${pop2}\n${genos}/${pop3}" > merge.list
      comm -12 <(sort ${pop1}.totalsnps) <(sort ${pop2}.totalsnps) <(sort ${pop3}.totalsnps) > merged.totalsnps
      ${plink} --merge-list merge.list --extract merged.totalsnps --make-bed --out ${all} ${common_plink}
    else
      echo -e "\n\nMerged fileset found!\n"
fi

#make train subset
if [ ! -f train.txt ]
    then
        cat ${pop1}.train ${pop2}.train ${pop3}.train > train.keep
fi

pca=${genos}/
step=$(( sample/10 ))