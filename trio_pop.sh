#!/usr/bin/env bash

#!/bin/sh -xv
set -e
source ~/.bashrc
cwd=$PWD
cpus=16
mem=37000
membytes=$(( mem * 1000000 ))
# Path to genotypes
genos=$1
# Path to code
code=$2
# Path and binary of plink
# TODO: remove it as argument and assume is in the path??
plink=$3
# Number of individuals to be scored
sample=$4
# Space separated string (needs to be quoted) with the populations prefix
pops=$5
# Whether to use covariates of not. Dafault is Not
covs=$6

echo "Executing code with the following arguments:"
echo "genos = ${genos}"
echo "code = ${code}"
echo "plink = ${plink}"
echo "sample = ${sample}"
echo "pops = ${pops}"
echo "covs = ${covs}"

corr()
{
  awk 'pass==1 {sx+=$3; sy+=$6; n+=1} pass==2 {mx=sx/(n-1)
  my=sy/(n-1); cov+=($3-mx)*($6-my)
  ssdx+=($3-mx)*($3-mx); ssdy+=($6-my)*($6-my);} END {
  print (cov / ( sqrt(ssdx) * sqrt(ssdy)) )^2 }' pass=1 $1 pass=2 $1
}
outp()
{
  if [ ! -f $4 ]; then echo -e "Pop\t\$R^2$\tFrac\tIndex" > $4; fi
  echo -e "$2\t`corr $1`\t$3\t$5" >> $4
}

perfrac()
{
  preline="\$R^2_{${pop1}}$\t\$R^2_{${pop2}}$\t\$R^2_{${pop3}}$"
  if [ ! -f trio_df.tsv ]; then
    echo -e "${pop1}\t${pop2}\t${pop3}\t${preline}" > trio_df.tsv
  fi
  echo -e "$1\t$3\t$5\t`corr $2`\t`corr $4`\t`corr $6`" >> trio_df.tsv
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

read -r -a array <<< "${pops}"
for i in `seq 0 $(( ${#array[@]} - 1 ))`
do
  j=$(( i + 1 ))
  echo -e "\ni is $i and j is $j, declaring pop${j} as ${array[i]}"
  declare pop${j}=${array[i]}
done


if [ "$covs" == TRUE ]
  then
    do_covs "${pops}"
fi

common_pheno="-m 100 -b 0.5 -f 0 -t ${cpus} --force_h2 -M ${membytes} ${covs}"
common_plink="--keep-allele-order --allow-no-sex --threads ${cpus} --memory ${mem}"


all=`echo ${pops} | tr ' ' 'n'`
if [ ! -f ${genos}/${all}.bed ]
    then
      echo -e "\nGenerating merged fileset"
      echo -e "${genos}/${pop1}\n${genos}/${pop2}\n${genos}/${pop3}" > merge.list
      comm -12 <(comm -12 <(sort ${genos}/${pop1}.bim) <(sort ${genos}/${pop2}.bim)) <(sort ${genos}/$pop3.bim) > merged.totalsnps
      ${plink} --merge-list merge.list --extract merged.totalsnps --make-bed --out ${genos}/${all} ${common_plink}
    else
      echo -e "\nMerged fileset found!\n"
fi
all="${genos}/${all}"
# generate the phenos
if [ ! -f train.pheno ]; then
    echo -e "\nGenerating phenotypes\n"
    export plink
    python3 ${code}/qtraitsimulation.py -p train -B ${all} ${common_pheno}
    else
      echo -e "\nPhenotypes already present... moving on\n"
fi

if [ ! -f ${pop3}.test ]; then
    echo -e "\nGenerating keep files"
    split_train_test "${pops}"
    else
      echo -e "\nKeep files already present\n"
fi

if [ ! -f ${pop3}_test.bed ]; then
    echo -e "\nGenerating test filesets"
    gen_tests "${pops}"
    else
      echo -e "\nTest filesets already present... moving on\n"
fi


#make train subset
if [ ! -f train.txt ]
    then
        cat ${pop1}.train ${pop2}.train ${pop3}.train > train.keep
fi

#if [ ! -f train.eigenvec  ]
#    then
#        #cut -f1,2,5,6,7,8 ${genos}/pca_proj_mydata.sscore| tail -n +2 > train.eigenvec
#        cut -f1,2,5,6,7,8 ${genos}/pca_proj_mydata.sscore > train.eigenvec
#        #${plink} --bfile ${all} --pca 4 --out train ${common_plink}
#fi

step=$(( sample/10 ))
sequence=`seq 0 $step $sample`
if [ ! -f trios.txt ]
 then python -c "import numpy as np;from itertools import product;open('trios.txt','w').write('\n'.join([' '.join([str(np.round(y,2)) for y in x]) for x in product(np.arange(0,1,0.1), np.arange(0,1,0.1), np.arange(0,1,0.1)) if sum(x) == 1]))"
fi
# patching for missing fractions
echo -e "\n0.1 0.9 0.0\n0.3 0.6 0.1\n0.6 0.3 0.1\n0.7 0.2 0.1\n0.2 0.7 0.1\n" >> trios.txt
sed '/^$/d' trios.txt | sort -u > temp  && mv temp trios.txt
if [ -f done.txt ]; then
  comm -3 <(sort trios.txt) <(sort done.txt) > execute.txt
  else
    cp trios.txt execute.txt
fi
index=0
while read -r -a p
  do
    eu=${p[0]}
    as=${p[1]}
    af=${p[2]}
    fn="trio_${eu}_${as}_${af}.keep"
    echo -e "\nExecuting ${fn%.keep} on ${sample} individuals"
    if [ -f ${fn} ]; then rm ${fn}; fi
    t=`bc <<< "(${eu} == 0)"`
    if [ ${t} -ne 1 ]; then
      eur=`bc <<< "(${eu} * ${sample})/1"`
      echo "  Including ${eur} individuals from ${pop1}"
      sort -R ${pop1}.train| head -n ${eur} >> ${fn}
    fi
    t=`bc <<< "(${as} == 0)"`
    if [ ${t} -ne 1 ]; then
      asn=`bc <<< "(${as} * ${sample})/1"`
      echo "  Including ${asn} individuals from ${pop2}"
      sort -R ${pop2}.train| head -n ${asn} >> ${fn}
    fi
    t=`bc <<< "(${af} == 0)"`
    if [ ${t} -ne 1 ]; then
      afr=`bc <<< "(${af} * ${sample})/1"`
      echo "  Including ${afr} individuals from ${pop3}"
      sort -R ${pop3}.train| head -n ${afr} >> ${fn}
    fi
    pcs='PC1 PC2 PC3 PC4'
    outfn=${fn%.keep}
    echo -e "\nComputing summary statistics for $fn\n"
    ${plink} --bfile ${all} --keep ${fn} --make-bed --out current ${common_plink}
    flashpca --bfile current -n ${cpus} -m ${mem} -d 4
    ${plink} --bfile current --linear hide-covar --pheno train.pheno --covar pcs.txt --covar-name ${pcs} --out ${outfn} ${common_plink}
    echo -e "\nClumping for $fn"
    ${plink} --bfile ${all} --keep ${fn} --clump ${outfn}.assoc.linear --clump-p1 0.01 --pheno train.pheno --out ${outfn} ${common_plink}

    if [ -f ${outfn}.clumped ]; then
      awk -F' ' '{if (NR!=1) { print $3 }}' ${outfn}.clumped | xargs -n 100 -I {} grep {} ${outfn}.assoc.linear > ${outfn}.myscore
    else
      echo -e "${eu} ${as} ${af}" >> done.txt
      echo -e "\n${fn%.keep} failed"
      continue
    fi

    sort -u ${outfn}.myscore > temp.txt && mv temp.txt ${outfn}.myscore
    for k in `seq 0 $(( ${#array[@]} - 1 ))`
    do
      pop=${array[k]}
      if [ -z ${p[k]} ]; then f="NA";else f=${p[k]};fi
      echo -e "\nScoring in ${pop}"
      ${plink} --bfile ${pop}_test --score ${outfn}.myscore 2 4 7 sum center --pheno train.pheno --out ${pop}_${outfn} ${common_plink}
      outp ${pop}_${outfn}.profile ${pop} ${f} trio.tsv ${index}
    done
    index=$(( index + 1 ))
#    outp ${pop1}_${outfn}.profile ${pop1} ${eu} trio.tsv
#    outp ${pop2}_${outfn}.profile ${pop2} ${as} trio.tsv
#    outp ${pop3}_${outfn}.profile ${pop3} ${af} trio.tsv
    perfrac ${eu} ${pop1}_${outfn}.profile ${as} ${pop2}_${outfn}.profile ${af} ${pop3}_${outfn}.profile
    echo -e "${eu} ${as} ${af}" >> done.txt
  done <execute.txt