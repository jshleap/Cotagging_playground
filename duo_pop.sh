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

compute_duo()
{
  # 1 : prefix of output
  # 2 : fraction computed
  # 3 : Path to merged plink fileset
  # 4 : flags common to plink calls
  # 5 : vector with names of populations
  pcs='PC1 PC2 PC3 PC4'
  prefix="${1}_${2}"
  if [ ! -f ${prefix}.clumped ]
  then
    echo -e "\nComputing summary statistics for ${prefix}\n"
    ${plink} --bfile $3 --keep ${2}.keep --make-bed --out current_prop $4
    flashpca --bfile current_prop -n ${cpus} -m ${mem} -d 4
    $plink --bfile current_prop --linear hide-covar --pheno train.pheno --covar pcs.txt --covar-name ${pcs} --out ${prefix} $4
    $plink --bfile current_prop --clump ${prefix}.assoc.linear --clump-p1 0.01 --pheno train.pheno --out ${prefix} $4
  else
    echo -e "${prefix} has already been done"
  fi
  grep -w "$(awk -F' ' '{if (NR!=1) { print $3 }}' ${prefix}.clumped)" ${prefix}.assoc.linear > ${prefix}.myscore
  for pop in $5
  do
    $plink --bfile ${pop}_test --score ${prefix}.myscore 2 4 7 sum center --pheno train.pheno --out ${pop}_${prefix} $4
    outp ${pop}_${prefix}.profile ${pop} ${1}.tsv
  done
}

common_plink="--keep-allele-order --allow-no-sex --threads ${cpus} --memory ${mem}"
common_pheno="-m 100 -b 0.5 -f 0 -t ${cpus} --force_h2 -M ${membytes} ${covs}"

if [ "$covs" == TRUE ]
  then
    cut -d' ' -f1,2 ${genos}/${target}.fam|sed 's/$/ 1/' > Covs.txt
    cut -d' ' -f1,2 ${genos}/EUR.fam|sed 's/$/ 0/' >> Covs.txt
    covs='--covs Covs.txt'
fi

if [ ! -f ${genos}/EURnASNnAFRnAD.bed ]
    then
      echo -e "\n\nGenerating merged filesets"
      echo -e "${genos}/EUR\n${genos}/ASN\n${genos}/AFR\n${genos}/AD" > merge.list
      comm -12 <(comm -12 <(comm -12 <(sort ${genos}/EUR.bim) <(sort ${genos}/ASN.bim)) <(sort ${genos}/AFR.bim)) <(sort ${genos}/AD.bim) > merged.totalsnps
      ${plink} --merge-list merge.list --extract merged.totalsnps --make-bed --out ${genos}/EURnASNnAFRnAD ${common_plink}
      cat ${genos}/EUR.fam ${genos}/${target}.fam > duo.keep
      ${plink} --bfile ${genos}/EURnASNnAFRnAD --keep duo.keep --make-bed --out ${genos}/EURn${target} ${common_plink}
fi

pops4=${genos}/EURnASNnAFRnAD
all=${genos}/EURn${target}
# generate the phenos
if [ ! -f train.pheno ]; then
    echo -e "\n\nGenerating phenotypes\n"
    export plink
    python3 ${code}/qtraitsimulation.py -p train -B ${pops4} ${common_pheno}
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
    $plink --bfile ${genos}/${target} --keep ${target}.test --make-bed --out ${target}_test ${common_plink}
    $plink --bfile ${genos}/EUR --keep EUR.test --make-bed --out EUR_test ${common_plink}
fi


#make train subset
if [ ! -f train.txt ]
    then
        cat ${target}.train EUR.train > train.keep
        cat train.keep initial.keep | sort | uniq > train.txt
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
    else
        sequ=$sequence
fi

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
    fi
    # Compute sumstats and clump for proportions
    compute_duo proportions ${i} ${all} ${common_plink} "EUR ${target}" ${i}.keep
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
   compute_duo init ${i} ${all} ${common_plink} "EUR ${target}" init_${i}.keep
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
    compute_duo cost ${j} ${all} ${common_plink} "EUR ${target}" frac_${j}.keep
done