#! \usr\bin\sh
#Argument 1: number of causal variants
#Argument 2: strategy (so fat sum and hyperbola
#Argument 3: column to be optimized Cotagging, Tagging EUR, Tagging EUR
#Argument 4: heritability
#Argument 5: extra arguments for qtraitsimulation
path_to_bed=/Volumes/project/gravel/hleap_projects/Cotagging/newsim/test/multiSNP/Linear10K
bins=/Users/jshleap/my_gits/Cottagging_playground
plink=/Users/jshleap/Programs/plink_mac/plink
threshs=1.0,0.8,0.5,0.4,0.3,0.2,0.1,0.08,0.05,0.02,0.01,10E-3,10E-4,10E-5,10E-6,10E-7,10E-8
#threshs=1.0,0.5,0.1,0.05,0.01,10E-3,10E-5,10E-7,10E-8

python $bins/qtraitsimulation.py -p EUR -m $1 -b $4 -P $plink -t 1 -M 3000 -B $path_to_bed/EUR10K_5K -2 $path_to_bed/AFR10K_5K $5

python $bins/qtraitsimulation.py -p AFR -m $1 -b $4 -P $plink -t 1 -M 3000 -B $path_to_bed/AFR10K_5K -2  $path_to_bed/EUR10K_5K  --causal_eff EUR.full $5

python $bins/plinkGWAS.py -p EUR -B $path_to_bed/EUR10K_5K -f EUR.pheno -P $plink -v 5 -V EUR.totalsnps -s -l -t 1 -M 3000

python $bins/ppt.py -b EUR_test -p EURppt -s EUR_gwas.assoc.linear -P EUR10K_5K_test.pheno -n $plink -d 0.8 -C $threshs -z -L EUR -t -T 1 -M 3000

python $bins/ppt.py -b $path_to_bed/AFR10K_5K  -p AFRppt -s EUR_gwas.assoc.linear -P AFR.pheno -n $plink -d 0.8 -C $threshs -z -L AFR -t -T 1 -M 3000

python $bins/ranumo.py -p Null -g EUR_gwas.assoc.linear -b $path_to_bed/AFR10K_5K -R EUR_test -f AFR.pheno -i EUR10K_5K_test.pheno -P $plink -c $path_to_bed/Cotagging10K_5K_taggingscores.tsv -s 1 -l EUR AFR -r EURppt.results -t AFRppt.results -H $4 -F 0.1 -q pdf -T 1 

python $bins/prankcster.py -p prancster -b EUR_test -c $path_to_bed/AFR10K_5K -L EUR AFR -T AFRppt.results -r EURppt.results -R Null_merged.tsv -d $path_to_bed/Cotagging10K_5K_taggingscores.tsv -s EUR_gwas.assoc.linear -f AFR.pheno -t 1 -P $plink -E 1 -S 0.1 -g $2 -F 0.1 -H $4 -Q Null.qrange -C "$3"
