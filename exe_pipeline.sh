#! \usr\bin\sh

python /Users/jshleap/my_gits/Cottagging_playground/qtraitsimulation.py -p EUR -m 100 -b 0.66 -P ~/Programs/plink_mac/plink -t 8 -M 3000 -B ../EUR10K_5K

python /Users/jshleap/my_gits/Cottagging_playground/qtraitsimulation.py -p AFR -m 100 -b 0.66 -P ~/Programs/plink_mac/plink -t 8 -M 3000 -B ../AFR10K_5K --causal_eff EUR.full

python /Users/jshleap/my_gits/Cottagging_playground/plinkGWAS.py -p EUR -B ../EUR10K_5K -f EUR.pheno -P ~/Programs/plink_mac/plink -v 5 -V EUR.totalsnps -s -l -t 8 -M 3000

python /Users/jshleap/my_gits/Cottagging_playground/ppt.py -b EUR_test -p EURppt -s EUR_gwas.assoc.linear -P EUR.pheno -n ~/Programs/plink_mac/plink -d 0.8 -C 1.0,0.8,0.5,0.4,0.3,0.2,0.1,0.08,0.05,0.02,0.01,10E-3,10E-4,10E-5,10E-6,10E-7,10E-8 -z -L EUR -t

python /Users/jshleap/my_gits/Cottagging_playground/ppt.py -b ../AFR10K_5K  -p AFRppt -s EUR_gwas.assoc.linear -P AFR.pheno -n ~/Programs/plink_mac/plink -d 0.8 -C 1.0,0.8,0.5,0.4,0.3,0.2,0.1,0.08,0.05,0.02,0.01,10E-3,10E-4,10E-5,10E-6,10E-7,10E-8 -z -L AFR -t

python /Users/jshleap/my_gits/Cottagging_playground/ranumo.py -p Null -g EUR_gwas.assoc.linear -b ../AFR10K_5K -R EUR_test -f AFR.pheno -i EUR.pheno -P ~/Programs/plink_mac/plink -c /Volumes/project/gravel/hleap_projects/Cotagging/newsim/test/multiSNP/Linear10K/Cotagging10K_5K_taggingscores.tsv -s 1 -l EUR AFR -r EURppt.results -t AFRppt.results -H 0.66 -F 0.1 -q pdf -y

python /Users/jshleap/my_gits/Cottagging_playground/prankcster.py -p prancster -b EUR_test -c ../AFR10K_5K -L EUR AFR -T AFRppt.results -r EURppt.results -R Null_merged.tsv -d /Volumes/project/gravel/hleap_projects/Cotagging/newsim/test/multiSNP/Linear10K/Cotagging10K_5K_taggingscores.tsv -s EUR_gwas.assoc.linear -f AFR.pheno -t 8 -P ~/Programs/plink_mac/plink -E 1 -S 0.1 -g sum -F 0.1 -H 0.66 -y
