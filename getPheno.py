import pandas as pd
from sys import argv

# Arg1 = pheno file
# Arg2 = pheno name
# Arg3 = Ethnicity
# Arg4 = prefix
## read
df = pd.read_table(argv[1],sep='\t')

## Drop missing
df = df[(df.Ethnicity == argv[3]) & (df.loc[:, argv[2]] != -9)]

## Write new pheno
df.loc[:,['IID','FID', argv[2]]].to_csv('%s.pheno'%(argv[4]),header=False, 
                                        index=False, sep=' ')

## Write individuals subset
df.loc[:,['IID','FID']].to_csv('%s.keep'%(argv[4]), header=False, index=False, 
                               sep=' ')