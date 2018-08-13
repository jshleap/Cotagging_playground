library(latex2exp)
library(viridis)
library(ggtern)

df = read.table('Averaged_trio.tsv', sep='\t', header=TRUE, row.names=1)

AFR = ggtern(df,aes(EUR,ASN,AFR)) +  scale_fill_viridis(TeX('R^2_{AFR}'), 
                                                        option='plasma', direction=-1) +
  geom_hex_tern(binwidth=0.001,  aes(value=X.R.2_.AFR..), fun=mean)
EUR = ggtern(df,aes(EUR,ASN,AFR)) +  scale_fill_viridis(TeX('R^2_{EUR}'), 
                                                        option='plasma', direction=-1) +
  geom_hex_tern(binwidth=0.001,  aes(value=X.R.2_.EUR..), fun=mean)
ASN = ggtern(df,aes(EUR,ASN,AFR)) +  scale_fill_viridis(TeX('R^2_{ASN}'), 
                                                        option='plasma', direction=-1) +
  geom_hex_tern(binwidth=0.001,  aes(value=X.R.2_.ASN..), fun=mean)

ggsave("./AFR_trio.pdf",AFR,width=10,height=10)
ggsave("./EUR_trio.pdf",EUR,width=10,height=10)
ggsave("./ASN_trio.pdf",ASN,width=10,height=10)

