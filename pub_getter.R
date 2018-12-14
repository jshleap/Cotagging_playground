# Title     : TODO
# Objective : Fetch publuications of GWAS AND UKBB
# Created by: jshleap
# Created on: 22/11/18

library('fulltext')

sources = c('crossref', 'plos', 'bmc', 'entrez', 'arxiv', 'biorxiv', 'europe_pmc', 'scopus')
# Make the search for GWAS first
queries = c("GWAS",'UK biobank', 'UKBB')
if (!file.exists("PLOS_intersect.R")){
  print('Getting PLOS')
  # plos + uk biobank 194
  plosgwas <- ft_search(query = "GWAS + 'UK biobank'", from = "plos", limit=1000)
  
#PLOS Search API rate limit: Please limit your API requests to 7200 requests a day,
# plos GWAS 4858
#plosgwas <- ft_search(query = "GWAS", from = "plos", limit=4858)
# plos UKBB 31
#plosukbb <- ft_search(query = "UKBB", from = "plos", limit=31)
# plos 'UK biobank' 2034
#plosukbb2 <- ft_search(query = "'UK biobank'", from = "plos", limit=2034)
#PLOS <- intersect(unique(union(plosukbb2$plos$data$id, plosukbb$plos$data$id)), plosgwas$plos$data$id)
save(PLOS, file = 'PLOS_intersect.R')}else{
    load('PLOS_intersect.R')
}

######
if(file.exists("BMC.R")){
    load("BMC.R")}else{
      print('Getting BMC')
    # BMC GWAS 15110
    bmcdois <- c()
    for (i in seq(1,15110,50)){
      gwas_bmc <- bmc_search('GWAS', offset = i, limit=50)
      bmcdois <- c(bmcdois, gwas_bmc$records$doi)
      }
    # BMC UKBB 879
    bmcukbb <- c()
    # BMC UK BioBank 1193
    for (j in seq(1,1193,50)){
        ukbb_bmc <- bmc_search('"UK biobank"', offset = j, limit=50)
        bmcukbb <- c(bmcukbb, ukbb_bmc$records$doi)
        }
    # BMC UKBB 879
    for (k in seq(1,879,50)){
        ukbb_bmc <- bmc_search('"UK biobank"', offset = k, limit=50)
        bmcukbb <- c(bmcukbb, ukbb_bmc$records$doi)
        }
    bmcukbb <- unique(bmcukbb)
    BMC <- intersect(bmcukbb, bmcdois)
    save(BMC, file = "BMC.R")
}
######

# CROSSREF search has a cap on 1000
# crossref GWAS 1013 (total) 981 (journals?) 974 with full text
if (file.exists("CROSSREF.R")){
  load("CROSSREF.R")} else {
  crossrefgwas <- ft_search(query = "GWAS", from = "crossref",
  crossrefopts=list(filter=c(type="journal-article", has_full_text = TRUE)), limit=1000)
  # crossref UKBB 1
  crossrefukbb <- ft_search(query = "UKBB", from = "crossref",
  filter=c(type="journal-article", has_full_text = TRUE))
  # crossref UK BioBank 68232
  crossrefukbb2 <- rcrossref::cr_works(query="'UK biobank'", cursor='*', cursor_max=68232, filter=c(type="journal-article", has_full_text = TRUE))
  
  CROSSREF = intersect(unique(c(crossrefukbb2$data$doi,
  crossrefukbb$crossref$data$doi)), crossrefgwas$data$doi)
  save(CROSSREF, file = "CROSSREF.R")
  }

####
# Entrez combined 12 enries
if (file.exists("ENTREZ.R")){
  load("ENTREZ.R")} else {
  entrez <- ft_search(query = 'GWAS AND "UK biobank" AND homo[organism]', from = "entrez", limit=1000)
  ENTREZ <- entrez$entrez$data$doi
  save(ENTREZ, file = "ENTREZ.R")
}
####
# arxiv gwas 153
if (file.exists("ARXIV.R")){
  load("ARXIV.R")} else{
  arxivgwas <- ft_search(query = 'GWAS', from = "arxiv", limit=153)
  # arxiv UKBB 1
  arxivukbb <- ft_search(query = 'UKBB', from = "arxiv")
  # arxiv UK BioBabnk 17
  arxivukbb1 <- ft_search(query = '"UK biobank"', from = "arxiv", limit=17)
  ARXIV <- intersect(unique(c(arxivukbb$arxiv$data$doi, arxivukbb1$arxiv$data$doi)), arxivgwas$arxiv$data$doi)
  save(ARXIV, file = "ARXIV.R")
  }
####
if (file.exists("BIOARXIV.R")){
  load("BIOARXIV.R")} else {
  # bioarxiv GWAS 2426
  bioarxivgwas <- biorxiv_search(query="GWAS", limit=2426)
  # bioarxiv UKBB 60
  bioarxivukbb <-  biorxiv_search(query="UKBB", limit=60)
  # bioarxiv UK bionank 592
  bioarxivukbb2 <- biorxiv_search(query='"UK biobank"', limit=592)
  BIOARXIV <- intersect(unique(c(bioarxivukbb2$data$doi, bioarxivukbb$data$doi)), bioarxivgwas$data$doi)
  save(BIOARXIV, file = "BIOARXIV.R")
  }
####
if (file.exists("SCOPUS.R")){
  # Scopus GWAS + "UK biobnak" 665
  SCOPUS <- ft_search(query='GWAS "UK biobank"', from='scopus', limit=665)$scopus$data$'prism:doi'
  SCOPUS <- SCOPUS[!is.na(SCOPUS)]
  save(SCOPUS, file = "SCOPUS.R")
  }

allofthem <- c(SCOPUS, BIOARXIV, ARXIV, ENTREZ, CROSSREF, BMC, PLOS)
allofthem <- unique(allofthem)
ft_get(allofthem, type=c('xml','pdf'))
write(allofthem, 'dois.txt')



