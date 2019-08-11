from Bio import Entrez
import nltk
import sys

def fetchPMC(query):
    Entrez.email = "jose.hleaplozano@mcgill.ca"
    # reldate can be used to limit the time (in days)
    with Entrez.esearch(db="pmc", term=query, usehistory="y") as handle:
        res=Entrez.read(handle)
    env=res["WebEnv"]
    key=res["QueryKey"]
    try:
        with Entrez.efetch(db="pmc", retmax=10000, retmode="xml", webenv=env,
                           query_key=key) as fetch:
            data=fetch.read()
        return data
    except:
        return None


query='("UK biobank" OR "UKBB") AND (GWAS OR "Genome Wide Association")'
data=fetchPMC(query)
if data == None:
    print("No data")
else:
    with open("pmcresults.txt" ,"w") as f:
        f.write(data)

# try to process it with NLTK? nltk.corpus.reader.xmldocs
# try context search? <text>.concordance(<query>)