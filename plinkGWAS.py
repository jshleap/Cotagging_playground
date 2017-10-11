#!/usr/bin/env python
#coding:utf-8
"""
  Author:  Jose Sergio Hleap --<>
  Purpose: Run plink gwas analyses
  Created: 09/30/17
"""
import os
import argparse
import matplotlib
import numpy as np
import pandas as pd
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utilities4cotagging import train_test, executeLine
plt.style.use('ggplot')

#----------------------------------------------------------------------
def gwas(plinkexe, bfile, outprefix, pheno, covs=None, nosex=False, 
         threads=False, maxmem=False, validsnpsfile=None):
    """
    Execute plink gwas. This assumes continuos phenotype and that standard qc 
    has been done already

    :param str plinkexe: Path and executable of plink
    :param str bfile: Prefix for the bed fileset
    :param str outprefix: Prefix for the outputs
    :param str pheno: Filename to phenotype in plink format
    :param bool nosex: If option --allow-no-sex should be used
    :param str covs: Filename with the covariates to use
    :param str validsnpsfile: file with valid snps to pass to --extract 
    :param int maxmem: Maximum allowed memory
    :param int trheads: Maximum number of threads to use
    """   
    ## for plinkgwas string:
    ## 1) plink path and executable
    ## 2) prefix for the bed fileset
    ## 3) Name of covariate file
    ## 4) names of the columns to use in the covariance file separated by "," 
    ## or '-' if range
    ## 5) prefix for outputs
    # Format CLA
    plinkgwas = "%s --bfile %s --assoc fisher-midp --linear --pheno %s"
    plinkgwas+= " --prune --out %s_gwas --ci 0.95 --keep-allele-order --vif 100"
    plinkgwas = plinkgwas%(plinkexe, bfile, pheno, outprefix)
    # Include a subset of snps file if required
    if validsnpsfile is not None:
        plinkgwas+= " --extract %s" % validsnpsfile
    # Include Covariates
    if covs:
        plinkgwas += " --covar %s keep-pheno-on-missing-cov" % covs
    # Ignore sex
    if nosex:
        plinkgwas += ' --allow-no-sex'
    else:
        plinkgwas += ' --sex'
    # Inlcude threads and memory
    if threads:
        plinkgwas += ' --threads %s' % threads
    if maxmem:
        plinkgwas += ' --memory %s' % maxmem 
    # execulte CLA
    out = executeLine(plinkgwas)  
    # Set the Outfile
    fn = '%s_gwas.assoc.linear' % outprefix
    # Return the dataframe of the GWAS and its filename
    return pd.read_table(fn, delim_whitespace=True), fn


#----------------------------------------------------------------------
def manhattan_plot(outfn, p_values, causal_pos=None, alpha = 0.05, title=''):
    """ 
    Generates a manhattan plot for a list of p-values. Overlays a horizontal 
    line indicating the Bonferroni significance threshold assuming all p-values 
    derive from independent test.
    
    :param str outfn: Outfilename
    :param list causal_pos: List with floats to include vertical lines as causal
    :param list p-values: A list of single-test p-values
    :param float alpha: The single-site significance threshold
    :param str title: Include a title in the plot
    """
    # Get the lenght of the p-values array
    L = len(p_values)
    # Compute the bonferrony corrected threshold
    bonferroni_threshold = alpha / L 
    # Make it log
    logBT = -np.log10(bonferroni_threshold)
    # Fix infinites
    p_values = np.array(p_values)
    p_values[np.where(p_values < 1E-10)] = 1E-10
    # Make the values logaritm
    vals = -np.log10(p_values)
    # Plot it
    fig = plt.figure()
    ax2 = fig.add_subplot(111)
    # Add threshold line
    ax2.axhline(y=logBT, linewidth=1, color='r', ls='--')
    # Add shaded regions on the causal positions
    if causal_pos is not None:
        [ax2.axvspan(x-0.2,x+0.2, facecolor='0.8', alpha=0.8) for x in 
         causal_pos]
    # Plot one point per value
    ax2.plot(vals, '.', ms=1)
    # Zoom-in / limit the view to different portions of the data
    Ymin = min(vals)
    Ymax = max(vals)
    ax2.set_ylim(0, Ymax+0.2)  # most of the data
    ax2.set_xlim([-0.2, len(vals)+1])
    plt.xlabel( r"marker index")
    plt.ylabel( r"-log10(p-value)")
    plt.title(title)
    plt.savefig(outfn)
    
#----------------------------------------------------------------------
def plink_gwas(plinkexe, bfile, outprefix, pheno, covs=None, nosex=False,
               threads=False, maxmem=False, validate=None, validsnpsfile=None, 
               plot=False, causal_pos=None):
    """
    execute the gwas
    
    :param str plinkexe: Path and executable of plink
    :param str bfile: Prefix for the bed fileset
    :param str outprefix: Prefix for the outputs
    :param bool nosex: If option --allow-no-sex should be used
    :param str covs: Filename with the covariates to use
    :param int/bool validate: To split the bed into test and training 
    """
    # Create a test/train validation sets
    if validate is not None:
        train, test = train_test(outprefix, bfile, plinkexe, validate, maxmem,
                                 threads)
    else:
        train, test = bfile, None   
    # Run the GWAS
    gws, fn = gwas(plinkexe, train, outprefix, pheno, covs=covs, nosex=nosex, 
               threads=threads, maxmem=maxmem, validsnpsfile=validsnpsfile)
    # Make a manhatan plot
    if plot:
        manhattan_plot('%s.manhatan.pdf' % outprefix, gws.P, causal_pos, 
                       alpha=0.05)
    # Return the gwas dataframe, its filename, and the train and test sets 
    # prefix
    return gws, os.path.join(os.getcwd(), fn), train, test
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prefix', help='prefix for outputs', 
                        required=True)  
    parser.add_argument('-B', '--bfile', default='EUR')
    parser.add_argument('-f', '--pheno', help='Phenotype file', required=True)
    parser.add_argument('-P', '--plinkexe', default='~/Programs/plink_mac/plink'
                        )
    parser.add_argument('-v', '--validate', default=None, type=int)
    parser.add_argument('-V', '--validsnpsfile', default=None)
    parser.add_argument('-C', '--covs', default=None, action='store')
    parser.add_argument('-s', '--nosex', default=False, action='store_true')
    parser.add_argument('-l', '--plot', help=('Generate a manhatan plot'),
                        default=False, action='store_true') 
    parser.add_argument('-t', '--threads', default=1, type=int)
    parser.add_argument('-M', '--maxmem', default=1700, type=int)    
    args = parser.parse_args()
    plink_gwas(args.plinkexe, args.bfile, args.prefix,args.pheno,covs=args.covs, 
               nosex=args.nosex, threads=args.threads, maxmem=args.maxmem,
               validate=args.validate, validsnpsfile=args.validsnpsfile, 
               plot=args.plot)     