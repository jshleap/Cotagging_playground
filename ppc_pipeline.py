#!/usr/bin/env python
#coding:utf-8
"""
  Author:  ppc_pipeline --<>
  Purpose: Run simulation, P+T, null model and P+C
  Created: 10/04/17
"""

from qtraitsimulation import qtraits_simulation

def execute(args):
    """
    Execute pipeline
    """
    prs_eur = qtraits_simulation(args.prefi2135x, args.bfile, args.h2, args.ncausal, 
                             args.plinkexe)

if __name__ == '__main__':
   