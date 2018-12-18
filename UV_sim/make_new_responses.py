# -*- coding: utf-8 -*-
"""

Usage:
  mcmc_UV.py [options]

Options:
  -h --help  Show help text
  --abund=ABUND  what kind of abundance file do you want? [default: coronal]

"""
from docopt import docopt
from pylab import *
import scipy.optimize as opt
import pyfits as pf
import numpy as np
import pickle
from emeasure import *
from plot_output import *
import ChiantiPy.filters as chfilters
import ChiantiPy.core as ch
from response import *
from upper_limit import *
from bkg import *
import math
from UV_atmos import *
import emcee
import matplotlib.pyplot as pl
import triangle
from scipy.stats import *

data_dir = '/home/astro/phrmat/storage/XMM_HD209/analysis/'
#data_dir = '/Users/toml42/storage/XMM_HD209/analysis/'

def main():

#  NH = 20.8 # highest it could be
  NH = 18.37
  chrom_d = 1e8
  corona_d = 1e8

  arguments = docopt(__doc__)
  abund = arguments['--abund']
  generate_all(0,abund,NH,chrom_d,corona_d)
#  generate_all(40)


def generate_all(chrom,abund,NH,chrom_d,corona_d):

  import ChiantiPy.filters as chfilters
  import ChiantiPy.core as ch

#  note: it is best to cut off very low energy photons, as these seem to inspire odd behaviour in the 
# mos cameras.

  proc = 16

  cut =[0,25.0]

  t = 4.1 + 0.1*arange(40)

  em = [27.0]*len(t)

  density=array([chrom_d]*chrom + [corona_d]*(len(t) - chrom))

  x = em


  en = [0.15,2]

  c = 2.998e8
  h = 6.62607e-34
  Kev = 1.6e-16

  maxwvl = 920


# rosat band is 5 to 125 angstrom

  wvl_xray = np.linspace(0.1,250.0,10000)
  wvl_euv = np.linspace(5,maxwvl,(maxwvl-5)*10)

#  here's where new models are calculated
  grid_control('xray_b',t,wvl_xray,density,proc,abund)

#  grid_control('XUV_sanz_b',t,wvl_euv,density,proc,abund)

#  quit()

  # load the response matrices...
  rsp = load_response(data_dir + 'PN.rmf')
  rsp2 = load_response(data_dir + 'M1.rmf')
  rsp3 = load_response(data_dir + 'M2.rmf')

  # fold the X-rays through response matrix

  x_grid = [0.0]*len(t)

  for i in range(0,len(t)):
    pkl_file_x = open(abund+'_abund_spectra_bin/spectra_'+'xray_b'+'_'+str(t[i])+'_'+str(log10(density[i]))+'.pkl', 'rb')
    x_grid[i] = array(list(pickle.load(pkl_file_x)))
  fold_xrays(rsp,rsp2,rsp3,x_grid,wvl_xray,t,density,cut,NH,abund)

  e1 = 0.2
  e2 = 0.5
  e3 = 1.0
  e4 = 2.0
  e5 = 4.5
  e6 = 12

  b1 = [e1, e2]
  b2 = [e2, e3]
  b3 = [e3, e4]
  b4 = [e4, e5]
  b5 = [e5, e6]

  b1 = [0, 20]
  b2 = [0, 20]
  b3 = [0, 20]
  b4 = [0, 20]
  b5 = [0, 20]

  bands = {'b1':b1,'b2':b2,'b3':b3,'b4':b4,'b5':b5}

  for b in bands:
    for i in range(0,len(t)):
      pkl_file_x = open(abund+'_abund_spectra_bin/spectra_'+'xray_b'+'_'+str(t[i])+'_'+str(log10(density[i]))+'.pkl', 'rb')
      x_grid[i] = array(list(pickle.load(pkl_file_x)))
    fold_xrays(rsp,rsp2,rsp3,x_grid,wvl_xray,t,density,bands[b],NH,abund,b)

main()