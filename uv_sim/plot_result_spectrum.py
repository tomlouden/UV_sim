# -*- coding: utf-8 -*-
"""
Uses the emcee module to explore the chiantipy parameter fitting space for DEM retrieval.

Usage:
  plot_result_spectrum.py [options] (<FILENAME>)

Options:
  -h --help  Show help text
  --n=n  what kind of abundance file do you want? [default: 1e6]

Run an emcee! yay!

"""
from docopt import docopt
#from pylab import *
import scipy.optimize as opt
import pyfits as pf
import numpy as np
import pickle
from emeasure import *
import chianti.filters as chfilters
import chianti.core as ch
from response import *
from upper_limit import *
from bkg import *
import math
from UV_atmos import *
from UV_model import *
import emcee
import triangle
from scipy.stats import *
import fitsio
import sys
from load_data import *
import matplotlib.pyplot as plt
from spectrum_breakdown import *

data_dir = '/home/astro/phrmat/storage/XMM_HD209/analysis/'
#data_dir = '/Users/toml42/storage/XMM_HD209/analysis/'

font = {'family' : 'normal', 'weight': 'normal', 'size': 12}

def main():

  import chianti.filters as chfilters
  import chianti.core as ch

  arguments = docopt(__doc__)
  nlines = float(arguments['--n'])
  chain_name=arguments['<FILENAME>']

  result, out_lp, out_fluxes, out_xcounts, abund, chrom, chroma_d, corona_d, NH, out_flux, out_mloss = load_chain(chain_name,nlines)

  efficiency = 0.2

  out_mloss = np.log10(efficiency*(10**(out_mloss)))

  one_sig_flux = np.percentile(out_flux, [50, 50+(68.27/2.0),50-(68.27/2.0)],axis=0)

  one_sig_mloss = np.percentile(out_mloss, [50, 50+(68.27/2.0),50-(68.27/2.0)],axis=0)

  low_lim_mloss = np.percentile(out_mloss, 50+(99.73/2.0),axis=0)

  print(low_lim_mloss)

  print('3 sig upper limit is',round((10**low_lim_mloss/1e10),2),' x 10^10 g/s')
  print(round((8*10**10)/(10**low_lim_mloss),2))

  print(one_sig_flux)

  one_sig_mloss_u =  10**one_sig_mloss

  one_sig_mloss_u[1] = round((one_sig_mloss_u[1] - one_sig_mloss_u[0])/ 1e10,2)
  one_sig_mloss_u[2] = round((one_sig_mloss_u[2] - one_sig_mloss_u[0])/ 1e10,2)
  one_sig_mloss_u[0] = round((one_sig_mloss_u[0])/ 1e10,2)

  print(one_sig_mloss_u)

#  one_sig_mloss[1] = round(one_sig_mloss[1] - one_sig_mloss[0],2)
#  one_sig_mloss[2] = round(one_sig_mloss[2] - one_sig_mloss[0],2)
#  one_sig_mloss[0] = round(one_sig_mloss[0],2)

#  print one_sig_mloss

#  mloss_histogram(out_mloss)

  best_x = result[argmax(out_lp)]

  new_sim = True

  sigma = 0.9973

  distance = 49.63
  rsun = 1.148
  msun = 1.162
  semi = 0.04516
  rp = 1.380 #(southworth 2010)
  mp = 0.714
  dem_file = 'hd209_full'
  dem_file = abund+'_abund_spectra_bin/'+dem_file

  proc = 16

  rs = 6.955e10
  pc = 3.08567758e18
  distance = distance*pc
  rsun  = rsun*rs
  au = 1.49597871e13
  r_j = 0.69911e10
  m_j = 1.8986e30
  msun = msun*1.99e33
  semi = au*semi
  mp = m_j*mp
  rp = r_j*rp

  listfile='full_line_list'

# the geometric factor changes surface intensity into flux from a sphere

  geometry = pi
  factor = geometry*((rsun/distance))**2.0
  lum = 4.0*np.pi*geometry*(rsun)**2.0

  t = 4.1 + 0.1*arange(40)

  density=array([chroma_d]*chrom + [corona_d]*(len(t) - chrom))

  #calculate the emission measures
#  new_fname = 'ch8_hd209_full'
#  star_dist = 47.1
#  star_radius = 1.14
#  density = 1e9
#  calc_em(new_fname,star_dist, star_radius, density)
#  quit()

  dt = t*0

  for i in range(1,(len(t) - 1)):
    dt[i] = (10.0**t[i+1] - 10.0**t[i-1])/2.0

  dt[0] = (10**(t[0]+0.1) - 10**(t[0]-0.1))/2.0
  dt[-1] = (10**(t[-1]+0.1) - 10**(t[-1]-0.1))/2.0

  c = 2.998e8
  h = 6.62607e-34
  Kev = 1.6e-16

  maxwvl = 920
  wvl_euv = np.linspace(5,maxwvl,(maxwvl-5)*10)

  euv_grid = [0.0]*len(t)

  bands = [[5,100],[100,920],[5,920]]

  x = np.percentile(result,50,axis=0)

  em = chebyshev(x,t)

  temp_contribution = []
  w_temp_contribution = []

  for i in range(0,len(t)):
    pkl_file_euv = open(abund+'_abund_spectra_bin/spectra_'+'XUV_sanz_b'+'_'+str(t[i])+'_'+str(log10(density[i]))+'.pkl', 'rb')
    euv_grid[i] = array(list(pickle.load(pkl_file_euv)))
    temp_contribution += [flux_breakdown(euv_grid[i],wvl_euv,bands=bands)]
    print(t[i], em[i])
    w_temp_contribution += [flux_breakdown(euv_grid[i]*10**em[i],wvl_euv,bands=bands)]


  temp_contribution = np.array(temp_contribution)
  w_temp_contribution = np.array(w_temp_contribution)

  total_cont = np.sum(temp_contribution,axis=0)
  w_total_cont = np.sum(w_temp_contribution,axis=0)

  temp_contribution = temp_contribution/total_cont[2]
  w_temp_contribution = w_temp_contribution/w_total_cont[2]



  temp_contribution = temp_contribution/max(temp_contribution[:,2])

  plt.plot(t,temp_contribution[:,0],'k:')
  plt.plot(t,temp_contribution[:,1],'k--')
  plt.plot(t,temp_contribution[:,2],'k')

  plt.ylim(0,1.1)

  plt.ylabel('Relative flux level')
  plt.xlabel('Log(T) [K]')
  tight_layout()

  plt.savefig('relative_flux.png')
  plt.savefig('relative_flux.pdf')

  plt.close()

  plt.plot(t,w_temp_contribution[:,0],'k:')
  plt.plot(t,w_temp_contribution[:,1],'k--')
  plt.plot(t,w_temp_contribution[:,2],'k')

  plt.ylabel('Fraction of ionising flux')
  plt.xlabel('Log(T) [K]')
  tight_layout()

  plt.savefig('w_relative_flux.png')
  plt.savefig('w_relative_flux.pdf')

  plt.close()

  run_name = chain_name
  star_name = 'hd209'

  with open('line_response_'+star_name+'_'+run_name+'.pkl', 'rb') as open_pickles:
    spec_dict = pickle.load(open_pickles)

  with open('xray_response_'+run_name+'.pkl', 'rb') as open_pickles:
    xray_temp = np.array(pickle.load(open_pickles)) + np.log10(lum)

  with open('rosat_response_'+run_name+'.pkl', 'rb') as open_pickles:
    rosat_temp = np.array(pickle.load(open_pickles)) + np.log10(lum)

  with open('XUV_response_'+run_name+'.pkl', 'rb') as open_pickles:
    XUV_temp = np.array(pickle.load(open_pickles)) + np.log10(lum)

  with open('sanz_euv_response_'+run_name+'.pkl', 'rb') as open_pickles:
    sanz_euv_temp = np.array(pickle.load(open_pickles)) + np.log10(lum)

  with open('high_euv_response_'+run_name+'.pkl', 'rb') as open_pickles:
    high_euv_temp = np.array(pickle.load(open_pickles)) + np.log10(lum)

  with open('all_euv_response_'+run_name+'.pkl', 'rb') as open_pickles:
    all_euv_temp = np.array(pickle.load(open_pickles)) + np.log10(lum)

  with open('xmm_response_'+run_name+'.pkl', 'rb') as open_pickles:
    xmm_temp = factor*np.array(pickle.load(open_pickles))

  with open('m_xmm_response_'+run_name+'.pkl', 'rb') as open_pickles:
    m_xmm_temp = pickle.load(open_pickles)


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

  bands = {'b1':b1,'b2':b2,'b3':b3,'b4':b4,'b5':b5}

  bands_xmm_temp = {}
  for b in bands:
    with open('xmm_response_'+run_name+b+'.pkl', 'rb') as open_pickles:
      bands_xmm_temp[b] = factor*np.array(pickle.load(open_pickles))

  best_em = chebyshev(best_x,t)

  print(best_x)
  print(best_em - log10(dt))

# recorded solar emission measure

  element_breakdown(em,t,density,wvl_euv)

  out_em = em - log10(dt)

  em2=chebyshev(array([-5.47607140e+02,4.56543607e+02,-7.10319631e+01,4.85300835e+00,-1.22315896e-01]),t) + log10(dt)

  plot_name = chain_name

  plot_euv_spec(t,em,distance,semi,euv_grid,wvl_euv,rsun,msun,mp,rp,NH,plot_name,factor)

  quit()

  plot_result(result,out_lp,dem_file,plot_name,factor,t,density,abund,m_xmm_temp,bands_xmm_temp,NH,solar=em2)

  plot_frac_errors(em,spec_dict,t)

  print('hd209')
  print('x_counts', 'xray_flux','rosat_flux', 'XUV_flux', 'sanz_euv_flux', 'high_euv_flux','all_euv_flux', 'beta')
  print(return_fluxes(em,t,xmm_temp,m_xmm_temp,bands_xmm_temp,xray_temp,rosat_temp,XUV_temp,sanz_euv_temp,high_euv_temp,all_euv_temp,factor))
  print('sun')
  print('x_counts', 'xray_flux','rosat_flux', 'XUV_flux', 'sanz_euv_flux', 'high_euv_flux','all_euv_flux', 'beta')
  u_f = return_fluxes(em2,t,xmm_temp,m_xmm_temp,bands_xmm_temp,xray_temp,rosat_temp,XUV_temp,sanz_euv_temp,high_euv_temp,all_euv_temp,factor)
  print(u_f)

  f = [t < 5.7 ]
  print('hd209')
  print('x_counts', 'xray_flux','rosat_flux', 'XUV_flux', 'sanz_euv_flux', 'high_euv_flux','all_euv_flux', 'beta')
  print(return_fluxes(em[f],t[f],xmm_temp,m_xmm_temp,bands_xmm_temp,xray_temp,rosat_temp,XUV_temp,sanz_euv_temp,high_euv_temp,all_euv_temp,factor))
  print('sun')
  print('x_counts', 'xray_flux','rosat_flux', 'XUV_flux', 'sanz_euv_flux', 'high_euv_flux','all_euv_flux', 'beta')
  u_f_2 = return_fluxes(em2[f],t[f],xmm_temp,m_xmm_temp,bands_xmm_temp,xray_temp,rosat_temp,XUV_temp,sanz_euv_temp,high_euv_temp,all_euv_temp,factor)
  print(u_f_2)

  orb_factor = pi*((rsun/semi))**2.0

  c = np.log10(orb_factor) - np.log10(lum)

  print(c)

  print('AT ORBITAL DISTANCE')
  print('x_counts', 'xray_flux','rosat_flux', 'XUV_flux', 'sanz_euv_flux', 'high_euv_flux','all_euv_flux', 'beta')
  print(return_fluxes(em,t,xmm_temp,m_xmm_temp,bands_xmm_temp,xray_temp+c,rosat_temp+c,XUV_temp+c,sanz_euv_temp+c,high_euv_temp+c,all_euv_temp+c,factor))

#  sun_fluxes = [u_f[0],u_f[3],u_f[7],u_f[5],u_f[4],u_f[2],u_f[6],u_f[1]]

  sun_fluxes = {'out_xcounts':u_f[0],'out_xray':u_f[3],'out_rosat':u_f[4],'out_XUV':u_f[5],'out_sanz':u_f[6],'out_high':u_f[7],'out_all':u_f[8],'out_beta':u_f[9]}

  flux_histograms(out_fluxes,plot_name,sun_fluxes)

def plot_frac_errors(em,spec_dict,t):

  output = new_DEM_fit(em,spec_dict,t)

  elementlist = []

  namelist = set(spec_dict['name'])

  print(namelist)

  for ion in namelist:
    print(ion, sum(output[np.array(spec_dict['name']) == ion]) / sum(spec_dict['flux'][np.array(spec_dict['name']) == ion]))

def mloss_histogram(out_mloss):
  plt.hist(out_mloss,histtype='step',color='k')
  show()


def flux_histograms(out_fluxes,plot_name,solar):

  label_x = ['Counts'] + ['Log10(Flux)'] + ['beta'] + ['Log10(Flux)']*5
  i = 0

  with open(plot_name+'_out_fluxes','w') as outfile:
    outline = 'bandname solar_value fit_value p_err n_err mean_solar sanz2011'
    print(outline)
    outfile.write(outline+'\n')

  for flux in out_fluxes:
    name = plot_name + '_' + flux

    # we had to make this solar correction, because all the luminosities assume a star the size
    # of HD209458, no the size of the sun.
    solar_correction = 0
    if label_x[i] != 'beta':
      solar_correction = log10(1.14**2)

    make_histogram(out_fluxes[flux],name,label_x[i],plot_name,solar[flux]-solar_correction)
    i += 1

def make_histogram(flux,name,label_x,chain_name,solar):


  three_sig = np.percentile(flux, [50, 50+(99.73/2.0),50-(99.73/2.0)],axis=0)
  one_sig = np.percentile(flux, [50, 50+(68.27/2.0),50-(68.27/2.0)],axis=0)

  perc = one_sig

  distance_frac_error = 1.97 / 49.63

  if (label_x != 'beta') & (label_x != 'xcounts'):
    perc[1] =   (perc[1] - perc[0]) + log10(1.0+distance_frac_error)
    perc[2] =   (perc[2] - perc[0]) - log10(1.0+distance_frac_error)
  else:
    perc[1] =   (perc[1] - perc[0])
    perc[2] =   (perc[2] - perc[0])

  perc = [round(p,2) for p in perc]

  mean_solar_x = 27.3784796138
  mean_solar_sanz = 27.9333605275
  mean_solar_XUV = 28.0401252951

  sanz_x = 26.40
  sanz_sanz = 27.74
  sanz_XUV = log10(10**sanz_x + 10**sanz_sanz)

  with open(chain_name+'_out_fluxes','a') as outfile:
    outline = name.split('_')[-1] +' '+str(round(solar,2))+' '+str(perc[0]) +' +'+str(perc[1]) +' '+ str(perc[2])
    if 'out_sanz' in name:
      outline += ' '+str(round(mean_solar_sanz,2))+' '+str(round(sanz_sanz,2))
    if 'out_xray' in name:
      outline += ' '+str(round(mean_solar_x,2))+' '+str(round(sanz_x,2))
    if 'out_XUV' in name:
      outline += ' '+str(round(mean_solar_XUV,2))+' '+str(round(sanz_XUV,2))
    print(outline)
    outfile.write(outline+'\n')
  plt.hist(flux,histtype='step',color='k')
  plt.xlabel(label_x)
  tight_layout()
  savefig(name+'.png')
  savefig(name+'.pdf')
  clf()

def m_xmm_lim(m_xmm_temp,t,flux):
  return log10(flux) - m_xmm_temp

def bands_xmm_lim(bands_xmm_temp,t,flux):
  print(log10(flux),log10(bands_xmm_temp))
  return(log10(flux) - log10(bands_xmm_temp))

def plot_result(all_x,lp,dem_file,plot_name,factor,t,density,abund,m_xmm_temp,bands_xmm_temp,nh,solar=[]):

  X_ray_counts = 23.8065922617

  m_xmm_measure = {'b1_temp':[7.57223e-16,2.20843e-16],'b2_temp':[0.0,9.02837e-17],'b3_temp':[9.23528e-18,6.75021e-17],'b4_temp':[2.00557e-16,3.05533e-16],'b5_temp':[2.98078e-16,1.10395e-15]}

  this_xlim = x_limit(t,density,X_ray_counts,30e3,factor,nh,abund)

  dt = t.copy()

  divide = 5.67

  emplot, fig = plot_dem(dem_file,dt,divide)

  for i in range(1,(len(t) - 1)):
    dt[i] = (10.0**t[i+1] - 10.0**t[i-1])/2.0

  dt[0] = (10**(t[0]+0.1) - 10**(t[0]-0.1))/2.0
  dt[-1] = (10**(t[-1]+0.1) - 10**(t[-1]-0.1))/2.0

#  emplot = plt.subplot(111)

#  font = {'family' : 'normal', 'weight': 'normal', 'size': 12}
  emplot.set_ylabel('Differential Emission Measure [cm$^{-5}$K$^{-1}$]', **font)
  emplot.set_xlabel('log(T) [K]', **font)

  em_grid = []

  for i in range(0,len(all_x)):
    x =  all_x[i]
    em = chebyshev(x,t)
    em_grid += [em]
#    if np.random.randint(10000) > 9900:
#      plot(t,(10.0**em)/dt,'k',alpha=0.01,lw=1)

  em_grid = array(em_grid)


  three_sig = np.percentile(em_grid, [50-(99.73/2.0), 50, 50+(99.73/2.0)],axis=0)
  one_sig = np.percentile(em_grid, [50-(68.27/2.0), 50, 50+(68.27/2.0)],axis=0)

  with open(plot_name+'_output_dem','w') as outfile:
    for i in range(0,len(t)):
      lower = '_{-'+str(round(one_sig[2][i] - one_sig[1][i],3))+'}'
      upper = '^{+'+str(round(one_sig[1][i] - one_sig[0][i],3))+'}'
      outline = str(t[i])+' & $'+ str(round(one_sig[1][i]-log10(dt[i]),3))+upper+lower+'$\\\\\n'
      outline = str(t[i])+' & $'+ str(round(one_sig[1][i]-log10(dt[i]),1))+'$\\\\\n'

      outfile.write(outline)

  perc = three_sig


  bf = chebyshev(all_x[argmax(lp)],t)

  if len(solar > 0):
    emplot.plot(t,(10.0**solar)/dt,'k--',lw=2,alpha=0.5)

  emplot.plot(t,(10.0**perc[1])/dt,'r',lw=2)

  emplot.plot(t,(10.0**perc[2])/dt,'r:',lw=2)

  emplot.plot(t,(10.0**perc[0])/dt,'r:',lw=2)

#  plot(t,(10.0**bf)/dt,'g:',lw=2)

#  plot(t,(10.0**xlim)/dt,'b')

#  for key in m_xmm_temp:
#    print key
#    result = m_xmm_lim(m_xmm_temp[key],t,(m_xmm_measure[key][0] + 3*m_xmm_measure[key][1])/factor)
#    print result
#    plot(t,(10.0**result)/dt,'k')

  PN_xmm_measure = {'b1':[0.000540662,0.000206315],'b2':[0.0,0.000156369],'b3':[0.0,4.52034e-05],'b4':[3.31264e-05,7.89352e-05],'b5':[8.30302e-05,0.000139883]}
  M1_xmm_measure = {'b1':[0.000127277,9.5751e-05],'b2':[0.0,2.52859e-05],'b3':[0.0,2.82096e-05],'b4':[0.0,3.49932e-05],'b5':[0.0,2.57288e-05]}
  M2_xmm_measure = {'b1':[0.000280392,0.000123981],'b2':[0.0,2.36201e-05],'b3':[6.64185e-05,8.11162e-05],'b4':[0.000206929,0.000116249],'b5':[0.0,2.64216e-05]}

  keys = ['b1','b2','b3']

  colours = ['r','g','b']

  i = 0
  for key in keys:
    result_pn = bands_xmm_lim(bands_xmm_temp[key][0],t,(PN_xmm_measure[key][0] + 3*PN_xmm_measure[key][1]))
    result_m1 = bands_xmm_lim(bands_xmm_temp[key][1],t,(M1_xmm_measure[key][0] + 3*M1_xmm_measure[key][1]))
    result_m2 = bands_xmm_lim(bands_xmm_temp[key][2],t,(M2_xmm_measure[key][0] + 3*M2_xmm_measure[key][1]))

#    plot(t,(10.0**result_pn)/dt,'r')
#    plot(t,(10.0**result_m1)/dt,'b')
#    plot(t,(10.0**result_m2)/dt,'g')

    result_pn = 10**result_pn
    result_m1 = 10**result_m1
    result_m2 = 10**result_m2

    s = 10
    result_ep = log10(((1/result_pn)**(s-1) + (1/result_m1)**(s-1) + (1/result_m2)**(s-1)) / ((1/result_pn)**s + (1/result_m1)**s + (1/result_m2)**s))

    print(result_ep)
    emplot.plot(t[t>(divide-0.0)],((10.0**result_ep)/dt)[t>(divide-0.0)],colours[i])
    emplot.text(t[-1], ((10.0**result_ep)/dt)[-1], key,**font)
    i += 1

  tight_layout()

  savefig(plot_name+'_DEM_plot.png')
  savefig(plot_name+'_DEM_plot.pdf')

  clf()

def load_chain(chain_name,maxn):

  fits=fitsio.FITS(chain_name)

  h = fitsio.read_header(chain_name,'MCMC')

  abund = h['--ABUND']

  chrom_height = int(h['--CHROM'])

  chroma_d = float(h['--chroma_d'])
  corona_d = float(h['--corona_d'])

  nh = h['--nh']

  data = fits[1][-1*maxn:]

  outgrid = []
  out_lp = []

  out_xray = []
  out_rosat = []
  out_XUV = []
  out_sanz = []
  out_high = []
  out_all = []
  out_beta = []

  out_xcounts = []

  out_flux = []
  out_mloss = []

  for line in data:
    outgrid += [line[0]]
    out_lp += [line[1]]
    out_xray += [line[2]]
    out_rosat += [line[3]]
    out_XUV += [line[4]]
    out_sanz += [line[5]]
    out_high += [line[6]]
    out_all += [line[7]]
    out_beta += [line[8]]
    out_xcounts += [line[9]]
    out_flux += [line[-2]]
    out_mloss += [line[-1]]

  outgrid = array(outgrid)
  out_lp = array(out_lp)

  out_xray = array(out_xray)
  out_rosat = array(out_rosat)
  out_XUV = array(out_XUV)
  out_sanz = array(out_sanz)
  out_high = array(out_high)
  out_all = array(out_all)
  out_beta = array(out_beta)

  out_xcounts = array(out_xcounts)

  out_flux = array(out_flux)
  out_mloss = array(out_mloss)

  out_fluxes = {'out_xcounts':out_xcounts,'out_xray':out_xray,'out_rosat':out_rosat,'out_XUV':out_XUV,'out_sanz':out_sanz,'out_high':out_high,'out_all':out_all,'out_beta':out_beta}

  return outgrid, out_lp, out_fluxes, out_xcounts, abund, chrom_height, chroma_d, corona_d, float(nh), out_flux, out_mloss

def plot_euv_spec(t,em,distance,semi,euv_grid,wvl_euv,rsun,msun,mp,rp,NH,plot_name,factor1):

  sum_euv = 0
  sum_lyman = 0
  sum_x = 0
  sum_uv = 0

  factor = pi*((rsun/semi))**2.0

  for i in range(0,len(t)):
    sum_euv += factor*euv_grid[i]*10**em[i]
    if t[i] >= 5.5:
      sum_x += factor*euv_grid[i]*10**em[i] 
    if t[i] < 5.5:
      sum_uv += factor*euv_grid[i]*10**em[i] 

  w, h = figaspect(0.5*3.0/4.0)

  fig = Figure(figsize=(w,h))

  ax = fig.add_subplot(111)

  with open(plot_name+'_spectra_out','w') as outfile:
    outfile.write('Wavelength (A),Flux (ergs/s/A/cm^2)\n')
    for i in range(0,len(wvl_euv)):
      outfile.write('{},{}\n'.format(wvl_euv[i],sum_euv[i]))

  ax.plot(wvl_euv,sum_euv,'k',lw=0.5)

  ax.set_ylabel('Flux (ergs/s/\AA/cm$^2$)', **font)
  ax.set_xlabel('Wavelength (\AA)', **font)
  fig.tight_layout()
  fig.savefig(plot_name+'_result_spectra.png')
  fig.savefig(plot_name+'_result_spectra.pdf')

  clf()

  plot(log10(wvl_euv),log10(sum_euv.copy()),'k',lw=0.5)
  ylabel('Log(Flux) (ergs/s/\AA/cm$^2$)', **font)
  xlabel('Log(Wavelength) (\AA)', **font)
  tight_layout()
  savefig(plot_name+'_log_result_spectra.png')
  savefig(plot_name+'_log_result_spectra.pdf')

  clf()

  plot(wvl_euv,factor1*sum_euv/factor,'k',lw=0.5)
  ylabel('Flux (ergs/s/$\AA$/cm$^2$)', **font)
  xlabel('Wavelength ($\AA$)', **font)
  tight_layout()

  savefig(plot_name+'_unabsorbed_result_spectra.png')
  savefig(plot_name+'_unabsorbed_result_spectra.pdf')
  a,b=ylim()

  clf()

  absorbed = absorption(NH,[wvl_euv,sum_euv.copy()])
  plot(wvl_euv,factor1*absorbed/factor,'r',lw=0.5)
  ylabel('Flux (ergs/s/$\AA$/cm$^2$)', **font)
  xlabel('Wavelength ($\AA$)', **font)
  tight_layout()
  ylim(a,b)

  savefig(plot_name+'_absorbed_result_spectra.png')
  savefig(plot_name+'_absorbed_result_spectra.pdf')

  clf()


  return



  tick_params(axis='both', which='major', labelsize=16)

  show()

  print('UV CONSTRAINED CONTRIBUTION')
  xflux(t,em,sum_uv,wvl_euv,rsun,msun,distance,mp,rp,semi)

  plot(wvl_euv,sum_uv,'g')

  ylabel('Flux (ergs/s/$\AA$/cm$^2$)', **font)
  xlabel('Wavelength ($\AA$)', **font)

  print('X-RAY CONSTRAINED CONTRIBUTION')
  xflux(t,em,sum_x,wvl_euv,rsun,msun,distance,mp,rp,semi)

  plot(wvl_euv,sum_x,'r')

  ylabel('Flux (ergs/s/$\AA$/cm$^2$)', **font)
  xlabel('Wavelength ($\AA$)', **font)

  show()

def plot_dem(fname,dt,divide):

# fname = raw_input("Enter Dictionary name to print: ")

  pkl_file = open(fname+'_out.pkl', 'rb')

  data = pickle.load(pkl_file)

  rs = 6.96e10
  r_star = 1.14*rs
  S = 4*pi*r_star**2

  pkl_file.close()

  w, h = figaspect(1.5*0.5*3.0/4.0)

  emfig = plt.figure(figsize=(w,h))

  ax1 = emfig.add_subplot(111)

  ax1.set_xlabel('log(T) K')
  ax1.set_yscale('log')
  ax1.set_ylabel('Differential Emission Measure [cm$^{-5}$ K$^{-1}$]')
#  ax1.set_yscale('log')
  ax1.set_xlim(3.8,8.2)
  ax1.set_ylim([1e19, 1e31])

#  yscale('log')
#  xlim(3.8,8.2)
#  ylim([1e19, 1e31])

  y = 10**(np.linspace(17,31))

  x = [divide]*len(y)

  for i in range(len(data)):
# the extra 2*pi*pi is to correct a mistake in the dem file definition, should really go back and correct it at the source
    em1 = 2*pi*pi*data[i]['emeasure']
    temp1 = log10(data[i]['temperature'])
    dt1 = (10**(temp1+0.05) - 10**(temp1-0.05))
    data[i]['emeasure'] = 2*pi*pi*data[i]['emeasure']/dt1
    data[i]['em_03'] = find_avg(data[i])
    ax1.plot(temp1, em1/dt1)
    #errorbar(data[i]['tmax'], data[i]['em_03'], xerr=0.15, fmt='rx')
    if(isinf(data[i]['em_03']) == False):
      ax1.plot(data[i]['tmax'], data[i]['em_03'], 'rs')
      name0 = str(data[i]['name'].rsplit('_')[0])
      name = name0.title()
      number = int(data[i]['name'].rsplit('_')[1])
      if (number == 2):
          name2 = 'II'
          name22 = 'ii'
      if (number == 3):
          name2 = 'III'
          name22 = 'iii'
      if (number == 4):
          name2 = 'IV'
          name22 = 'iv'
      if (number == 5):
          name2 = 'V'
          name22 = 'v'
      if (number == 6):
          name2 = 'VI'
          name22 = 'vi'
      if (number == 7):
          name2 = 'VII'
          name22 = 'vii'
      if (number == 8):
          name2 = 'VIII'
          name22 = 'viii'
      if (number == 9):
          name2 = 'IX'
          name22 = 'ix'
      if (number == 10):
          name2 = 'X'
          name22 = 'x'
      if (number == 11):
          name2 = 'XI'
          name22 = 'xi'
      if (number == 12):
          name2 = 'XII'
          name22 = 'xii'

      outplotname = '\mbox{'+name+'\,{\sc '+name22+'}}'

      if name+' '+name2 == 'C II':
        ax1.text(data[i]['tmax']+0.04, data[i]['em_03']*1.1, outplotname)
      elif name+' '+name2 == 'S II':
        ax1.text(data[i]['tmax']+0.05, data[i]['em_03']*1.1, outplotname)
      elif name+' '+name2 == 'N V':
        ax1.text(data[i]['tmax']-0.09, data[i]['em_03']*0.15, outplotname)
      elif name+' '+name2 == 'Si III':
        ax1.text(data[i]['tmax']-0.17, data[i]['em_03']*0.22, outplotname)
      elif name+' '+name2 == 'C IV':
        ax1.text(data[i]['tmax']-0.03, data[i]['em_03']*0.18, outplotname)
      elif name+' '+name2 == 'Si IV':
        ax1.text(data[i]['tmax']-0.09, data[i]['em_03']*0.09, outplotname)
      else:
        ax1.text(data[i]['tmax']+0.03, data[i]['em_03']*0.2, outplotname)
    else:
      print(data[i]['name'], data[i]['tmax'], 'em03 could not be calculated')
#    text(4.3,3e28, 'UV line constrained')
#    text(6.5,3e28, 'X-ray flux constrained')

#  plot(x,y,'--', color=(0,0,0))

  return ax1, emfig


main()
