# -*- coding: utf-8 -*-
from pylab import *
import scipy.optimize as opt
import pyfits as pf
import numpy as np
import pickle
from emeasure import *
from plot_output import *
import chianti.filters as chfilters
import chianti.core as ch
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

  import chianti.filters as chfilters
  import chianti.core as ch

  new_sim = True

#  note: it is best to cut off very low energy photons, as these seem to inspire odd behaviour in the 
# mos cameras.

  sigma = 0.9973

  distance = 49.63
  rsun = 1.14
  msun = 1.13
  semi = 0.045
  rp = 1.35
  mp = 0.71
  listfile='full_line_list'
  line_dict = pickle.load(open('hd209_full_out.pkl', 'rb'))
  dem_file = 'hd209_full'
  p = [ 332.0092476,-161.58417561,14.40813655,-0.43103837]
  geometry = 2.0*pi
  NH = 19.5

  proc = 16

  rs = 6.955e10
  pc = 3.08567758e18
  distance = distance*pc
  rsun  = rsun*rs
  au = 1.49597871e13
  r_j = 0.69911e10
  m_j = 1.8986e30
  msun = msun*1.99e33
  semi = au*0.045
  mp = m_j*mp
  rp = r_j*rp


#  the geometric factor which ensures the total fluxes are correct
  factor = geometry*((rsun/distance))**2.0
  print('factor is ', factor)

  cut =[0,25.0]

  t = 4.1 + 0.1*arange(40)
  em = [27.0]*len(t)

  chrom = 40
  density=array([1e11]*chrom + [1e8]*(len(t) - chrom))

  dt = t*0

  for i in range(1,(len(t) - 1)):
    dt[i] = (10.0**t[i+1] - 10.0**t[i-1])/2.0

  dt[0] = (10**(t[0]+0.1) - 10**(t[0]-0.1))/2.0
  dt[-1] = (10**(t[-1]+0.1) - 10**(t[-1]-0.1))/2.0

  x = em


  en = [0.15,2]

  c = 2.998e8
  h = 6.62607e-34
  Kev = 1.6e-16

  maxwvl = 920


# rosat band is 5 to 125 angstrom

  wvl_xray = np.linspace(0.1,250.0,10000)
  wvl_euv = np.linspace(5,maxwvl,(maxwvl-5)*10)
  wvl_ly = np.linspace(1200,1230,3000)
  wvl_solar = np.linspace(400,1100,7000)

#  if necessary, here's where new models are calculated
  new_sim = True
  if new_sim == True:
    print('about to start')
#    grid_control('1_b',t,wvl1,density,proc)
#    grid_control('2_b',t,wvl2,density,proc)
    grid_control('xray_b',t,wvl_xray,density,proc)
    grid_control('XUV_sanz_b',t,wvl_euv,density,proc)
#    grid_control('lyman_b',t,wvl_ly,density,proc)
#    grid_control('solar_',t,wvl_solar,density,proc)
    print('done')

  # load the response matrices...
  rsp = load_response(data_dir + 'PN.rmf')
  rsp2 = load_response(data_dir + 'M1.rmf')
  rsp3 = load_response(data_dir + 'M2.rmf')
#  rsp = load_response(data_dir + 'PN.rmf')
##  rsp2 = load_response(data_dir + 'epn_ff20_sY9_medium.rmf')
#  rsp3 = load_response(data_dir + 'epn_ff20_sY9_thin.rmf')

  # if necessary (just calculated new models?) fold the X-rays through response matrix

  x_grid = [0.0]*len(t)

  for i in range(0,len(t)):
    pkl_file_x = open('spectra_bin/spectra_'+'xray_b'+'_'+str(t[i])+'_'+str(log10(density[i]))+'.pkl', 'rb')
    x_grid[i] = array(list(pickle.load(pkl_file_x)))

  if new_sim == True:
    fold_xrays(rsp,rsp2,rsp3,x_grid,wvl_xray,t,density,cut,NH)

  quit()

  template_spectra1 = pickle.load(open('spectra_bin/flat_1_.pkl', 'rb'))
  wvl1 = template_spectra1['wave']
  template_spectra2 = pickle.load(open('spectra_bin/flat_2_.pkl', 'rb'))
  wvl2 = template_spectra2['wave']

  wvl = list(template_spectra1['wave']) + list(template_spectra2['wave'])
  hub_spec = list(template_spectra1['flux']) + list(template_spectra2['flux'])

#  add_hubble('flist1')
#  remove_continuum(1)

#  add_hubble('flist2')
#  remove_continuum(2)

#  create x-ray grid with even energy coverage, 2 to 20 Kev
# loading the actual spectral data to compare with later.

  in_dict = pickle.load(open('spectra_bin/flat_'+str(1)+'_.pkl', 'rb'))

  flux = [in_dict['flux']]
  error = [in_dict['error']]

  in_dict = pickle.load(open('spectra_bin/flat_'+str(2)+'_.pkl', 'rb'))

  flux += [in_dict['flux']]
  error += [in_dict['error']]

  n = 2


  # loading the measured line strengths 

  data = np.loadtxt(listfile, delimiter='&', usecols=[1,3,4])
  wavelist = data[:,0]
  fluxlist = data[:,1]
  errorlist = data[:,2]

  namelist = []
  for line in open(listfile):
    col = line.split('&')[0]    
    name = col.split('{')[1].rstrip('}') + '_' + col.split('{')[2].split('}')[0]
    namelist += [name]

  spec_dict ={'name':namelist,'wave':wavelist,'flux':fluxlist, 'error':errorlist}

#  calculate line strengths at grid temperatures.

  spec_dict = calcstrengths(spec_dict,line_dict,t,density,factor=factor)

  # load all desired simulated spectra

  spec_grid = [0.0]*len(t)
  spec_grid1 = [0.0]*len(t)
  spec_grid2 = [0.0]*len(t)
  x_grid = [0.0]*len(t)
  euv_grid = [0.0]*len(t)
  xmm_grid = [0.0]*len(t)
  lyman_grid = [0.0]*len(t)
  solar_grid = [0.0]*len(t)

  for i in range(0,len(t)):

    # load left and right spectra ranges
    pkl_file1 = pickle.load(open('spectra_bin/spectra_'+str(1)+'_'+str(t[i])+'_'+str(log10(density[i]))+'.pkl', 'rb'))
    pkl_file2 = pickle.load(open('spectra_bin/spectra_'+str(2)+'_'+str(t[i])+'_'+str(log10(density[i]))+'.pkl', 'rb'))
    spec_grid1[i] = factor*array(list(pkl_file1))
    spec_grid2[i] = factor*array(list(pkl_file2))
    spec_grid[i] = factor*array(list(pkl_file1) + list(pkl_file2))

    # load x-ray models
    pkl_file_x = open('spectra_bin/spectra_'+'xray_b'+'_'+str(t[i])+'_'+str(log10(density[i]))+'.pkl', 'rb')
    x_grid[i] = array(list(pickle.load(pkl_file_x)))

    # load euv models
    pkl_file_euv = open('spectra_bin/spectra_'+'XUV_sanz_b'+'_'+str(t[i])+'_'+str(log10(density[i]))+'.pkl', 'rb')
    euv_grid[i] = array(list(pickle.load(pkl_file_euv)))

#    pkl_file5 = open('spectra_bin/spectra_'+'lyman'+'_'+str(t[i])+'_'+str(log10(density[i]))+'.pkl', 'rb')
#    pkl_file_s = open('spectra_bin/spectra_'+'solar'+'_'+str(t[i])+'_'+str(log10(density[i]))+'.pkl', 'rb')
#    lyman_grid[i] = factor*array(list(pickle.load(pkl_file5)))
#    solar_grid[i] = factor*array(list(pickle.load(pkl_file_s)))


#  load the simulated XMM responses

  for i in range(0,len(t)):
    pkl_file = open('spectra_bin/spectra_XMM_'+str(t[i])+'_'+str(log10(density[i]))+'.pkl', 'rb')
    xmm_grid[i] = array(list(pickle.load(pkl_file)))
    pkl_file.close()
  show()

  # what is the number of X-ray counts we need to reproduce?
  X_ray_counts = maxcounts(data_dir,sigma)
  show()

  X_ray_counts = 23.8065922617

#  plot(template_spectra1['wave'], template_spectra1['flux']*1e15, 'b')
#  plot(template_spectra2['wave'], template_spectra2['flux']*1e15, 'b')

  font = {'family' : 'normal', 'weight': 'normal', 'size': 16}

#  text(1176,2,'C III', **font)
#  text(1190,7.5,'Si III', **font)
#  text(1335,5.5,'C II', **font)
#  text(1640,4.3,'He II')
#  text(1393,4.0,'Si IV', **font)
#  text(1403,2.6,'Si IV', **font)
#  text(1550,5.6,'C IV', **font)
#  text(1218,1.8,'O V', **font)
#  text(1243,0.6,'N V', **font)

#  ylabel('Flux (ergs/s/$\AA$/cm$^2$ x 10$^{15}$)', **font)
#  xlabel('Wavelength ($\AA$)', **font)


#  show()


####################################################################################
# EMCEE goes in here?

  avg_en = (rsp['E_MIN'] + rsp['E_MAX'])/2.0
  avg_en2 = (rsp2['E_MIN'] + rsp2['E_MAX'])/2.0
  avg_en3 = (rsp3['E_MIN'] + rsp3['E_MAX'])/2.0

  range1 = avg_en
  range2 = avg_en2
  range3 = avg_en3

  min_en = 0.2

  max_en = 2


  tobs = array([26.57e3, 32.83e3, 32.82e3])

  xmm_temp = [[],[],[]]

  for i in range(0,len(t)):
    xmm_temp[0] += [tobs[0]*sum(xmm_grid[i][0][((avg_en > min_en) & (avg_en < max_en))])]
    xmm_temp[1] += [tobs[1]*sum(xmm_grid[i][1][((avg_en2 > min_en) & (avg_en2 < max_en))])]
    xmm_temp[2] += [tobs[2]*sum(xmm_grid[i][2][((avg_en3 > min_en) & (avg_en3 < max_en))])]

  xmm_temp[0] = array(xmm_temp[0])
  xmm_temp[1] = array(xmm_temp[1])
  xmm_temp[2] = array(xmm_temp[2])

  xray_weight = 1.0

  xray_error = sqrt(X_ray_counts)/xray_weight

  print(log10(X_ray_counts/array(xmm_temp)))

  xlim = x_limit(t,density,X_ray_counts,30e3,factor)

  print(xlim)

  output = open('xmm_response_b.pkl', 'wb')
  pickle.dump(xmm_temp, output)	

  print('done!')

  euv_temp = []
  for i in range(0,len(t)):
    euv_temp += [xflux(t,em,euv_grid[i],wvl_euv,rsun,msun,distance,mp,rp,semi)[0]]
  output = open('euv_response.pkl', 'wb')
  pickle.dump(euv_temp, output)	

  rosat_temp = []
  for i in range(0,len(t)):
    rosat_temp += [xflux(t,em,euv_grid[i][(wvl_euv > 5) & (wvl_euv < 125)],wvl_euv[(wvl_euv > 5) & (wvl_euv < 125)],rsun,msun,distance,mp,rp,semi)[0]]
  output = open('rosat_response.pkl', 'wb')
  pickle.dump(rosat_temp, output)	

  quit()

  euv_temp = [-2.4412733434477918, -1.5113963408242335, -1.1982188428076976, -0.80678620073649276, -0.42446261564949389, -0.22877274468142694, 0.044011573739097705, 0.41982958826128952, 0.71808227688566484, 0.83138939203773965, 0.91472133123298127, 1.0061812957512657, 1.0579719104831651, 1.0398715393608293, 0.83205466818917473, 0.75714972471974384, 0.85090026857599355, 0.88580939549409077, 0.9111428378028843, 0.93483427514656781, 0.91395171076746384, 0.85673292290300951, 0.70312011578637335, 0.40893271337522275, 0.10688502361313636, -0.1420441931023351, -0.34438428338228, -0.49902803706323468, -0.52836213135432986, -0.43897772552993181, -0.40607466124232477, -0.53857987498218818, -0.73975059265270326, -0.91734872694217096, -1.0568588571722368, -1.1689893587479929, -1.2641530829646177, -1.3470702285920748, -1.4177510500218542, -1.4753579031541553]

  # get inital best guess...

#  x, success = opt.leastsq(DEM_fit, p, args=(spec_dict,t,xmm_temp,X_ray_counts,xray_error), maxfev=1e4)

#  x = array([ 322.69325698, -159.33158776, 14.64728809, -0.44964712])
  x = array([ 320.87024996,-161.86944331,14.49503734,-0.41539493])

#  x = array([ 270,-180,16,-0.35])

  quit()

####################################################################################
#-----------------------------------------------------------------------------------
# here is where the fit actually takes place, after loading all the data

#  x, success = opt.leastsq(simple_fit, p, args=(spec_dict,spec_grid,x_grid,wvl,wvl_xray,t,False,hub_spec,rsp,rsp2,rsp3,xmm_grid,X_ray_counts,factor,dt,rsun,msun,distance,mp,rp,semi,xray_weight,NH), maxfev=1e3)

#-----------------------------------------------------------------------------------

  # the final fitted emission measure
  em = chebyshev(x,t)

#  em = array([24.6, 24, 23.5, 22.9, 22.4, 22.0, 21.6, 21.3, 21.0, 20.8, 20.7,20.6,20.6,20.7,20.7,20.8,20.9,20.9,21.0,20.9,20.8,20.6,20.2,19.6,18.9])

# recorded solar emission measure
  em2=chebyshev(array([-5.47607140e+02,4.56543607e+02,-7.10319631e+01,4.85300835e+00,-1.22315896e-01]),t) + log10(dt)


# plot the weighted visual spectra
  co_add_spec(x, template_spectra1, template_spectra2, error, spec_grid1, spec_grid2, array(wvl), n,t, True,distance,rsun) 

# plot the emission measures
  simple_fit(x, spec_dict,spec_grid,x_grid,wvl,wvl_xray,t,True,hub_spec,rsp,rsp2,rsp3,xmm_grid,X_ray_counts,factor,dt,rsun,msun,distance,mp,rp,semi,xray_weight,NH)

  #plot the approximate em values and emission loci

  emplot = plt.subplot(111)

  emplot.set_yscale('log')

  emplot.set_ylim([1e15, 1e32])

  ylabel('Differential Emission Measure', **font)
  xlabel('log(T/K)', **font)

  plot(t,(10.0**xlim)/dt)
  plot(t,(10.0**em)/dt,'g')

  #overplot solar data?
#  plot(t,(10.0**em2)/dt)

  plot_dem(dem_file,dt)

  print(list(x))
  
  emeasure = em

  sum_euv = 0
  sum_lyman = 0
  sum_x = 0
  sum_uv = 0


  temp_cont = [0]*len(t)

  incident = (distance/semi)**2

  for i in range(0,len(t)):
    sum_euv += euv_grid[i]*10**em[i]
    if t[i] >= 5.5:
      sum_x += euv_grid[i]*10**em[i] 
    if t[i] < 5.5:
      sum_uv += euv_grid[i]*10**em[i] 
#    sum_lyman += lyman_grid[i]*10**em[i] 
    temp_cont[i] = xflux(t,em,euv_grid[i]*10**0,wvl_euv,rsun,msun,distance,mp,rp,semi)

  print('THIS IS THE EUV FLUX')

  print('between', min(wvl_euv), min(wvl_euv))

  xflux(t,em,sum_euv,wvl_euv,rsun,msun,distance,mp,rp,semi)

  print('THIS IS THE EUV FLUX')

  B = blackbody(wvl_euv,5777.0,distance,rsun)

  plot(wvl_euv,sum_euv)

  absorbed = absorption(NH,[wvl_euv,sum_euv])

  plot(wvl_euv,absorbed)

  ylabel('Flux (ergs/s/$\AA$/cm$^2$)', **font)
  xlabel('Wavelength ($\AA$)', **font)

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


  plot(t,temp_cont/max(temp_cont))

  ylabel('Relative level of ionising flux')

  xlabel('Temperature (log(T/K))')

  print(temp_cont)
  show()

main()
