# -*- coding: utf-8 -*-
import numpy as np
from fitsio import write as fwrite
from fitsio import read as fread
from fitsio import FITS as fFITS
import pickle
from response import *
from upper_limit import *
from UV_atmos import *
from astropy import constants as const
from astropy import units as u

data_dir = '/home/astro/phrmat/storage/XMM_HD209/analysis/'

def kev_to_wvl(kev):
  return (((const.h*const.c)/(kev*u.keV)).to('Angstrom')).value

def load_chain(chain_name,maxlength=1e6):

  #keeping this seperate for now in case I want to implement burning or something.

  with fFITS(chain_name,'r') as fits:
    length = fits['MCMC']._info['nrows']
    if length > maxlength:
      data_dict = fits['MCMC'][length-maxlength:length]
    else:
      data_dict = fits['MCMC'][0:length]
      
  return data_dict

def load_args(chain_name,frac,star_name,newdata=False,optimize=False,dim=4,abund='coronal',chrom=0,chroma_d=1e11,corona_d=1e8,NH=19.5):

  run_name = chain_name

  if star_name == 'hd209':
    distance = 49.63
    rsun = 1.14
    msun = 1.13
    listfile='full_line_list'
    if dim==3:
      x = [-295.977668849, -143.617235623, 13.052658363899999]
    if dim==4:
      x = [295.977668849, -143.617235623, 13.052658363899999, -0.39552862634500002]
    if dim==5:
      x = [295.977668849, -143.617235623, 13.052658363899999, -0.39552862634500002,1e-6]
#      x = [-5.47607140e+02,4.56543607e+02,-7.10319631e+01,4.85300835e+00,-1.22315896e-01]
    if dim==6:
      x=[-8.86260923e+03,7.63602136e+03,-1.40641129e+03,1.30510374e+02,-6.01574014e+00,1.09534764e-01]
      x=[-7.56268583e+04,6.55092379e+04,-1.22244891e+04,1.15335288e+03,-5.41699964e+01,1.00578835e+00]
      x=[ -7.50484719e+04,6.50234491e+04,-1.21370332e+04,1.14540071e+03,-5.38098582e+01,1.0]
    is_x_limit = True
    par_per_err = 0.03970
    # 2pi is from definitions in chianti, second pi is converting surface flux to intensity
    geometry = 2.0*np.pi


  if star_name == 'sun':
    distance = 1.0/206265.0
    rsun = 1.0
    msun = 1.0
    listfile='sun_lines'
    x = [295.977668849, -143.617235623, 13.052658363899999, -0.39552862634500002]
    x=[ -1.86352016e+03,1.37355920e+03,-1.96459731e+02,1.25405108e+01,-2.98780433e-01]
    x=[-2.73261246e+03,2.09884570e+03,-3.26727715e+02,2.44327726e+01,-8.43839484e-01,9.99194468e-03]

    # 2pi is from definitions in chianti, second pi is converting surface flux to intensity
    geometry = 2.0*np.pi

    is_x_limit = False
    par_per_err = 0


  distance = 49.63
  rsun = 1.148
  msun = 1.162
  semi = 0.04516
  rp = 1.380 #(southworth 2010)
  mp = 0.714

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

# the geometric factor changes surface intensity into flux from a sphere
  geometry = np.pi

  factor = geometry*((rsun/distance))**2.0

  lum = 4.0*np.pi*geometry*(rsun)**2.0

  t = 4.1 + 0.1*np.arange(40)

  dt = t.copy()

  for i in range(1,(len(t) - 1)):
    dt[i] = (10.0**t[i+1] - 10.0**t[i-1])/2.0

  dt[0] = (10**(t[0]+0.1) - 10**(t[0]-0.1))/2.0
  dt[-1] = (10**(t[-1]+0.1) - 10**(t[-1]-0.1))/2.0

  density=np.array([chroma_d]*chrom + [corona_d]*(len(t) - chrom))

  newdata = True
  if newdata == True:
    line_dict = pickle.load(open(star_name+'_full_out.pkl', 'rb'))
    spec_data = load_specdata(listfile)
    from UV_atmos import calcstrengths
    spec_dict = calcstrengths(spec_data,line_dict,t,density,factor=factor)
    with open('line_response_'+star_name+'_'+run_name +'.pkl', 'wb') as output:
      pickle.dump(spec_dict, output)	
    create_responses(t,density,abund,factor,rsun,msun,distance,mp,rp,semi,run_name,NH)
    
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

  X_ray_counts = 23.8065922617
  xray_error = np.sqrt(67.0)

  args=[spec_dict,t,xmm_temp,m_xmm_temp,bands_xmm_temp,X_ray_counts,xray_error,xray_temp,rosat_temp,XUV_temp,sanz_euv_temp,high_euv_temp,all_euv_temp,x,frac,is_x_limit,dt,par_per_err,factor,semi,mp,msun,rp]

  if optimize == True:
    import scipy.optimize as opt
    from UV_model import DEM_fit
    x, success = opt.leastsq(DEM_fit, x, args=(spec_dict,t,xmm_temp,m_xmm_temp,X_ray_counts,xray_error,is_x_limit), maxfev=int(1e6))

  return args, x

def old_create_responses(t,density,abund,factor,rsun,msun,distance,mp,rp,semi,run_name,NH):

  # load all desired simulated spectra

  em = [27.0]*len(t)

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
#    pkl_file1 = pickle.load(open(abund+'_abund_spectra_bin/spectra_'+str(1)+'_'+str(t[i])+'_'+str(np.log10(density[i]))+'.pkl', 'rb'))
#    pkl_file2 = pickle.load(open(abund+'_abund_spectra_bin/spectra_'+str(2)+'_'+str(t[i])+'_'+str(np.log10(density[i]))+'.pkl', 'rb'))
#    spec_grid1[i] = factor*np.array(list(pkl_file1))
#    spec_grid2[i] = factor*np.array(list(pkl_file2))
#    spec_grid[i] = factor*np.array(list(pkl_file1) + list(pkl_file2))

    # load x-ray models
    pkl_file_x = open(abund+'_abund_spectra_bin/spectra_'+'xray_b'+'_'+str(t[i])+'_'+str(np.log10(density[i]))+'.pkl', 'rb')
    x_grid[i] = np.array(list(pickle.load(pkl_file_x)))

    # load euv models
    pkl_file_euv = open(abund+'_abund_spectra_bin/spectra_'+'XUV_sanz_b'+'_'+str(t[i])+'_'+str(np.log10(density[i]))+'.pkl', 'rb')
    euv_grid[i] = np.array(list(pickle.load(pkl_file_euv)))

  # load the response matrices...
  rsp = load_response(data_dir + 'PN.rmf')
  rsp2 = load_response(data_dir + 'M1.rmf')
  rsp3 = load_response(data_dir + 'M2.rmf')

#  load the simulated XMM responses

  for i in range(0,len(t)):
    pkl_file = open(abund+'_abund_spectra_bin/spectra_XMM_'+str(t[i])+'_'+str(np.log10(density[i]))+'nh'+str(NH)+'.pkl', 'rb')
    xmm_grid[i] = np.array(list(pickle.load(pkl_file)))
    pkl_file.close()

  # what is the number of X-ray counts we need to reproduce?
#  X_ray_counts = maxcounts(data_dir,sigma)

  X_ray_counts = 23.8065922617

  font = {'family' : 'normal', 'weight': 'normal', 'size': 16}

  avg_en = (rsp['E_MIN'] + rsp['E_MAX'])/2.0
  avg_en2 = (rsp2['E_MIN'] + rsp2['E_MAX'])/2.0
  avg_en3 = (rsp3['E_MIN'] + rsp3['E_MAX'])/2.0

  range1 = avg_en
  range2 = avg_en2
  range3 = avg_en3

  min_en = 0.2

  max_en = 2


  tobs = np.array([26.57e3, 32.83e3, 32.82e3])

  xmm_temp = [[],[],[]]
  for i in range(0,len(t)):
    xmm_temp[0] += [tobs[0]*sum(xmm_grid[i][0][((avg_en > min_en) & (avg_en < max_en))])]
    xmm_temp[1] += [tobs[1]*sum(xmm_grid[i][1][((avg_en2 > min_en) & (avg_en2 < max_en))])]
    xmm_temp[2] += [tobs[2]*sum(xmm_grid[i][2][((avg_en3 > min_en) & (avg_en3 < max_en))])]
  xmm_temp[0] = np.array(xmm_temp[0])
  xmm_temp[1] = np.array(xmm_temp[1])
  xmm_temp[2] = np.array(xmm_temp[2])

  xray_weight = 1.0

  xray_error = np.sqrt(X_ray_counts)/xray_weight

  print np.log10(X_ray_counts/np.array(xmm_temp))

  output = open('xmm_response_'+run_name+'.pkl', 'wb')
  pickle.dump(xmm_temp, output)	

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

  for b in bands:

    min_en = bands[b][0]
    max_en = bands[b][1]

    for i in range(0,len(t)):
      pkl_file = open(abund+'_abund_spectra_bin/spectra_XMM_'+str(t[i])+'_'+str(np.log10(density[i]))+b+'nh'+str(NH)+'.pkl', 'rb')
      xmm_grid[i] = np.array(list(pickle.load(pkl_file)))
      pkl_file.close()

    xmm_temp = [[],[],[]]
    for i in range(0,len(t)):
      xmm_temp[0] += [sum(xmm_grid[i][0][((avg_en > min_en) & (avg_en < max_en))])]
      xmm_temp[1] += [sum(xmm_grid[i][1][((avg_en2 > min_en) & (avg_en2 < max_en))])]
      xmm_temp[2] += [sum(xmm_grid[i][2][((avg_en3 > min_en) & (avg_en3 < max_en))])]
    xmm_temp[0] = np.array(xmm_temp[0])
    xmm_temp[1] = np.array(xmm_temp[1])
    xmm_temp[2] = np.array(xmm_temp[2])

    print b, xmm_temp
    output = open('xmm_response_'+run_name+b+'.pkl', 'wb')
    pickle.dump(xmm_temp, output) 


  print 'done!'

  maxwvl = 920
  wvl_euv = np.linspace(5,maxwvl,(maxwvl-5)*10)

  wvl_x = np.linspace(0.1,250.0,10000)

  e1 = 0.2
  e2 = 0.5
  e3 = 1.0
  e4 = 2.0
  e5 = 4.5
  e6 = 12

  b1 = [kev_to_wvl(e2), kev_to_wvl(e1)]
  b2 = [kev_to_wvl(e3), kev_to_wvl(e2)]
  b3 = [kev_to_wvl(e4), kev_to_wvl(e3)]
  b4 = [kev_to_wvl(e5), kev_to_wvl(e4)]
  b5 = [kev_to_wvl(e6), kev_to_wvl(e5)]

  b1_temp = []
  b2_temp = []
  b3_temp = []
  b4_temp = []
  b5_temp = []

  for i in range(0,len(t)):
    b1_temp += [xflux(t,em,x_grid[i],wvl_x,rsun,msun,distance,mp,rp,semi,b1[0],b1[1])[0]]
    b2_temp += [xflux(t,em,x_grid[i],wvl_x,rsun,msun,distance,mp,rp,semi,b2[0],b2[1])[0]]
    b3_temp += [xflux(t,em,x_grid[i],wvl_x,rsun,msun,distance,mp,rp,semi,b3[0],b3[1])[0]]
    b4_temp += [xflux(t,em,x_grid[i],wvl_x,rsun,msun,distance,mp,rp,semi,b4[0],b4[1])[0]]
    b5_temp += [xflux(t,em,x_grid[i],wvl_x,rsun,msun,distance,mp,rp,semi,b5[0],b5[1])[0]]

  m_xmm_temp = {'b1_temp':b1_temp,'b2_temp':b2_temp,'b3_temp':b3_temp,'b4_temp':b4_temp,'b5_temp':b5_temp}

  output = open('m_xmm_response_'+run_name+'.pkl', 'wb')
  pickle.dump(m_xmm_temp, output) 

  xray_temp = []
  rosat_temp = []
  XUV_temp = []
  sanz_euv_temp = []
  high_euv_temp = []
  all_euv_temp = []

  for i in range(0,len(t)):
    xray_temp += [xflux(t,em,euv_grid[i],wvl_euv,rsun,msun,distance,mp,rp,semi,5,100)[0]]
    rosat_temp += [xflux(t,em,euv_grid[i],wvl_euv,rsun,msun,distance,mp,rp,semi,5,125)[0]]
    XUV_temp += [xflux(t,em,euv_grid[i],wvl_euv,rsun,msun,distance,mp,rp,semi,5,920)[0]]
    sanz_euv_temp += [xflux(t,em,euv_grid[i],wvl_euv,rsun,msun,distance,mp,rp,semi,100,920)[0]]
    high_euv_temp += [xflux(t,em,euv_grid[i],wvl_euv,rsun,msun,distance,mp,rp,semi,50,400)[0]]
    all_euv_temp += [xflux(t,em,euv_grid[i],wvl_euv,rsun,msun,distance,mp,rp,semi,50,900)[0]]

  output = open('xray_response_'+run_name+'.pkl', 'wb')
  pickle.dump(xray_temp, output) 

  output = open('rosat_response_'+run_name+'.pkl', 'wb')
  pickle.dump(rosat_temp, output)	

  output = open('XUV_response_'+run_name+'.pkl', 'wb')
  pickle.dump(XUV_temp, output)	

  output = open('sanz_euv_response_'+run_name+'.pkl', 'wb')
  pickle.dump(sanz_euv_temp, output) 

  output = open('high_euv_response_'+run_name+'.pkl', 'wb')
  pickle.dump(high_euv_temp, output) 

  output = open('all_euv_response_'+run_name+'.pkl', 'wb')
  pickle.dump(all_euv_temp, output) 
