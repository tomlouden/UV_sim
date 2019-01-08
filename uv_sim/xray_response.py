import pyfits as pf
import numpy as np
import os
import pickle
import glob
import math

import matplotlib.pyplot as plt

def clip_xmm_response(xmm_response,avg_en,min_en,max_en):

  dmask = [(avg_en > min_en) & (avg_en < max_en)]
  clipped_xmm = []
  for i in range(0,len(xmm_response)):
    clipped_xmm += [[]]
    for j in range(0,len(xmm_response[i])):
      clipped_xmm[i] += [xmm_response[i][j][dmask]]
  clipped_xmm = np.array(clipped_xmm)
  return clipped_xmm


def load_xmm_data(xmm_name,bkg_name):

  all_dat = pf.open(bkg_name)['SPECTRUM'].data
  bkg = {}
  for n in all_dat.names:
    bkg[n] = all_dat.field(n)

  all_dat = pf.open(xmm_name)['SPECTRUM'].data
  dat = {}
  for n in all_dat.names:
    dat[n] = all_dat.field(n)

  hdr = pf.open(xmm_name)[0].header

  tobs =hdr['DURATION']

  return dat,bkg,tobs

def create_responses(t,density,abund,NH,response_name,abspath):

  # load all desired simulated spectra

  x_grid = [0.0]*len(t)
  xmm_grid = [0.0]*len(t)

  for i in range(0,len(t)):
    # load x-ray models
    pkl_file_x = open('spectra_bin/'+abund+'/spectra_xray'+'_'+str(t[i])+'_'+str(np.log10(density[i]))+'.pkl', 'rb')
    x_grid[i] = np.array(list(pickle.load(pkl_file_x)))

  for i in range(0,len(t)):
    pkl_file = open('spectra_bin/'+abund+'/spectra_XMM_'+str(t[i])+'_'+str(np.log10(density[i]))+'nh'+str(NH)+'.pkl', 'rb')
    xmm_grid[i] = np.array(list(pickle.load(pkl_file)))

  xmm_grid = np.array(xmm_grid)
  output = open(abspath+'xmm_grid_nh_'+str(NH)+'.pkl', 'wb')
  pickle.dump(xmm_grid, output) 

def xflux(t,sum_spec,wvl,wvl_low=100,wvl_max=920):

    int_flux = 0.0
    for x in range(1,len(sum_spec)):
      if(math.isnan(sum_spec[x]) == False):
        dl = wvl[x] - wvl[x-1]
        if (wvl[x] > wvl_low) & (wvl[x] < wvl_max):
          int_flux += sum_spec[x]*dl

    return np.log10(int_flux)

def mass_loss_rate(t,em,sum_spec,wvl,rsun,msun,distance,m_p,r_p,semi,wvl_low=100,wvl_max=920):

    int_flux = 0.0
    eff_flux = 0.0

    ec = 1.60217657e-19
    h = 6.626068e-34
    c = 299792458

    efficiency = np.array([0.0]*len(sum_spec))

    for x in range(1,len(sum_spec)):
      if(math.isnan(sum_spec[x]) == False):
        dl = wvl[x] - wvl[x-1]
        p_en = h*c/(wvl[x]*1e-10)
        efficiency[x] = (p_en - 13.6*ec)/p_en
        if efficiency[x] < 0:
          efficiency[x] = 0
        efficiency[x] = 1.0
        if (wvl[x] > wvl_low) & (wvl[x] < wvl_max):
          int_flux += sum_spec[x]*dl
          eff_flux += sum_spec[x]*dl*efficiency[x]


    G =  6.67259e-8

    incident = int_flux*(distance/semi)**2

    eff_incident = eff_flux*(distance/semi)**2

    r_hill = semi*(m_p/(3.0*msun))**(1.0/3.0)

    eta = r_hill/(1.0*r_p)

    K_tide = 1.0 - (3.0/(2.0*eta)) + 1.0/(2.0*eta**3.0)
  
    print(K_tide)

    m_loss = (np.pi*eff_incident*(1.1*r_p)**3)/(G*m_p*K_tide)

    print(eff_incident)

    F = incident*np.pi*r_p**2

    out = 4*np.pi*int_flux*(distance)**2

    print('mean efficiency is :', mean(efficiency))
    print('total integrated flux is :', np.log10(out), 'ergs / s, between ',min(wvl),' and ',max(wvl))
    print('Using sanz forcadas technique, euv flux should be:', 4.8 + 0.86*np.log10(out), 'ergs / s, with total:')
    print('energy limited mass loss is :', m_loss/1e10, 'x 10^10 g / s')

    return np.log10(int_flux), m_loss

def generate_spectra(abund,t,density,wvl,response_name,tag,NH='none',proc=4,xray_distance_factor=1):

  import chianti.filters as chfilters
  import chianti.core as ch

#  note: it is best to cut off very low energy photons, as these seem to inspire odd behaviour in the 
# mos cameras.

  if not os.path.isdir('spectra_bin/'+abund):
    os.makedirs('spectra_bin/'+abund)

  cut =[0,25.0]

  c = 2.998e8
  h = 6.62607e-34
  Kev = 1.6e-16

#  here's where new models are calculated
  grid_control(tag,t,wvl,density,proc,abund)

  # load the response matrices...

  rsp = load_response(response_name)

  # fold the X-rays through response matrix

  x_grid = [0.0]*len(t)

  for i in range(0,len(t)):
    pkl_file_x = open('spectra_bin/'+abund+'/spectra_'+tag+'_'+str(t[i])+'_'+str(np.log10(density[i]))+'.pkl', 'rb')
    x_grid[i] = pickle.load(pkl_file_x)

  fold_xrays(rsp,x_grid,wvl,t,density,cut,abund,NH,xray_distance_factor)

def absorption(NH,spectrum,spec_type='Spectrum'):
  import astropy.constants as  const
  import astropy.units as u
#  absorb = np.loadtxt('abs.txt')

  absorb = np.loadtxt('NH_18.txt')

  wvl = np.array((const.h*const.c/((absorb[:,0]*u.keV).to(u.J))).to(u.angstrom))
  I = absorb[:,2]

  sigma = -np.log(I)/(10**18)

  output  = []

  for x in -sigma*(10**NH - 10**18):
    try:
      out = np.e**x
      output += [out]
    except(OverflowError):
      print('overflow during interstellar absorption!')
      output += [0]
  modify = I*np.array(output)

  oldspectrum = spectrum[1].copy()

  out_spectrum = []

  for i in range (0,len(spectrum[1][spec_type])):
    x = np.argmin(abs(wvl - spectrum[0][i]))
    out_spectrum += [spectrum[1][spec_type][i]*modify[x]]

  out_spectrum = np.array(out_spectrum)
#  plot(spectrum[0],oldspectrum)
#  plot(spectrum[0],spectrum[1])
#  show()

  return out_spectrum

def grid_control(tag,t,wvl,density,p,abund,clobber=False):

  import chianti.filters as chfilters
  import chianti.core as ch

  dir_name = 'spectra_bin/'+abund+'/'
  direc = glob.glob(dir_name+'*')

  emeasures = np.array(1e0)
  for i in range(0,len(t)):
    fname = dir_name+'spectra_'+str(tag)+'_'+ str(t[i])+'_'+str(np.log10(density[i]))+'.pkl'
    if((fname in direc) & (clobber == False)):
      print(fname+' already exists!')
    else:
      output = open(fname, 'wb')
      temperatures = np.array(10.0**t[i])
      print('calculating temp ',temperatures,' density ',np.log10(density[i]))
      verbose = False
#      s = (ch.spectrum(temperatures, density[i], wvl, filter = (chfilters.gaussian,.1), em=emeasures,verbose=verbose))
      s = (ch.mspectrum(temperatures, density[i], wvl, filter = (chfilters.gaussian,.1), em=emeasures, proc=p,verbose=verbose))
#      s = (ch.ipymspectrum(temperatures, density[i], wvl, filter = (chfilters.gaussian,.1), em=emeasures, doContinuum=1, minAbund=1.e-100,verbose=1))
#      pickle.dump(np.array(s.Spectrum['integrated']), output) 
#      plt.plot(s.Spectrum['intensity'])
 #     plt.plot(s.FreeBound['intensity'])
  #    plt.plot(s.FreeFree['intensity'])
   #   plt.show()

      pickle.dump({'Spectrum':s.Spectrum['intensity'],'FreeFree':s.FreeFree['intensity'],'FreeBound':s.FreeBound['intensity']}, output) 

def fold_xrays(rsp,xgrid,wvl,t,density,cut,abund,xray_distance_factor,NHfile='none',band='',clobber=False):


    dir_name = 'spectra_bin/'+abund+'/'
    direc = glob.glob(dir_name+'*')

    for i in range(0,len(t)):
      fname = 'spectra_bin/'+abund+'/spectra_XMM_'+ str(t[i]) +'_'+str(np.log10(density[i]))+band+'nh'+str(NHfile)+'.pkl'
      if((fname in direc) & (clobber == False)):
        print(fname+' already exists!')
      else:
        output = open( fname, 'wb')
        input_s = xgrid[i]
        if NHfile != 'none':
          input_s = absorption(NHfile,[wvl,input_s])  
        out_x, out_counts1 = fold_model(rsp,[wvl,input_s],'pn',cut,xray_distance_factor)
        fname = 'spectra_XMM_'+ str(t[i]) +'_'+str(np.log10(density[i]))+band+'nh'+str(NHfile)
        print('writing out')
        pickle.dump(out_x,output)

def fold_model(rsp,raw_spectrum,name,cut):

  t_response = np.array([0.0]*len(rsp['CHANNEL']))


  plt.plot(raw_spectrum)
  plt.show()

  spectrum = bin_spectrum(raw_spectrum,rsp,cut,xray_distance_factor)

  center_bin = (rsp['ENERG_LO'] + rsp['ENERG_HI'])/2.0
  center_bin2 = (rsp['E_MIN'] + rsp['E_MAX'])/2.0

  if(math.isnan(sum(spectrum))):
    return t_response, 0

  for i in range(0,len(rsp['MATRIX'])):

    start = 0
    for j in range(rsp['N_GRP'][i]):
      low = rsp['F_CHAN'][i][j]
      high = rsp['F_CHAN'][i][j]+rsp['N_CHAN'][i][j]
      redist = spectrum[i]*rsp['MATRIX'][i]*rsp['SPECRESP'][i]
      finish = start + (high-low)
      t_response[low:high] += redist[start:finish]
      start = finish
  
  n_counts = sum(t_response)

  return t_response, n_counts

def bin_spectrum(spectrum,rsp,cut):

# bin the number of photons per energy channel present in the input spectrum

  binned_spectrum = np.array([0.0]*len(rsp['ENERG_LO']))
  center_bin = (rsp['ENERG_LO'] + rsp['ENERG_HI'])/2.0

  min_en = min(rsp['ENERG_LO'])
  max_en = max(rsp['ENERG_HI'])

  h = 6.62606957e-34
  c = 3.0e8
  kev = 6.24150974e15

  for i in range(0,len(spectrum[0])):
    w = spectrum[0][i]*1e-10
    e_1_photon = h*c/w
    e_kev = e_1_photon*kev
    if(i==0):
        dl = abs(spectrum[0][i+1] - spectrum[0][i])
    elif(i==(len(spectrum[0]) - 1)):
        dl = abs(spectrum[0][i] - spectrum[0][i-1])
    else:
        dl = abs((spectrum[0][i+1])/2.0 - (spectrum[0][i-1])/2.0)
    if ((e_kev > min_en) and (e_kev < max_en) and (e_kev > cut[0]) and (e_kev < cut[1])):
      n_photons = dl*(1e-7)*spectrum[1][i]/e_1_photon
      index = np.argmin(abs(rsp['ENERG_LO']-e_kev))
      if ((e_kev > rsp['ENERG_LO'][index])):
        binned_spectrum[index] += n_photons
      else:
        binned_spectrum[index-1] += n_photons
  
  return binned_spectrum

def load_response(fname):

  f = pf.open(fname)
  arfname = fname.replace('.rmf','.arf')

  rmf1 = f[1].data
  rmf2 = f[2].data
  arf = pf.open(arfname)['SPECRESP'].data

  rsp = {}

  for n in rmf1.names:
    rsp[n] = rmf1.field(n)

  for n in rmf2.names:
    rsp[n] = rmf2.field(n)

  for n in arf.names:
    rsp[n] = arf.field(n)

  return rsp