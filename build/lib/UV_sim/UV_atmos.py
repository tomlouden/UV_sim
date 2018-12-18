# -*- coding: utf-8 -*-
import scipy.optimize as opt
import pyfits as pf
import numpy as np
import pickle
import math
import ChiantiPy.core as ch
from UV_sim.UV_model import *
from matplotlib.pyplot import *

data_dir = '/home/astro/phrmat/storage/XMM_HD209/analysis/'

def simple_fit(p,spec_dict,spec_grid,x_grid,wvl,wvl_xray,t,plott,hub_spec,rsp,rsp2,rsp3,xmm_grid,upper_limit,factor,dt,rsun,msun,distance,m_p,r_p,semi,xray_weight,NH):

  em = chebyshev(p,t)

#  em=chebyshev(np.array([-5.47607140e+02,4.56543607e+02,-7.10319631e+01,4.85300835e+00,-1.22315896e-01]),t) + np.log10(dt)


  print(em - np.log10(dt))

  sum_spec = 0
  sum_x = 0
  sum_xmm = [0.0]*3

  avg_en = (rsp['E_MIN'] + rsp['E_MAX'])/2.0
  avg_en2 = (rsp2['E_MIN'] + rsp2['E_MAX'])/2.0
  avg_en3 = (rsp3['E_MIN'] + rsp3['E_MAX'])/2.0

  range1 = avg_en
  range2 = avg_en2
  range3 = avg_en3

  min_en = 0.2

  max_en = 2

  for i in range(0,len(t)):
#    sum_spec += spec_grid[i]*10**em[i]
    sum_x += factor*x_grid[i]*10**em[i]
    sum_xmm[0] += xmm_grid[i][0][((avg_en > min_en) & (avg_en < max_en))]*10**em[i]
    sum_xmm[1] += xmm_grid[i][1][((avg_en2 > min_en) & (avg_en2 < max_en))]*10**em[i]
    sum_xmm[2] += xmm_grid[i][2][((avg_en3 > min_en) & (avg_en3 < max_en))]*10**em[i]

  diff = []
  diff2 = []

  intensities = []

  for i in range(0,len(spec_dict['intensities'])):
    intensities += [0.0]
    for x in range(0,len(t)):
      intensities[i] += factor*spec_dict['intensities'][i][x]*10**em[x]

  s = 25
#  consolodate he2

  he_index = []
  total_he = 0.0

  for i in range(0,len(spec_dict['allwaves'])):
    if((spec_dict['allwaves'][i] > 1640.0) and (spec_dict['allwaves'][i] < 1641.0)):
      he_index += [i]
      total_he += intensities[i]

  for x in he_index:
    intensities[x] = total_he

  for i in range(0,len(spec_dict['wave'])):
#    match = np.argmin((spec_dict['wave'][i] - np.array(wvl))**2)
    match2 = np.argmin((spec_dict['wave'][i] - spec_dict['allwaves'])**2)
#    total_line = 0
#    hub_line = 0
#    for x in range(match-s,match+s):
#      total_line += (sum_spec[x]*(wvl[x]-wvl[x-1]))
#      hub_line += (hub_spec[x]*(wvl[x]-wvl[x-1]))
#    print spec_dict['name'][i]
#    plot(sum_spec[match-s:match+s])
#    show()
#    dict_line = ((206265*47.1)**2.0)*(intensities[match2][0])
    dict_line = (intensities[match2][0])

    diff += [spec_dict['flux'][i] - dict_line]
    diff2 += [sqrt(dict_line/spec_dict['flux'][i])]

  output = np.array(diff)/spec_dict['error']

  #to print out the total x ray flux
  x = xflux(t,em,sum_x,wvl_xray,rsun,msun,distance,m_p,r_p,semi)

  tobs = np.array([26.57e3, 32.83e3, 32.82e3])
#  tobs = [32.83e3, 32.83e3, 32.82e3]

#  tobs = tobs*2.0*1.3

  x_counts = (tobs[0]*sum(sum_xmm[0]) + tobs[1]*sum(sum_xmm[1]) + tobs[2]*sum(sum_xmm[2]))

  print('x-ray counts: ',x_counts, tobs[0]*sum(sum_xmm[0]),tobs[1]*sum(sum_xmm[1]),tobs[2]*sum(sum_xmm[2]))

  output = np.array(list(output) + [xray_weight*(x_counts  - upper_limit)/sqrt(upper_limit)])

  print('output is:', output)
  print('chisquare:', sum(output**2))
  print('reduced chisquare:', sum(output**2)/(len(output) - len(p) - 1.0))

  if(plott==True):

    for i in range(0,len(spec_dict['wave'])):
      color = 'k'
      if(spec_dict['name'][i].split('_')[0] == 'C'):
        color = 'g'
      if(spec_dict['name'][i].split('_')[0] == 'Si'):
        color = 'r'
      if(spec_dict['name'][i].split('_')[0] == 'Fe'):
        color = 'b'
      if(spec_dict['name'][i].split('_')[0] == 'O'):
        color = 'c'
      if(spec_dict['name'][i].split('_')[0] == 'He'):
        color = 'm'



      errorbar(spec_dict['wave'][i], diff2[i], yerr=spec_dict['error'][i]/spec_dict['flux'][i], fmt=color+'s')

    x_total = xflux(t,em,sum_x,wvl_xray,rsun,msun,distance,m_p,r_p,semi)

    ylabel('Fractional error')
    xlabel('Wavelegth ($\AA$)')

    show()

    plot(wvl_xray,sum_x*((distance/semi)**2)/factor)

#    for i in range(0,len(spec_dict['name')):
#      if(isinf(spec[i]['em_03']) == False):

    show()

    plot(avg_en[((avg_en > min_en) & (avg_en < max_en))],tobs[0]*sum_xmm[0])
    plot(avg_en2[((avg_en2 > min_en) & (avg_en2 < max_en))],tobs[1]*sum_xmm[1])
    plot(avg_en3[((avg_en3 > min_en) & (avg_en3 < max_en))],tobs[2]*sum_xmm[2])

    ylabel('Counts')
    xlabel('Energy [keV]')
    
    show()

  return output

def remove_continuum(tag):

#This function attempts to fit a blackbody to the spectrum,
#Then returns the spectrum with the blackbody subtracted.

  h = 6.626068e-34
  c = 299792458
  k = 1.38e-23


  in_dict = pickle.load(open('spectra_bin/co_add.pkl', 'rb'))
  x1 = [1e-18,5000,1,1,1]
  x, success = opt.leastsq(func, x1, args=(in_dict))
  w = in_dict['wave']

  print("Subtracted a blackbody of temperature ", x[1])


  z = (2*h*c**2 / (w*1e-10)**5)

  y = z*x[0]/((exp(h*c/(w*k*x[1]*1e-10))) - 1)
  flux = in_dict['flux'] - y

#  plot(in_dict['wave'],in_dict['flux'])
#  plot(w,y)
#  show()


#  plot(in_dict['wave'],flux)
#  show()
  in_dict['flux'] -= y


  output = open('spectra_bin/'+ 'flat_'+str(tag)+'_.pkl', 'wb')
  pickle.dump(in_dict, output)	


def func(x, in_dict):

  h = 6.626068e-34
  c = 299792458
  k = 1.38e-23


  w = in_dict['wave']
  flux = in_dict['flux']

  z = (2*h*c**2 / (w*1e-10)**5)

  y = z*x[0]/((exp(h*c/(w*k*x[1]*1e-10))) - 1)

  return (y-flux)

def add_hubble(file_name):

  # How many spectra are there?
  n=0
  for line in open(file_name):
    n+=1


  # Define a dictionary template
  with open(file_name) as f:
    template_spectra = read_spectra(str(f.readline()).rstrip('\n'))

#  create memory space
  s=[template_spectra]*n
  i = 0
  for line in open(file_name):
    s[i] = read_spectra(str(line).rstrip('\n'))
    i+=1

  wvlc = template_spectra['wvl']
  fluxc = zeros((len(wvlc),len(s)))
  errorc = zeros((len(wvlc),len(s)))

  diff = [0]*len(s)

  #align all spectra and place in data 'cube' fluxc

  for x in range(0,len(s)):

    compwave = s[x]['wvl']
    compflux = s[x]['flux']
    comperror = s[x]['error']

    shift = template_spectra['wvl'] - compwave

    A_shift = mean(shift)

    scale = template_spectra['wvl'][1] - template_spectra['wvl'][0]
    diff[x] = int(round(A_shift/scale))

    point1 = int(abs(diff[x]))
    point2 = int(len(wvlc) - point1)

    for i in range(point1,point2):
      fluxc[i][x] = 1.0*compflux[i+diff[x]]
      errorc[i][x] = 1.0*comperror[i+diff[x]]

  flux_out=zeros(len(wvlc))
  error_out=zeros(len(wvlc))

  for i in range(0,len(wvlc)):
    total_weight=0
    for x in range(0,len(s)):
      if(fluxc[i][x] != 0):
        weight = s[x]['exposure']
        flux_out[i] += weight*fluxc[i][x]
        error_out[i] += (weight*errorc[i][x])**2.0
        total_weight +=weight
    if(total_weight != 0):
      flux_out[i] = flux_out[i]/total_weight
      error_out[i] = sqrt(error_out[i])/total_weight

    print(total_weight, flux_out[i])

  flux_out = np.array(flux_out)

#  plot(wvlc, flux_out)
  show()
  out_dict = {}

  for i in range(0,len(wvlc)):
    valid1 = flux_out[i]
    if (valid1**2 > 0):
      valid1 = i
      break

  for n in range(0,len(wvlc)):
    i = len(wvlc) - n - 1
    valid2 = flux_out[i]
    if (valid2**2 > 0):
      valid2 = i
      break

 
  out_dict['flux'] = flux_out[valid1:valid2]
  out_dict['wave'] = wvlc[valid1:valid2]
  out_dict['error'] = (error_out[valid1:valid2])

  output = open('spectra_bin/'+ 'co_add' +'.pkl', 'wb')
  pickle.dump(out_dict, output)	

  return out_dict

def co_add_spec(p, template_spectra1, template_spectra2, error, spec_grid1,spec_grid2, wvl, n,t, plot_switch,distance,rsun):

  em = chebyshev(p,t)
  sum_spec1 = 0.0
  sum_spec2 = 0.0

  B = blackbody(wvl,6000,distance,rsun)

  for i in range(0,len(t)):
    sum_spec1 += spec_grid1[i]*10**em[i]
    sum_spec2 += spec_grid2[i]*10**em[i]

#    error[tag-1] = [1 if x<=0 else x for x in error[tag-1]]

  diff1 = list(np.array(template_spectra1['flux'])-sum_spec1)
  diff2 = list(np.array(template_spectra2['flux'])-sum_spec2)

#    output = (np.array(diff)/np.array(error[tag-1]))**2.0


  if (plot_switch == True):
    subplot(211)
    plot(template_spectra1['wave'], template_spectra1['flux'], 'b-')
    plot(template_spectra2['wave'], template_spectra2['flux'], 'b-')
    plot(template_spectra1['wave'], sum_spec1, 'r-')
    plot(template_spectra2['wave'], sum_spec2, 'r-')
    subplot(212, sharex=gca())
    plot(template_spectra1['wave'], diff1, 'b-')
    plot(template_spectra2['wave'], diff2, 'b-')
    show()

def old_grid_control(tag,t,wvl,density,p,abund):

  import ChiantiPy.filters as chfilters
  import ChiantiPy.core as ch

  emeasures = np.array(1e0)
  for i in range(0,len(t)):

    fname = 'spectra_'+str(tag)+'_'+ str(t[i])+'_'+str(np.log10(density[i]))
    output = open(abund+'_abund_spectra_bin/'+ fname +'.pkl', 'w')
    temperatures = np.array(10.0**t[i])
    print('calculating temp ',temperatures,' density ',np.log10(density[i]))
#    s = (ch.mspectrum(temperatures, density[i], wvl, filter = (chfilters.gaussian,.1), em=emeasures, proc=p))
    s = (ch.ipymspectrum(temperatures, density[i], wvl, filter = (chfilters.gaussian,.1), em=emeasures, doContinuum=1, minAbund=1.e-100,verbose=1))
    pickle.dump(np.array(s.Spectrum['integrated']), output)	
#    plot(wvl,s.Spectrum['integrated'])
#    show()
#    quit()


def read_spectra(n):

  FNAME = 'lb4m'+n+'_x1dsum.fits'

  hdulist = pf.open('HST_COS_HD209458/'+FNAME)
  tbdata = hdulist[1].data

  wavelength1 = tbdata[1][3]
  flux1 = tbdata[1][4]
  error1 = tbdata[1][5]

  wavelength2 = tbdata[0][3]
  flux2 = tbdata[0][4]
  error2 = tbdata[0][5]

  wavelength = np.array(list(wavelength1) + list(wavelength2))
  flux = list(flux1) + list(flux2)
  error = list(error1) + list(error2)

  spectra_file = {}
  spectra_file['flux'] = flux
  spectra_file['error'] = error
  spectra_file['wvl'] = wavelength
  spectra_file['exposure'] = tbdata[0][1]

  plot(wavelength, flux)

  return spectra_file

def absorption_old(wvl, input_spectra):

  correction = np.loadtxt('model.dat')

  wvl_model = correction[:,0]*10

  factor = correction[:,2] / max(correction[:,2])

  print(len(input_spectra))

  output_spectra=[0]*len(input_spectra)

  for i in range(0,len(input_spectra)):
    cor = factor[np.argmin((wvl[i] - wvl_model)**2)]
    output_spectra[i] = cor*input_spectra[i]

  return output_spectra

def old_xflux(t,em,sum_spec,wvl,rsun,msun,distance,m_p,r_p,semi,wvl_low=100,wvl_max=920):

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

def old_absorption(NH,spectrum):
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
      out = e**x
      output += [out]
    except(OverflowError):
      print('overflow during interstellar absorption!')
      output += [0]
  modify = I*np.array(output)

  oldspectrum = spectrum[1].copy()

  for i in range (0,len(spectrum[1])):
    x = np.argmin(abs(wvl - spectrum[0][i]))
    spectrum[1][i] = spectrum[1][i]*modify[x]

#  plot(spectrum[0],oldspectrum)
#  plot(spectrum[0],spectrum[1])
#  show()

  return spectrum[1]

def old_fold_xrays(rsp1,rsp2,rsp3,xgrid,wvl,t,density,cut,NHfile,abund,band=''):

    out_x=[0.0]*3
  
    for i in range(0,len(t)):
      if NHfile != 'none':
        xgrid[i] = absorption(NHfile,[wvl,xgrid[i]])  
      out_x[0], out_counts1 = fold_model(rsp1,[wvl,xgrid[i]],'pn',cut)
      out_x[1], out_counts2 = fold_model(rsp2,[wvl,xgrid[i]],'m',cut)
      out_x[2], out_counts3 = fold_model(rsp3,[wvl,xgrid[i]],'m',cut)
      fname = 'spectra_XMM_'+ str(t[i]) +'_'+str(np.log10(density[i]))+band+'nh'+str(NHfile)
      output = open(abund+'_abund_spectra_bin/'+ fname +'.pkl', 'wb')
      pickle.dump(out_x,output)

def load_test(fname,rsp,cut):

  h = 6.62606957e-34
  c = 299792458.0
  kev = 6.24150974e15

  test_data = np.loadtxt(fname)

  energy = test_data[:,0]
  wavelength = (1.0e10)*h*c/(energy/kev)
  
  dl = wavelength*0.0

  for i in range(1,(len(wavelength) - 1)):
    dl[i] = abs((wavelength[i+1] - wavelength[i-1])/2.0)

  dl[0] = abs((wavelength[1] - wavelength[0])/2.0)
  dl[len(wavelength)-1] =abs((wavelength[len(wavelength)-1] - wavelength[len(wavelength)-2])/2.0)


  deltae = test_data[:,1]
  photons = test_data[:,2]*deltae*2

  flux = (1e7)*photons*(energy/kev)/dl

  print('sum photons', sum(photons))
  print('sum flux', sum(flux*dl))
  raw_spectrum = array([wavelength,flux])

  spectrum = bin_spectrum(raw_spectrum,rsp,cut)
  return spectrum

def fold_model(rsp,raw_spectrum,name,cut):

  t_response = array([0.0]*len(rsp['CHANNEL']))


  spectrum = bin_spectrum(raw_spectrum,rsp,cut)

#  spectrum = load_test('../XMM_HD209/analysis/truemodel.txt',rsp,cut)

  print('sum photons', sum(spectrum))

  sumofmatrix = 0.0

  center_bin = (rsp['ENERG_LO'] + rsp['ENERG_HI'])/2.0
  center_bin2 = (rsp['E_MIN'] + rsp['E_MAX'])/2.0

  if(math.isnan(sum(spectrum))):
    return t_response, 0

  for i in range(0,len(rsp['MATRIX'])):
    if (name == 'm'):
      m = rsp['F_CHAN'][i]
      #redist = spectrum[i]*rsp['MATRIX'][i]
      redist = spectrum[i]*rsp['MATRIX'][i]*rsp['SPECRESP'][i]
    else:
      m = rsp['F_CHAN'][i]
      #m = rsp['F_CHAN'][i][rsp['N_GRP'][i]-1]
      redist = spectrum[i]*rsp['MATRIX'][i]*rsp['SPECRESP'][i]

    t_response[(0+m):(len(redist) + m)] += redist
    sumofmatrix += sum(rsp['MATRIX'][i])
  

#  normalise = sumofmatrix/len(rsp['MATRIX'])
#  t_response = t_response/normalise

  n_counts = sum(t_response)

  print('n_counts are:', n_counts)

  dx = center_bin*0.0
  dx2 = center_bin2*0.0

  for i in range(1,(len(center_bin) - 1)):
    dx[i] = abs((center_bin[i+1] - center_bin[i-1])/2.0)

  dx[0] = abs((center_bin[1] - center_bin[0])/2.0)
  dx[len(center_bin)-1] =abs((center_bin[len(center_bin)-1] - center_bin[len(center_bin)-2])/2.0)

  for i in range(1,(len(center_bin2) - 1)):
    dx2[i] = abs((center_bin2[i+1] - center_bin2[i-1])/2.0)

  dx2[0] = abs((center_bin2[1] - center_bin2[0]))
  dx2[len(center_bin2)-1] =abs((center_bin2[len(center_bin2)-1] - center_bin2[len(center_bin2)-2]))



  xspec = np.loadtxt('../XMM_HD209/analysis/outputPN.txt',delimiter=',')

#	print xspec
  xs1 = xspec[:,0]
  xs = xspec[:,4]

#	plot(center_bin, spectrum/dx)
#	plot(center_bin2, t_response/dx2)
#	show()

#  print name
#  if(len(t_response)==len(xs)):
#    dif = t_response/xs
#    print 'number of counts, xspec counts, difference', sum(t_response), sum(xs), sum(t_response) - sum(xs)
#	  plot(dif)
#	  show()
#	  plot(xs)
#	  plot(t_response)
#	  show()

  return t_response, n_counts

def bin_spectrum(spectrum,rsp,cut):

# bin the number of photons per energy channel present in the input spectrum

  binned_spectrum = array([0.0]*len(rsp['ENERG_LO']))
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
      index = argmin(abs(rsp['ENERG_LO']-e_kev))
      if ((e_kev > rsp['ENERG_LO'][index])):
        binned_spectrum[index] += n_photons
      else:
        binned_spectrum[index-1] += n_photons
	
  return binned_spectrum

def blackbody(wvl,T,distance,rsun):

  c = 2.998e8
  h = 6.62607e-34
  kb = 1.3806488e-23
  wvl = wvl*1.0e-10

  return (1.0/(4.0*pi))*(1e-4)*(1e7)*(1e-10)*((rsun/distance)**2)*(wvl**-5.0)*2.0*h*c*c/(exp(h*c/(wvl*kb*T)) - 1.0)
