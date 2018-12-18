# -*- coding: utf-8 -*-
import numpy as np
import math

def return_fluxes(em,t,xmm_temp,m_xmm_temp,bands_xmm_temp,xray_temp,rosat_temp,XUV_temp,sanz_euv_temp,high_euv_temp,all_euv_temp,factor):

  d_err = 1.0

  m_xmm_out = {'b1_temp':0.0,'b2_temp':0.0,'b3_temp':0.0,'b4_temp':0.0,'b5_temp':0.0}

  for flux in m_xmm_temp:
    for x in range(0,len(t)):
      m_xmm_out[flux] += factor*10.0**(m_xmm_temp[flux][x] + em[x])

  x_counts = 0.0
  for x in range(0,len(t)):
    x_counts += d_err*(xmm_temp[0][x] + xmm_temp[1][x] + xmm_temp[2][x])*10**em[x]

  new_x_counts = {'b1':np.array([0.0]*3),'b2':np.array([0.0]*3),'b3':np.array([0.0]*3),'b4':np.array([0.0]*3),'b5':np.array([0.0]*3)}

  for x in range(0,len(t)):
    for k in new_x_counts:
      new_x_counts[k] += np.array([bands_xmm_temp[k][0][x],bands_xmm_temp[k][1][x],bands_xmm_temp[k][2][x]])*10**em[x]

  xray_flux = 0.0 
  rosat_flux = 0.0 
  XUV_flux = 0.0 
  sanz_euv_flux = 0.0 
  high_euv_flux = 0.0 
  all_euv_flux = 0.0 

  for x in range(0,len(t)):
    xray_flux += 10.0**(xray_temp[x] + em[x])
    rosat_flux += 10.0**(rosat_temp[x] + em[x])
    XUV_flux += 10.0**(XUV_temp[x] + em[x])
    sanz_euv_flux += 10.0**(sanz_euv_temp[x] + em[x])
    high_euv_flux += 10.0**(high_euv_temp[x] + em[x])
    all_euv_flux += 10.0**(all_euv_temp[x] + em[x])

  beta = high_euv_flux / all_euv_flux

  xray_flux = np.log10(xray_flux)
  rosat_flux = np.log10(rosat_flux)
  XUV_flux = np.log10(XUV_flux)
  sanz_euv_flux = np.log10(sanz_euv_flux)
  high_euv_flux = np.log10(high_euv_flux)
  all_euv_flux = np.log10(all_euv_flux)

  return x_counts, m_xmm_out, new_x_counts, xray_flux, rosat_flux, XUV_flux, sanz_euv_flux, high_euv_flux,all_euv_flux, beta

def lnprob(p,spec_dict,t,xmm_temp,m_xmm_temp,bands_xmm_temp,upper_limit,xray_error,xray_temp,rosat_temp,XUV_temp,sanz_euv_temp,high_euv_temp,all_euv_temp,x0,frac,is_x_limit,dt,par_per_err,factor,semi,m_p,msun,r_p):

# to include the error on the distance.
#  d_err = (1.0/(np.random.normal(1.0,par_per_err)))**2.0

  d_err = 1.0

  em = chebyshev(p,t)

  dem = np.log10((10.0**em)/dt)

  x_counts, m_xmm_out, new_x_counts, xray_flux, rosat_flux, XUV_flux, sanz_euv_flux, high_euv_flux,all_euv_flux, beta = return_fluxes(em,t,xmm_temp,m_xmm_temp,bands_xmm_temp,xray_temp,rosat_temp,XUV_temp,sanz_euv_temp,high_euv_temp,all_euv_temp,factor)

  m_xmm_measure = {'b1_temp':[7.57223e-16,2.20843e-16],'b2_temp':[0.0,9.02837e-17],'b3_temp':[9.23528e-18,6.75021e-17],'b4_temp':[2.00557e-16,3.05533e-16],'b5_temp':[2.98078e-16,1.10395e-15]}

  PN_xmm_measure = {'b1':[0.000540662,0.000206315],'b2':[0.0,0.000156369],'b3':[0.0,4.52034e-05],'b4':[3.31264e-05,7.89352e-05
],'b5':[8.30302e-05,0.000139883]}
  M1_xmm_measure = {'b1':[0.000127277,9.5751e-05],'b2':[0.0,2.52859e-05],'b3':[0.0,2.82096e-05],'b4':[0.0,3.49932e-05],'b5':[0.0,2.57288e-05]}
  M2_xmm_measure = {'b1':[0.000280392,0.000123981],'b2':[0.0,2.36201e-05],'b3':[6.64185e-05,8.11162e-05],'b4':[0.000206929,0.000116249],'b5':[0.0,2.64216e-05]}


  new_x_counts_residuals = []
  new_x_counts_errors = []

  for flux in PN_xmm_measure:
    new_x_counts_residuals += [PN_xmm_measure[flux][0] - new_x_counts[flux][0]]
    new_x_counts_errors += [PN_xmm_measure[flux][1]]

  for flux in M1_xmm_measure:
    new_x_counts_residuals += [M1_xmm_measure[flux][0] - new_x_counts[flux][1]]
    new_x_counts_errors += [M1_xmm_measure[flux][1]]

  for flux in M2_xmm_measure:
    new_x_counts_residuals += [M2_xmm_measure[flux][0] - new_x_counts[flux][2]]
    new_x_counts_errors += [M2_xmm_measure[flux][1]]

  m_xmm_residuals = []
  m_xmm_errors = []

  for flux in m_xmm_measure:
    m_xmm_residuals += [m_xmm_measure[flux][0] - m_xmm_out[flux]]
    m_xmm_errors += [m_xmm_measure[flux][1]]

#  print m_xmm_out

  residuals = DEM_fit(p,spec_dict,t,xmm_temp,upper_limit,xray_error,is_x_limit)

#  residuals = np.append(residuals[:-1],np.array(m_xmm_residuals)/np.array(m_xmm_errors))
  residuals = np.append(residuals[:-1],np.array(new_x_counts_residuals)/np.array(new_x_counts_errors))

  is_new_x_limit = True
  if is_x_limit:
    errors = np.array(list(spec_dict['error']) + [xray_error])
  if is_new_x_limit:
#    errors = np.append(spec_dict['error'],m_xmm_errors)
    errors = np.append(spec_dict['error'],new_x_counts_errors)
  else:
    errors = np.array(list(spec_dict['error']))

  likelyhood = (-0.5)*sum( residuals**2.0 + np.log(2.0*np.pi*(errors**2.0)))
  lp = lnprior(p,sanz_euv_flux,rosat_flux,x_counts,x0,frac,dem)

  lnoutput = lp + likelyhood

  PN_counts = [new_x_counts['b1'][0],new_x_counts['b2'][0],new_x_counts['b3'][0],new_x_counts['b4'][0],new_x_counts['b5'][0]]
  M1_counts = [new_x_counts['b1'][1],new_x_counts['b2'][1],new_x_counts['b3'][1],new_x_counts['b4'][1],new_x_counts['b5'][1]]
  M2_counts = [new_x_counts['b1'][2],new_x_counts['b2'][2],new_x_counts['b3'][2],new_x_counts['b4'][2],new_x_counts['b5'][2]]

  XUV_incident = XUV_flux + np.log10((1.0/(4.0*np.pi))*(1/semi)**2)

  mloss = calc_mass_loss(10**XUV_incident,semi,m_p,msun,r_p)

  blob = np.array([xray_flux, rosat_flux, XUV_flux, sanz_euv_flux, high_euv_flux, all_euv_flux, beta, x_counts]+PN_counts+M1_counts+M2_counts+[XUV_incident]+[mloss])

  if math.isnan(lnoutput):
      return -np.inf, blob

  if not np.isfinite(lp):
      return -np.inf, blob
  return lnoutput, blob

def calc_mass_loss(incident,semi,m_p,msun,r_p):

    G =  6.67259e-8

    r_hill = semi*(m_p/(3.0*msun))**(1.0/3.0)

    eta = r_hill/(1.0*r_p)

    K_tide = 1.0 - (3.0/(2.0*eta)) + 1.0/(2.0*eta**3.0)
  
    beta = 1.1

    m_loss = (np.pi*incident*(beta**2)*(r_p)**3)/(G*m_p*K_tide)

    return np.log10(m_loss)

def lnprior(theta,euv_flux,rosat_flux,x_counts,x0,frac,dem):

#    allowed fractional difference from best fit  
    mi = []
    ma = []

    for i in range(0,len(x0)):
      lims = np.array([(100.0 - frac)*x0[i]/100.0,(100.0 + frac)*x0[i]/100.0])
      mi += [min(lims)]
      ma += [max(lims)]

    return 0

#    if all([(mi[i] < theta[i] < ma[i]) for i in range(0,len(x0))]) and (0 < euv_flux < np.inf) and (0 < x_counts < np.inf) and (26.0 < rosat_flux < 28.0) and (dem[-1] < 18) and all(dem[0:20] > 18) and all(dem[0] > dem[10:]):
#    if all([(mi[i] < theta[i] < ma[i]) for i in range(0,len(x0))]) and (0 < euv_flux < np.inf) and (0 < x_counts < np.inf) and (0 < rosat_flux < np.inf) and (dem[-1] < 18) and all(dem[0] > dem[10:]) and all(dem[0:10] > 18):
    if all([(mi[i] < theta[i] < ma[i]) for i in range(0,len(x0))]) and (0 < euv_flux < np.inf) and (0 < x_counts < np.inf) and (0 < rosat_flux < np.inf):
        return 0.0
    return -np.inf

def chebyshev(a,x):
 
  T = [x]*len(a)

  T[0] = np.array([1.0]*len(x))
  T[1] = x

  for i in range(2,len(a)):
    T[i] = 2.0*x*T[i-1] - T[i-2]

  output = np.array(T[0])*0.0
  for i in range(0,len(T)):
    output += T[i]*a[i]

  return output

def new_DEM_fit(em,spec_dict,t):

  diff = []

  intensities = []

  for i in range(0,len(spec_dict['intensities'])):
    intensities += [0.0]
    for x in range(0,len(t)):
      intensities[i] += spec_dict['intensities'][i][x]*10**em[x]

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
    match = np.argmin((spec_dict['wave'][i] - spec_dict['allwaves'])**2)
    dict_line = (intensities[match][0])
    diff += [dict_line - spec_dict['flux'][i]]

  output = np.array(diff) + spec_dict['flux']

  return output

def DEM_fit(p,spec_dict,t,xmm_temp,upper_limit,xray_error,is_x_limit):

  em = chebyshev(p,t)

  diff = []

  intensities = []

  for i in range(0,len(spec_dict['intensities'])):
    intensities += [0.0]
    for x in range(0,len(t)):
      intensities[i] += spec_dict['intensities'][i][x]*10**em[x]

  x_counts = 0.0
  for z in range(0,len(t)):
    x_counts += (xmm_temp[0][z] + xmm_temp[1][z] + xmm_temp[2][z])*10**em[z]

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
    match = np.argmin((spec_dict['wave'][i] - spec_dict['allwaves'])**2)
    dict_line = (intensities[match][0])
    diff += [dict_line - spec_dict['flux'][i]]

  output = np.array(diff)/spec_dict['error']

  if is_x_limit:
    output = (np.array(list(output) + [(x_counts  - upper_limit)/xray_error]))

  return output