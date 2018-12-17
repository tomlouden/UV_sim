import numpy as np
import chianti.core as ch
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def sum_spectrum(em,spec_grid):
  spec = np.array([0.0]*len(spec_grid[0]))

  for i in range(0,len(em)):
    spec += spec_grid[i]*(10**np.array(em[i]))

  return spec

def sum_interp_spectrum(em,spec_grid,exdens,density):

  print(density)

  f = interp1d(exdens,spec_grid,kind='linear',bounds_error=False,fill_value=1e-3,axis=0)
  interped = f(density)

  print(np.shape(interped))

  plt.plot(interped)
  plt.show()

  quit()
  spec = np.array([0.0]*len(spec_grid[0]))

  for i in range(0,len(em)):
    spec += spec_grid[i]*(10**np.array(em[i]))

  return spec

def calcstrengths(spec_dict,line_dict,t,density,factor=1.0):
  intensities = []
  group_waves = []
  group_names = []
  group_tag = []

  for i in range(0,len(line_dict)):

    new = []
    for d in density:
      new += [sum_match_lines(line_dict[i]['name'],line_dict[i]['wvl'],10**t,d,search_width = 1.0)]

    intensities += [new]
    group_names += [line_dict[i]['name']]
    group_tag += [line_dict[i]['mult_tag']]
    group_waves += [line_dict[i]['wvl']]
#  allwaves = np.array(allwaves)

  spec_dict['group_names'] = np.array(group_names)
  spec_dict['group_waves'] = np.array(group_waves)
  spec_dict['group_tags'] = np.array(group_tag)

  spec_dict['group_intensities'] = np.array(intensities)*factor
#  spec_dict['allwaves'] = allwaves

  spec_dict['temperatures'] = t
  spec_dict['densities'] = density

  return spec_dict

def sum_match_lines(ion_name,wvls,t,density,search_width = 1.0):

  ion_n = ch.ion(ion_name, temperature=t, eDensity=density)

  intensity = t.copy()*0.0
  print(wvls)
  for wvl in wvls:
    ion_n.intensity(wvlRange=[min(wvls)-search_width,max(wvls)+search_width])

    if wvl in(ion_n.Intensity['wvl']): 
      intensity_s = ion_n.Intensity['intensity'][:,(ion_n.Intensity['wvl'] == wvl)]
      intensity += np.array(intensity_s.T[0])
      print('match')
    else:
      print('Something is wrong - cannot find matching line! for {} {}'.format(ion_name,wvl))
      intensity += 0
  return intensity

def measured_lines(spec_dict,line_dict):

  fluxes = []
  errors = []
  for i in line_dict:
    ion_name = i['name']
    group_name = i['mult_tag']

    sub_ins = 0.0
    sub_errs = 0.0

    for j in range(0,len(spec_dict['name'])):
      if(group_name == spec_dict['mult_tag'][j]):
        sub_ins += spec_dict['flux'][j]
        sub_errs += spec_dict['error'][j]**2
    fluxes += [sub_ins]
    errors += [sub_errs**0.5]
  fluxes = np.array(fluxes)
  errors = np.array(errors)

  return fluxes,errors

def linestrengths(em,t,spec_dict,line_dict,density,abund_mod_dict):

  intensities = []

  calc_dens = np.log10(spec_dict['densities'])

  for i in range(0,len(line_dict)):
    total_ins = []
    ins = spec_dict['group_intensities'][i]

    for j in range(0,len(ins[0])):
      dj = np.log10(density[j])

      f = interp1d(calc_dens,ins[:,j],kind='linear',bounds_error=False,fill_value=1e-3)
      total_ins += [f(dj)]

      if len(abund_mod_dict.keys()) > 0:
        for kk in range(0,len(abund_mod_dict['names'])):
          if(abund_mod_dict['names'][kk] in spec_dict['group_names'][i]):
            total_ins[-1] = total_ins[-1]*abund_mod_dict['mod'][kk]
  #          print('modifying ',spec_dict['group_names'][i],'by',abund_mod_dict['mod'][kk])

#      plt.plot(calc_dens,np.log10(ins[:,j]))
#      plt.plot(dj,np.log10(f(dj)),'ro')
#      print(dj,f(dj))
#    plt.show()

    sum_ins = np.sum(np.array(total_ins)*10**np.array(em))
    intensities += [sum_ins]

  intensities = np.array(intensities)

  return intensities

def linestrength_diff(em,t,spec_dict,line_dict,flux,density,abund_mod_dict):
  calc = linestrengths(em,t,spec_dict,line_dict,density,abund_mod_dict)

  diff = flux - calc

#  print(calc,'CALC')
#  print(flux,'MEASURED')

  return diff

def linestrength_chi2(em,t,spec_dict,line_dict,flux,err,density,exclude_list,abund_mod_dict):
  diff = linestrength_diff(em,t,spec_dict,line_dict,flux,density,abund_mod_dict)


  group_names = spec_dict['group_names']

  output = 0.0
  for i in range(0,len(group_names)):
    if not (group_names[i] in exclude_list):
      output += (diff[i]/err[i])**2

  return output

def linestrength_lnprob(em,t,spec_dict,line_dict,flux,err,density,abund_mod_dict,exclude_list=[]):

  group_names = spec_dict['group_names']

  terr = 0.0
  for i in range(0,len(group_names)):
    if not (group_names[i] in exclude_list):
      terr += np.log(2.0*np.pi*(err[i]**2.0))

  chi2 = linestrength_chi2(em,t,spec_dict,line_dict,flux,err,density,exclude_list,abund_mod_dict)
  likelyhood = (-0.5)*( chi2 + terr)
  return likelyhood
