import pickle
import numpy as np
import chianti.core as ch
from UV_sim.linestrengths import *
import glob

def register_lines(fname,star_dist,star_radius, density):

	# think carefully about what density to use when identifying lines

	input_dict = ion_dict_from_file(fname)

	dict_list=[]

	for i in range(len(input_dict)):
		print(input_dict[i]['wavelength'])
		dict = match_line(input_dict[i]['name'], input_dict[i]['wavelength'], input_dict[i]['flux'],input_dict[i]['tmax'], density)		
		dict_list += [input_dict[i]]
		dict_list[i]['wvl'] = dict['wvl']

	output = open(fname+'_line_data.pkl', 'wb')

	pickle.dump(dict_list, output)	

def load_specdata(listfile):

  namelist = []
  wavelist = []
  fluxlist = []
  errorlist = []
  taglist = []

  for line in open(listfile):
    name = line.split('&')[0].replace(' ','')
    namelist += [name]

    wave = float(line.split('&')[1].replace(' ',''))
    wavelist += [wave]

    flux = float(line.split('&')[2].replace(' ',''))
    fluxlist += [flux]

    err = float(line.split('&')[3].replace(' ',''))
    errorlist += [err]

    tag = line.split('&')[4].replace(' ','').strip('\n')
    taglist += [tag]

  spec_dict ={'name':namelist,'wave':wavelist,'flux':fluxlist, 'error':errorlist, 'mult_tag':taglist}
  return spec_dict


def ion_dict_from_file(fname):

	dict_list=[]
	dict={}

	for line in open(fname):
		name = line.rsplit('&')[0].replace(' ','')
		wavelength=float(line.rsplit('&')[1].replace(' ',''))
		flux=float(line.rsplit('&')[2])
		f_err=float(line.rsplit('&')[3])**2


		mult_tag = line.rsplit('&')[4].strip('\n').replace(' ','')
		is_mult = (len(mult_tag) > 0 )

		tmax = find_tm(name)
		dict = {'name':name,'flux':flux,'err':f_err,'tmax':tmax,'wavelength':[wavelength],'mult_tag':mult_tag}
		repeat = False
		for x in dict_list:
			if ((x['name']==name) & is_mult & (x['mult_tag']==mult_tag)):
				x['flux'] +=flux
				x['err'] +=f_err
				x['wavelength'] +=[wavelength]
				repeat = True
		if (repeat==False):
		  dict_list += [dict]

#		print('name flux tmax')
#		print(name, flux, tmax)

	for x in dict_list:
		x['err'] = x['err']**0.5
	return dict_list

def find_tm(ion_name):

	print('FINDING TM')
	h = 6.626068e-34
	c = 299792458
	k = 1.3806503e-23

	ion_state = int(ion_name.rsplit('_')[1])

	t = 10**(3.8+ 0.01*np.arange(400))

	ion = ch.ion(ion_name, temperature=t)

	ioneq = ion.IoneqOne
	W =  h*c*ion.Elvlc['ecm'][ion_state - 1]*100
	gt = ioneq*(t**-0.5)*np.exp(-W/(k*t))

	tm = t[np.argmax(gt)]

	return np.log10(tm)

def match_line(ion_name, wvls, s_flux, tmax, density):

#	t = 10**((tmax - 0.3) + 0.001*np.arange(600))

	lowtemp = 4.0
	hightemp = 8.0
	npoints = 1000
	t = 10**(np.linspace(lowtemp,hightemp,npoints))

	ion_n = ch.ion(ion_name, temperature=t, eDensity=density)
	ion_n.gofnt(wvlRange=[min(wvls)-1,max(wvls)+1])

	return ion_n.Gofnt


def e_measure(spec_dict, density,t):

	ion_name = spec_dict['name']
	wvls = spec_dict['wvl']
	flux = spec_dict['flux']

#	tmax = spec_dict['tmax']
#	t = 10**((tmax - 0.6) + 0.002*np.arange(1200))

	intensity = sum_match_lines(ion_name,wvls,t,density,search_width = 1.0)

# not sure why I thought there was an extra 2pi in the definition?
#	emeasure = (flux) / (2*np.pi*intensity)
	emeasure = flux / intensity

	return emeasure
