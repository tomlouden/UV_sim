# -*- coding: utf-8 -*-
import chianti.core as ch
import matplotlib.pyplot as plt
import sys
import pickle
import numpy as np

def calc_em(fname,star_dist,star_radius, density):

	input_dict = ion_dict_from_file(fname)

	dict_list=[]

	for i in range(len(input_dict)):
		print(input_dict[i]['wavelength'])
		dict = e_measure(input_dict[i]['name'], input_dict[i]['wavelength'], input_dict[i]['flux'],input_dict[i]['tmax'],star_dist,star_radius, density)		
		dict_list += [dict]
		dict_list[i]['name'] = input_dict[i]['name']
		dict_list[i]['flux'] = input_dict[i]['flux']
		dict_list[i]['tmax'] = input_dict[i]['tmax']

	output = open(fname+'_out.pkl', 'wb')

	pickle.dump(dict_list, output)	

def create_dict():

	dict_list=[]
	dict={}
	while 1:
		e_name = input("Enter element name: ")
		i_name = input("Enter ionisation state: ")
		name = e_name + '_' + str(i_name)
		flux = input("Enter surface flux: ")
		tmax = find_tm(name)
		dict = {'name':name,'flux':flux,'tmax':tmax}
		dict_list += [dict]
		end = query_yes_no("Do you want to add another Ion?")
		if (end == False):
			break

	print(dict_list)

	fname = input("Enter Dictionary name to save: ")
	output = open(fname +'.pkl', 'wb')
	pickle.dump(dict_list, output)	

def ion_dict_from_file(fname):

	dict_list=[]
	dict={}

	for line in open(fname):
		name = line.rsplit('&')[0].replace(' ','')
		wavelength=float(line.rsplit('&')[1].replace(' ',''))
		flux=float(line.rsplit('&')[2])
		f_err=float(line.rsplit('&')[3])**2
		tmax = find_tm(name)
		dict = {'name':name,'flux':flux,'err':f_err,'tmax':tmax,'wavelength':[wavelength]}
		repeat = False
		for x in dict_list:
			if (x['name']==name):
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


def old_e_measure(ion_name, wvls, s_flux, tmax, star_dist, star_radius, density):


	rs = 6.96e10
	pc = 3.08567758e18
	t = 10**((tmax - 0.3) + 0.001*np.arange(600))
	ion_n = ch.ion(ion_name, temperature=t, eDensity=density)

	ion_n.gofnt(wvlRange=[min(wvls)-1,max(wvls)+1])
	star_dist = star_dist*pc
	star_radius = star_radius*rs

	flux = s_flux*(star_dist/star_radius)**2.0

	print('ANYONE THERE?!')
	quit()

	ion_n.Gofnt['emeasure'] = (flux/density) / (2*np.pi*ion_n.Gofnt['gofnt'])
	return ion_n.Gofnt

def find_avg(temp,emeasure,tmax):

	tlow = 10**(tmax - 0.15)
	thigh = 10**(tmax + 0.15)
	lowi = np.argmin(abs(temp - tlow))
	highi = np.argmin(abs(temp - thigh))

	sum_em=0
	for i in range(lowi,highi):
		sum_em += (temp[i+1] - temp[i])*((emeasure[i] + emeasure[i+1])/2.0)
	avg_em = sum_em/(thigh-tlow)
	return avg_em

def old_find_tm(ion_name):

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

def query_yes_no(question, default="yes"):
    """Ask a yes/no question via input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is one of "yes" or "no".
    """
    valid = {"yes":True,   "y":True,  "ye":True,
             "no":False,     "n":False}
    if default == None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "\
                             "(or 'y' or 'n').\n")
