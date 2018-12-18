# -*- coding: utf-8 -*-
from pylab import *
from UV_sim.emeasure import *
import matplotlib.pyplot as plt
import pprint, pickle
from UV_sim.make_dicts import *
from UV_sim.linestrengths import * 
from UV_sim.xray_response import * 
from scipy.interpolate import interp1d

def compare_xmm_response(d1,d2,t,en,mask,clip_data,blur_data):


	h = 6.62606957e-34
	c = 3.0e8
	kev = 6.24150974e15

	wvl = np.linspace(0.1,250.0,10000)
	avg_en = (kev*h*c)/(wvl*1e-10)

	min_en =0.3
	max_en = 10
	mask2= (avg_en > min_en) & (avg_en < max_en)

	en2 = avg_en[mask2]

	NH = 'none'
	abund='photo'
	for i in range(23,len(t)):
		p_xmm1 = open('spectra_bin/'+abund+'/spectra_XMM_'+str(t[i])+'_'+str(np.log10(d1))+'nh'+str(NH)+'.pkl', 'rb')
		p_xmm2 = open('spectra_bin/'+abund+'/spectra_XMM_'+str(t[i])+'_'+str(np.log10(d2))+'nh'+str(NH)+'.pkl', 'rb')
		p_xray1 = open('spectra_bin/'+abund+'/spectra_xray_'+str(t[i])+'_'+str(np.log10(d1))+'.pkl', 'rb')
		p_xray2 = open('spectra_bin/'+abund+'/spectra_xray_'+str(t[i])+'_'+str(np.log10(d2))+'.pkl', 'rb')
		xmm1 = np.array(list(pickle.load(p_xmm1)))
		xmm2 = np.array(list(pickle.load(p_xmm2)))
		xray1 = np.array(list(pickle.load(p_xray1)))
		xray2 = np.array(list(pickle.load(p_xray2)))

		plt.title(t[i])

		plt.plot(en,xmm1[mask],label=np.log10(d1))
		plt.plot(en,xmm2[mask],label=np.log10(d2))


#		plt.plot(en2,np.max(xmm1[mask])*xray1[mask2]/np.max(xray1[mask2]),label=np.log10(d2))
#		plt.plot(en2,np.max(xmm1[mask])*xray2[mask2]/np.max(xray1[mask2]),label=np.log10(d2))


		plt.plot(en,np.max(xmm1[mask])*clip_data/np.max(clip_data),'rx',label=np.log10(d2))
		plt.plot(en,np.max(xmm1[mask])*blur_data/np.max(blur_data),'r',label=np.log10(d2))

		plt.axvline(0.3,c='k',ls='--')

		plt.legend()
		plt.show()




def plot_ratio_response(n1,n2,spec_dict):

	intensity_grid1 = spec_dict['group_intensities'][spec_dict['group_tags'] == n1][0]
	intensity_grid2 = spec_dict['group_intensities'][spec_dict['group_tags'] == n2][0]

	ratio_grid = intensity_grid1/intensity_grid2

	for i in range(0,len(ratio_grid)):
		for j in range(0,len(ratio_grid[i])):
			if not np.isfinite(ratio_grid[i][j]):
				ratio_grid[i][j] = 0.0

	fig, ax = plt.subplots()

	mapp = ax.imshow(ratio_grid,cmap='viridis',aspect='auto',origin='lower')

	t = spec_dict['temperatures']
	d = spec_dict['densities']

	jump = 5
	xtick_pos = np.arange(4,len(t))
	ax.set_xticks(xtick_pos[0::jump]+0.5)
	ax.set_xticklabels(t[4:][0::jump])

	jump = 2
	ytick_pos = np.arange(0,len(d))

	logd = np.array([np.log10(di) for di in d])

	ax.set_yticks(ytick_pos[0::jump]+0.5)
	ax.set_yticklabels(logd[0::jump])

	ax.set_ylabel('Log eDensity [cm$^{-3}$]')
	ax.set_xlabel('Log Temperature [K]')

	ax.set_title(pretty_name(n1)+' / '+pretty_name(n2))

	cb = plt.colorbar(mapp)

	cb.set_label('Intensity ratio')

def plot_xmm_response(xmm_response,t,response_name):

	rsp = load_response(response_name)
	avg_en = (rsp['E_MIN'] + rsp['E_MAX'])/2.0

	h = 6.62606957e-34
	c = 3.0e8
	kev = 6.24150974e15

	wvl = np.array((kev*h*c/avg_en)/1e-10)

	fig, ax = plt.subplots()

	dmask = [(avg_en > 0.15) & (avg_en < 15)]
	clipped_xmm = []
	for i in range(0,len(xmm_response)):
		clipped_xmm += [xmm_response[i][dmask]]
	clipped_xmm = np.array(clipped_xmm)

	clipped_xmm  = clipped_xmm / np.max(clipped_xmm)


	mapp = ax.pcolor(wvl[np.array(dmask)[0]],t,clipped_xmm,cmap='viridis')
#	mapp = ax.pcolor(avg_en[np.array(dmask)[0]],t,clipped_xmm,cmap='viridis')

#	t = spec_dict['temperatures']
#	d = spec_dict['densities']
#	jump = 5
#	xtick_pos = np.arange(4,len(t))
#	ax.set_xticks(xtick_pos[0::jump]+0.5)
#	ax.set_xticklabels(t[4:][0::jump])

#	jump = 2
#	ytick_pos = np.arange(0,len(d))

#	logd = np.array([np.log10(di) for di in d])

#	ax.set_yticks(ytick_pos[0::jump]+0.5)
#	ax.set_yticklabels(logd[0::jump])

	ax.set_ylabel('Log Temperature [K]')
	ax.set_xlabel('Wavelength [\AA]')

	ax.set_title('XMM response')
	ax.set_xlim(1,80)
	ax.set_ylim(4.1,8)



#	ax.set_xscale('log')

	cb = plt.colorbar(mapp)

	cb.set_label('Intensity')

def plot_spectrum_response(raw_spectrum_response,t,wvl):

	h = 6.62606957e-34
	c = 3.0e8
	kev = 6.24150974e15

	fig, ax = plt.subplots()

	mask = [wvl > 100]

	wvl = wvl[mask]

	spectrum_response = []
	for i in range(0,len(raw_spectrum_response)):
		spectrum_response += [raw_spectrum_response[i][mask]]
	spectrum_response = np.array(spectrum_response)


	min_non_zero = np.min(spectrum_response[spectrum_response > 0])
	min_non_zero = 1e-50

	spectrum_response = spectrum_response + min_non_zero

	spectrum_response = spectrum_response / np.max(spectrum_response)

	spectrum_response = np.log10(spectrum_response)


	mapp = ax.pcolor(np.log10(wvl),t,spectrum_response,cmap='inferno')

#	mapp = ax.pcolor(avg_en[np.array(dmask)[0]],t,clipped_xmm,cmap='viridis')

#	t = spec_dict['temperatures']
#	d = spec_dict['densities']
#	jump = 5
#	xtick_pos = np.arange(4,len(t))
#	ax.set_xticks(xtick_pos[0::jump]+0.5)
#	ax.set_xticklabels(t[4:][0::jump])

#	jump = 2
#	ytick_pos = np.arange(0,len(d))

#	logd = np.array([np.log10(di) for di in d])

#	ax.set_yticks(ytick_pos[0::jump]+0.5)
#	ax.set_yticklabels(logd[0::jump])

	ax.set_ylabel('Log Temperature [K]')
	ax.set_xlabel('Log Wavelength [\AA]')

	ax.set_title('XUV response')
	ax.set_xlim(np.min(np.log10(wvl)),np.max(np.log10(wvl)))
	ax.set_ylim(4.1,8)



#	ax.set_xscale('log')

	cb = plt.colorbar(mapp)

	cb.set_label('Intensity')

def plot_line_response(name,spec_dict):

	intensity_grid = spec_dict['group_intensities'][spec_dict['group_tags'] == name][0]

	intensity_grid = intensity_grid/np.max(intensity_grid)

	fig, ax = plt.subplots()

	mapp = ax.imshow(intensity_grid,cmap='viridis',aspect='auto',origin='lower')

	t = spec_dict['temperatures']
	d = spec_dict['densities']

	jump = 5
	xtick_pos = np.arange(4,len(t))
	ax.set_xticks(xtick_pos[0::jump]+0.5)
	ax.set_xticklabels(t[4:][0::jump])

	jump = 2
	ytick_pos = np.arange(0,len(d))

	logd = np.array([np.log10(di) for di in d])

	ax.set_yticks(ytick_pos[0::jump]+0.5)
	ax.set_yticklabels(logd[0::jump])

	ax.set_ylabel('Log eDensity [cm$^{-3}$]')
	ax.set_xlabel('Log Temperature [K]')

	ax.set_title(pretty_name(name))

	cb = plt.colorbar(mapp)

	cb.set_label('Intensity')

def plot_dem(fnames,data,datas,density_func=False,density_args=[],density_kwargs={},label_lines=False,overplot=[],factor=1.0,savename='',exclude_list=[],single_temp=None,limits=[]):

	direc = glob.glob(savename+'*')

	plt.clf()
	plt.close()

	em_data = []

	i = 0
	for fname in fnames:
		lname = fname.split('/')[-1]
		if savename+lname+'_em_data.pkl' not in direc:
			print("Calculating new emission measure locci in "+savename+lname+'_em_data.pkl')
			this_em = calc_dem_locci(datas[i],density_func,density_args,density_kwargs)
			em_data += this_em
			output = open(savename+lname+'_em_data.pkl', 'wb')
			pickle.dump(this_em, output)
		else:
			print('Using existing em locci in '+savename+lname+'_em_data.pkl')
			pkl_file = open(savename+lname+'_em_data.pkl', 'rb')
			em_data += pickle.load(pkl_file)
			pkl_file.close()
		print(np.shape(em_data))
		i += 1

	ax1 = plt.subplot(111)

	ax1.set_xlabel('log(T) K')
	ax1.set_yscale('log')
	ax1.set_ylabel('Differential Emission Measure [cm$^{-5}$ K$^{-1}$]')
#	ax1.set_ylim([1e18,1e29])
#	ax1.set_xlim([3.8,8.2])

	y = 10**(np.linspace(17,30))

	x = [5.67]*len(y)

	for ov in overplot:
		plt.plot(ov[0],factor*10**ov[1],lw=2,c='r')
		t = overplot[0][0]
		dt = t*0
		for i in range(1,(len(t) - 1)):
		  dt[i] = (10.0**t[i+1] - 10.0**t[i-1])/2.0
		dt[0] = (10**4.2 - 10**4.0)/2.0
		dt[39] = (10**8.1 - 10**7.9)/2.0
	if single_temp != None:
		dt = dt*0.0 + 1.0

#	ax1.set_ylim([1e18,10*np.max(factor*10**overplot[0][1])])
	ax1.set_ylim([1e18,1e30])
	ax1.set_xlim([4.0,8.0])


	if len(limits) > 0:
		print(len(limits[0]))
		print(len(limits[1]))
		print(len(t))
		plt.fill_between(t,factor*10**limits[0],factor*10**limits[1],color='grey')


#	plt.show()		

	for i in range(len(data)):

		if data[i]['name'] not in exclude_list:
			tmax = data[i]['tmax']

			em1 = factor*em_data[i]['em']
			temp1 = em_data[i]['temp']
			em_03 = factor*em_data[i]['em_03']

			new_tmax = emiss_func(temp1,em1,overplot[0][0],dt*factor*10**overplot[0][1])

			dt1 = (10**(temp1+0.05) - 10**(temp1-0.05))

			if single_temp != None:
				dt1 = dt1*0.0 + 1.0

			tmax = new_tmax

			new_mask = [abs(tmax-temp1) < 0.5]

			ax1.plot(temp1[new_mask],(em1/dt1)[new_mask])

			lim_flux = (em1/dt1)[argmin(abs(temp1 - tmax))]

			if label_lines == True:
				xpos = temp1[new_mask][np.isfinite(temp1[new_mask])][-1]
				ypos = (em1/dt1)[new_mask][np.isfinite((em1/dt1)[new_mask])][-1]
				print(xpos,ypos)
				ax1.text(xpos, ypos,str(data[i]['wvl']))
			if(isinf(em_03) == False):
				ax1.plot(tmax, lim_flux*1.5, 'rs')
				new_name = pretty_name(data[i]['name'])
				ax1.text(tmax+0.03, lim_flux*1.5, new_name)
			else:
				print(data[i]['name'], data[i]['tmax'], 'em03 could not be calculated')

		ax1.text(4.3,3e28, 'UV line constrained')
		ax1.text(6.5,3e28, 'X-ray flux constrained')
	ax1.plot(x,y,'--', color=(0,0,0))


	plt.savefig(savename+'dem.pdf',set_bbox_inches='tight')
	plt.savefig(savename+'dem.png',set_bbox_inches='tight')

#	show()

	plt.close()

def emiss_func(i_t,i_em,t,em):

	f = interp1d(t,np.log10(em),kind='linear',bounds_error=False,fill_value=1e-3)

	rats =[]
	rats2 =[]
	for i in range(0,len(i_t)):
		rats += [em[argmin(abs(10**i_t[i]-10**t))]/i_em[i]]
		rats2 += [(10**f(i_t[i]))/i_em[i]]

	rats = np.array(rats)
	rats2 = np.array(rats2)

#	plt.plot(i_t,np.log10(i_em))
#	plt.plot(t,np.log10(em))
#	plt.plot(t,np.log10(em),'bo')
#	plt.plot(i_t,np.log10(rats))
#	plt.plot(i_t,rats2)
#	plt.show()
	tmax = i_t[rats2>0][np.argmax(rats2[rats2>0])]
	return tmax


def calc_dem_locci(data,density_func,args,kwargs,	lowtemp = 4.0,hightemp = 8.0,npoints = 100):

	dem_dict = []

	for i in range(len(data)):
		sub_dict = {}

		temp1 = np.linspace(lowtemp,hightemp,npoints)
		density = 10**(density_func(temp1,*args,**kwargs))

		em1 = e_measure(data[i], density,10**temp1)

		tmax = data[i]['tmax']
#		temp1 = np.log10(10**((tmax - 0.6) + 0.002*np.arange(1200)))


	# the assumption here is that the used bins are 0.1 dex wide - probably shouldn't be hardcoded?
		dt1 = (10**(temp1+0.05) - 10**(temp1-0.05))

		sub_dict['em_03'] = find_avg(10**temp1,em1/dt1,data[i]['tmax'])
		sub_dict['em'] = em1
		sub_dict['temp'] = temp1

		dem_dict += [sub_dict]
	return dem_dict

def pretty_name(base_name):

	print(base_name)

	element = str(base_name.rsplit('_')[0])
	element = element.title()
	number = int(base_name.rsplit('_')[1])

	numeral = roman_numerals(number)
	name = element+' '+numeral
	return name

def roman_numerals(number):
	if (number == 1):
	    numeral = 'I'
	if (number == 2):
	    numeral = 'II'
	if (number == 3):
	    numeral = 'III'
	if (number == 4):
	    numeral = 'IV'
	if (number == 5):
	    numeral = 'V'
	if (number == 6):
	    numeral = 'VI'
	if (number == 7):
	    numeral = 'VII'
	if (number == 8):
	    numeral = 'VIII'
	if (number == 9):
	    numeral = 'IX'
	if (number == 10):
	    numeral = 'X'
	if (number == 11):
	    numeral = 'XI'
	if (number == 12):
	    numeral = 'XII'
	return numeral
