import numpy as np

def load_txt_spectrum(fname):
	dat = np.loadtxt(fname)
	return {'wvl':dat[:,0],'flux_at_1au':dat[:,1],'flux_at_earth':dat[:,2],'error_at_1au':dat[:,3],'error_at_earth':dat[:,4]}

def sum_spectrum(wvl,flux,error,start=0,finish=1e6,step=0.002):

	total = 0.0
	t_error = 0.0

	for i in range(0,len(wvl)):
		if((wvl[i] > start) & (wvl[i] < finish)):
			total += flux[i]*step
			t_error += (error[i]*step)**2
	t_error = t_error**0.5

	return total, t_error