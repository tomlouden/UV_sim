# -*- coding: utf-8 -*-
import pyfits as pf
import numpy as np

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

def old_fold_model(rsp,raw_spectrum,name,cut):

    t_response = array([0.0]*len(rsp['CHANNEL']))

    spectrum = bin_spectrum(raw_spectrum,rsp,cut)

#   spectrum = load_test('../XMM_HD209/analysis/truemodel.txt',rsp,cut)

    print('sum photons', sum(spectrum))

    sumofmatrix = 0.0

    center_bin = (rsp['ENERG_LO'] + rsp['ENERG_HI'])/2.0
    center_bin2 = (rsp['E_MIN'] + rsp['E_MAX'])/2.0

    show()

    if(math.isnan(sum(spectrum))):
        return t_response, 0

    for i in range(0,len(rsp['MATRIX'])):
        if (name == 'm'):
            m = rsp['F_CHAN'][i]
    #       redist = spectrum[i]*rsp['MATRIX'][i]
            redist = spectrum[i]*rsp['MATRIX'][i]*rsp['SPECRESP'][i]
        else:
            m = rsp['F_CHAN'][i]
    #       m = rsp['F_CHAN'][i][rsp['N_GRP'][i]-1]
            redist = spectrum[i]*rsp['MATRIX'][i]*rsp['SPECRESP'][i]

        t_response[(0+m):(len(redist) + m)] += redist
        sumofmatrix += sum(rsp['MATRIX'][i])
  

#   normalise = sumofmatrix/len(rsp['MATRIX'])

    print(name)

#   t_response = t_response/normalise

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

#   print xspec
    xs1 = xspec[:,0]
    xs = xspec[:,4]

#   plot(center_bin, spectrum/dx)
#   plot(center_bin2, t_response/dx2)
#   show()

    if(len(t_response)==len(xs)):
      dif = t_response/xs
#     print 'number of counts, xspec counts, difference', sum(t_response), sum(xs), sum(t_response) - sum(xs)
#     plot(dif)
#     show()
#     plot(xs)
#     plot(t_response)
#     show()

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
    
    print(min(spectrum[0]), max(spectrum[0]), min_en, max_en)
    return binned_spectrum

def load_response(fname):

    f = pf.open(fname)
    arfname = fname.rsplit('.')[0] + '.arf'

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