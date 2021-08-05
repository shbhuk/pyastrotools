from scipy import stats
import astropy
from astropy.modeling.blackbody import blackbody_lambda, blackbody_nu
import warnings
from astropy.utils.exceptions import AstropyDeprecationWarning
import numpy as np
import math
from astropy import constants as ac
from astropy import units as u

c = ac.c.value #[m/s]

filt_mean_wavelengths = {"U": 3650, #angstroms
                         "B": 4330,
                         "V": 5500,
                         "R": 7000,
                         "I": 8994,
                         "J": 12500,
                         "H": 16458,
                         "K": 21900}

filt_mean_wavelengths = {x:filt_mean_wavelengths[x]*u.Angstrom for x in filt_mean_wavelengths.keys()}

filt_bandpass         = {"U": 640.4, #angstroms
                         "B": 959.2,
                         "V": 893.1,
                         "R": 1591.0,
                         "I": 1495.1,
                         "J": 100.,
                         "H": 200.,
                         "K": 4100 }

filt_bandpass = {x:filt_bandpass[x]*u.Angstrom for x in filt_bandpass.keys()}



vega_zero_f_lam =   {"U": 426.6e-11,
                         "B": 682e-11,
                         "V": 380.2e-11,
                         "R": 173.8e-11,
                         "I": 83.2e-11,
                         "J": 33.1e-11,
                         "H": 11.38e-11,
                         "K": 3.981e-11  }  #ergs/cm2/s/Angstrom

#http://www.astronomy.ohio-state.edu/~martini/usefuldata.html
#http://web.ipac.caltech.edu/staff/fmasci/home/astro_refs/magsystems.pdf
#https://www.gemini.edu/sciops/instruments/midir-resources/imaging-calibrations/fluxmagnitude-conversion

def Vegamag_to_flux(mag,key='V'):
    '''
    Return flux in Watts/m^2/Hz and Watts/m^2/Angstrom,
    Input:
        m - Apparent magnitude
        wl - Wavelength [Angstrom]. Specified using u.
        key - 'U','B','V','R','I','J','K' (Johnson)
    Output:
        f_nu - Watts/m^2/Hz
        f_lam - Watts/m^2/Angstrom
    '''

    flambda = (10**(-0.4*mag))*vega_zero_f_lam[key] *  (1e-7)/(1e-4) *(u.W/u.m/u.m/u.Angstrom)
    fnu=(flambda*filt_mean_wavelengths[key]**2/ac.c).to(u.W/u.m/u.m/u.Hz)

    return fnu,flambda

def ABmag_to_flux(mag,wl,key='V'):
    '''
    Return flux in Watts/m^2/Hz and Watts/m^2/Angstrom,
    Input:
        m - Apparent magnitude
        wl - Wavelength [Angstrom]. Specified using u.
        key - 'AB'
    Output:
        f_nu - Watts/m^2/Hz
        f_lam - Watts/m^2/Angstrom
    '''

    wl = wl
    fnu=(10**(-0.4*(mag + 48.6))) * (1e-7)/(1e-4) * (u.W/u.m/u.m/u.Hz)# Convert to SI
    flambda = (fnu*ac.c/(wl*wl)).to(u.W/u.m/u.m/u.Angstrom)
    return fnu,flambda


def Vegamag_to_bb_flambda(mag, key, T, wavelengths = None, bandpass = [4000,8000], nbins = 1):
    '''
    mag - Magnitude of object
    T - Temperatue [K]
    key - '"AB' or Johnson ['U','B','V','R','I']


    wavelengths - Wavelength/s for which photon count is needed [Angstroms].
    OR
    Specify bandpass and the number of bins within the bandpass.
    For example if wavelengths = None, and bandpass = [4000,8000] with nbins = 3:
        wavelengths = [4000,6000,8000]



    '''

    if wavelengths is None:
        wavelengths = np.linspace(bandpass[0], bandpass[1], nbins)

    scale_wavelength = filt_mean_wavelengths[key]
    print('Scaling by {} mag at {}'.format(key, scale_wavelength))

    # Flux from apparent magnitude
    _,mag_flambda = Vegamag_to_flux(mag = mag,key = key) # [Watts/m^2/Angstrom]
    bbflux_lam = (blackbody_lambda(wavelengths, T)).to(u.W/(u.AA*u.m**2*u.sr))*4*np.pi*u.sr #[Watts/m^2/Angstrom/sr]


    search = np.where(wavelengths == scale_wavelength.value)[0]
    if len(search) == 0:
        scaling = mag_flambda / ((blackbody_lambda(scale_wavelength, T)).to(u.W/(u.AA*u.m**2*u.sr)) * 4 * np.pi* u.sr)#[Watts/m^2/Angstrom/sr]
    else:
        scaling = mag_flambda / bbflux_lam[search]

    scaled_bbflux = bbflux_lam * scaling # The blackbody flambda's sclaed by the flambda from the given Vega magnitude


    return wavelengths, scaled_bbflux

"""

def mag_to_photon(mag,T,bandpass,nbins=1,key='AB',scale_wavelength=0):
    '''
    Input:
        mag - Magnitude of object
        T - Temperatue [K]
        wl - Wavelength/s for which photon count is needed [Angstroms]
        bandpass - Entire bandpass (spectrum) for which flux is required [Angstroms]. Eg. [3500,9300]
        nbins - No. of spectral bins required.
        key - '"AB' or Johnson ['U','B','V','R','I']
        scale_wavelength - Wavelength at which to compare the BB curve to apparent luminosity to get scaling due to distance [Angstrom]
    Output:
        Lum - Luminosity across the wl array [Watts/m2]
        Rate - Photon rate across the wl array defined in [phot/sec/m^2] as received on Earth. Scaled by apparent magnitude.
    '''

    Lum,Rate = _Vegamag_to_photon(mag,T,np.linspace(*bandpass,num = 5000))
    Lum1 = np.zeros(nbins)
    Rate1 = np.zeros(nbins)
    count = 0
    for i in range(1,len(Lum)):
        Lum1[count]+= Lum[i]
        Rate1[count]+= Rate[i]

        if i%(len(Lum)/nbins)==0:
            count+=1


    return Lum1,Rate1



def _Vegamag_to_photon(mag,T,wl,key='AB',scale_wavelength=0, bin_size = None):
    '''
    Input:
        mag - Magnitude of object
        T - Temperatue [K]
        wl - Wavelength/s for which photon count is needed [Angstroms]
        key - '"AB' or Johnson ['U','B','V','R','I']
        scale_wavelength - Wavelength at which to compare the BB curve to apparent luminosity to get scaling due to distance [Angstrom]
        bin_size - dlambda [Angstrom]. Specified as bin_size = a*u.Angstrom
    Output:
        Lum - Luminosity across the wl array [Watts/m2]
        Rate - Photon rate across the wl array defined in [phot/sec/m^2] as received on Earth. Scaled by apparent magnitude.
    '''



    wl = wl.to(u.AA)

    if np.size(wl)==1:
        if bin_size is None:
            bin_size = 5500 * u.AA
    else:
        bin_size = np.median(np.diff(wl))*u.AA

    scale_wavelength = filt_mean_wavelengths[key]
    bb_wl = np.sort(np.linspace(np.max(wl),scale_wavelength,1000))
    bb_wl = np.append(wl.value, scale_wavelength.value)*u.AA
    mid =  len(bb_wl) - 1

    # Flux from apparent magnitude
    _,flambda = Vegamag_to_flux(mag = mag, wl = scale_wavelength,key = key) # [Watts/m^2/Angstrom]

    bbflux_lam = (blackbody_lambda(bb_wl, T)).to(u.W/(u.AA*u.m**2*u.sr)) #[Watts/m^2/Angstrom/sr]

    scaling = flambda / (bbflux_lam[mid]*np.pi*4*u.sr) # Scaling in [Watts/m^2/Angstrom]

    bbflux_lam = bbflux_lam[:-1]

    # Convert from specific intensity to luminosity (power)
    Lum = (((bbflux_lam * bin_size * 4 * np.pi * u.sr) * scaling)).to(u.W/u.m**2)

    # Number of photons per second
    Rate = (((Lum * wl / (ac.h*ac.c)))).to(1/u.s/u.m/u.m)

    return Lum,Rate


"""
