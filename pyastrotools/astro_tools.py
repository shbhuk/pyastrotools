'''
1. deg_to_rad() - Convert degree to radians
2. rad_to_deg() - Convert radians to degrees
3. dec_to_time() - Convert decimal to sexagesimal
4. time_to_dec() - Convert sexagesimal to decimal
5. galactic_to_eq_declination - Convert galactic coordinates into equatorial (declination)
6. galactic_to_eq_ra - Convet galactic coordinates into equatorial (RA)
7. ra_dec_difference - Distance between two RA,Dec coordinates
8. abs_to_app - Convert absolute magnitude to apparent magnitude
9. app_to_abs - Convert apparent magnitude to absolute magnitude
10.rv_magnitude - Calculate RV signal using Semi major axis
11.rv_magnitude_period - Calculate RV signal using Period
12.rv_magnitude_period_uncertainty - Calculate RV signal using Period, and also calculate error bars
13.rv_magnitude_period_uncertainty - Calculate RV signal using Period, and also calculate error bars
14.change_spectral_bin - Change dnu to dlambda or vice versa
15.get_stellar_data_and_mag
16.wav_airtovac - Convert wavelength air to vacuum
17.wav_vactoair - Convert wavelength vacuum to air
18.mdwarf_r_from_teff - Calculate Mdwarf radius from Teff
19.mdwarf_teff_from_r - Calculate Mdwarf Teff from radius
20.fgk_tess_from_mr_feh - Use M,R to calculate log(g), and then invert the Torres 2010 relation to find  Teff.
21.calculate_stellar_luminosity - Calculate Stellar luminosity from Radius and Temperature
22.calculate_insolation_flux - Calculate insolation flux on a planet
23.calculate_semi_major_axis - Calculate semi major axis for planet
24.calculate_orbvelocity - Calculate orbital velocity from period, or vice versa.
25.calculate_orbperiod - Calculate orbital period given primary stellar mass and semi major axis
26.calculate_eqtemperature = Calculate equilibrium temperature
27.calculate_TSM - Calculate the transmission spectroscopy metric from Kempton 2018
28. CalculateScaleHeight - Calculate planetary scale height 
29. CalculateSurfaceGravity - Calculate surface gravity for a given M and R
30. CalculateCoreMass_Fortney2007 - Calculate the core mass based on Fortney 2007
'''

import numpy as np
import math, os
from scipy import stats
from scipy.interpolate import interp1d, griddata
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from uncertainties.umath import *
from uncertainties import ufloat


import astropy
from astropy.io import fits
from astropy import constants as ac
from astropy import units as u
from astropy.utils.exceptions import AstropyDeprecationWarning
import astropy.units as u
from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
from astroquery.mast import Catalogs


c = ac.c.value #[m/s]

CodeDir = os.path.dirname(os.path.abspath(__file__))




def deg_to_rad(d):
	'''
	Input: Angle in degrees
	Output: Angle in radians
	
	Shubham Kanodia 27th April 2021
	'''

	return d*np.pi/180.

def rad_to_deg(r):
	'''
	Input: Angle in radians
	Output: Angle in degrees
	
	Shubham Kanodia 27th April 2021
	'''

	return r*180./np.pi

def dec_to_time(d):
	'''
	Decimal to Sexigesimal

	Input: Angle in degrees.
	Output: Angle in degrees,minutes,seconds

	Shubham Kanodia 27th April 2021
	'''
	degrees=math.floor(d)
	minutes=math.floor((d-degrees)*60.)
	seconds=(d-degrees-(minutes/60.))*3600.

	return degrees,minutes,seconds


def time_to_dec(d,m,s):
	'''
	Sexagesimal to Decimal

	Input:
		Degrees, Minutes, Seconds
	Output:
		Decimal angle
				
	Shubham Kanodia 27th April 2021
	'''

	return d+m/60.+s/3600.




def galactic_to_eq_declination(l,b,l_0,delta_0):
	'''
	Returns declination in degrees

	Shubham Kanodia 27th April 2021
	'''
	a=math.asin(math.cos(b)*math.sin(l-l_0)*math.sin(delta_0)+math.sin(b)*math.cos(delta_0))
	return rad_to_deg(a)




def galactic_to_eq_ra(l,b,delta,l_0,alpha_0):
	'''
	
	Returns RA in degrees, and hours
	
	Shubham Kanodia 27th April 2021
	'''

	A=math.cos(b)*math.cos(l-l_0)/math.cos(delta)
	C=math.acos(A)
	D=C+alpha_0

	if D>2*np.pi:
		D=D-2*np.pi


	return rad_to_deg(D),rad_to_deg(D)/15

def ra_dec_difference(delta_1,delta_2,alpha_1,theta):
	'''
	Distance between two RA,Dec coordinates
	All in degrees
	
	Shubham Kanodia 27th April 2021
	'''

	delta_1=deg_to_rad(delta_1)
	delta_2=deg_to_rad(delta_2)
	alpha_1=deg_to_rad(alpha_1)
	theta=deg_to_rad(theta)

	A=math.acos((math.sin(delta_1)*math.sin(delta_2))+(math.cos(delta_1)*math.cos(delta_2)))+alpha_1

	return rad_to_deg(A)


# H=rad_to_deg(math.acos(((0.866)-(math.sin(deg_to_rad(60.8))*math.sin(deg_to_rad(40.79))))/(math.cos(60.8)*math.cos(40.79))))


def abs_to_app(d,M=5):
	"""
	d is the distance in parsec
	M is the Absolute magnitude
	"""
	return M+2.5*math.log10((d/10)**2)

def app_to_abs(d,m):
	'''
	d is in pc
	m : Apparent magnitude
	'''
	return m-5*np.log10(d)+5

def rv_magnitude(pl_masse, st_mass, pl_orbsmax, pl_orbincl=np.pi/2, pl_orbeccen=0):
	'''
	INPUT:
		pl_masse - Mass of planet in Earth mass
		st_mass - Mass of star in sol mass
		pl_orbsmax - Semi Major Axis in AU
		pl_orbincl - Inclination wrt LOS, radians.
		pl_orbeccen - Eccentricity
	OUTPUT:
		RV in m/s

	from pyastrotools.astro_tools
	Shubham Kanodia 27th April 2021
	'''
	pl_rvamp = (28.4329 * pl_masse * np.sin(pl_orbincl) / ((ac.M_jup/ac.M_earth).value * np.sqrt(1-pl_orbeccen*pl_orbeccen))) * np.sqrt(1 / (st_mass * pl_orbsmax))

	return pl_rvamp

def rv_magnitude_period(pl_masse, st_mass, pl_orbper, pl_orbincl=np.pi/2, pl_orbeccen=0):
	'''
	INPUT:
		pl_masse - Mass of planet in Earth mass
		st_mass - Mass of star in sol mass
		pl_orbsmax - Semi Major Axis in AU
		pl_orbincl - Inclination wrt LOS, radians.
		pl_orbeccen - Eccentricity
		pl_orbper - Orbital Period in years

	OUTPUT:
		RV in m/s
		

	from pyastrotools.astro_tools
	Shubham Kanodia 27th April 2021
	'''
	pl_rvamp = (28.4329 * pl_masse * np.sin(pl_orbincl)/ ((np.sqrt(1-pl_orbeccen*pl_orbeccen)*317.8284065946748))) * (st_mass+(pl_masse*3.0034893488507934e-06))**(-2/3) * pl_orbper**(-1/3)

	return pl_rvamp

def rv_magnitude_period_uncertainty(pl_masse, st_mass, pl_orbper, pl_orbincl=np.pi/2, pl_orbeccen=0.0, pl_masseerr1=0.0, st_masserr1=0.0, pl_orbpererr1=0.0, pl_orbinclerr1=0.0, pl_orbeccenerr1=0.0):
	'''
	INPUT:
		pl_masse - Mass of planet in Earth mass
		st_mass - Mass of star in sol mass
		pl_orbincl - Inclination wrt LOS, radians
		pl_orbeccen - Eccentricity
		pl_orbper - Orbital Period in years
		pl_masserr1 - Uncertainty in planetary mass
		st_masserr1 - Uncertainty in stellar mass
		pl_orbpererr1 - Uncertainty in period
		pl_orbinclerr1 - Uncertainty in inclination angle, radians
		pl_orbeccenerr1 - Uncertainty in eccentricity

	OUTPUT:
		RV in m/s
		
	from pyastrotools.astro_tools
	Shubham Kanodia 27th April 2021
		
	'''
	from uncertainties import ufloat

	pl_masse = ufloat(pl_masse , pl_masseerr1)
	st_mass = ufloat(st_mass, st_masserr1)
	pl_orbper = ufloat(pl_orbper, pl_orbpererr1)
	pl_orbincl = ufloat(pl_orbincl, pl_orbinclerr1)
	e = ufloat(pl_orbeccen, pl_orbeccenerr1)

	pl_rvamp = (28.4329 * pl_masse * sin(pl_orbincl)/ ((sqrt(1-e*e)*317.8284065946748))) * (st_mass+(pl_masse *3.0034893488507934e-06))**(-2/3) * pl_orbper**(-1/3)

	return pl_rvamp

def rv_magnitude_period_autograd(mp,ms,P,i=np.pi/2,e=0.0, dmp=0.0, dms=0.0, dP=0.0, di=0.0, de=0.0):
	'''
	INPUT:
		mp - Mass of planet in Earth mass
		ms - Mass of star in sol mass
		i - Inclination wrt LOS
		e - Eccentricity
		P - Orbital Period in years

		dmp, dms, dP, di, de - Uncertainties in each term

	OUTPUT:
		RV in m/s
		
	from pyastrotools.astro_tools
	Shubham Kanodia 27th April 2021
	'''
	# https://github.com/HIPS/autograd
	import autograd.numpy as np
	from autograd import grad

	partial_mp = grad(rv_magnitude_period,0)(mp, ms, P, i, e)*dmp
	partial_ms = grad(rv_magnitude_period,1)(mp, ms, P, i, e)*dms
	partial_P = grad(rv_magnitude_period,2)(mp, ms, P, i, e)*dP
	partial_i = grad(rv_magnitude_period,3)(mp, ms, P, i, e)*di
	partial_e = grad(rv_magnitude_period,4)(mp, ms, P, i, e)*de

	dK = np.sqrt(partial_mp**2 + partial_ms**2 + partial_P**2 + partial_i**2 + partial_e**2)

	return dK


def change_spectral_bin(wavelength=None, nu=None, dnu=None, dlambda = None):
	'''
	Take dnu and wavelength to give dlambda
	INPUT:
		dnu : Width in frequency space[Hz]
		dlambda:
		wavelength: [m]
		nu : [Hz]

	OUTPUT:
		wavelength, dlambda, nu, dnu

	from pyastrotools.astro_tools
	Shubham Kanodia 27th April 2021

	'''

	if nu:
		wavelength = ac.c/nu
	else:
		nu = ac.c/wavelength

	if dnu:
		dlambda = dnu*wavelength*wavelength/ac.c
	elif dlambda:
		dnu = ac.c*dlambda/(wavelength**2)

	return {'lambda':wavelength.to(u.m),'dlambda':dlambda.to(u.m),'nu':nu.to(u.Hz),'dnu':dnu.to(u.Hz)}



def _QuerySimbad(Name, ReturnAll=False):
	'''
	Function to query Simbad for following stellar information RA, Dec, PMRA, PMDec, Parallax Epoch
	INPUTS:
		name = Name of source. Example


	'''
	warning = []

	print('Querying SIMBAD for {}'.format(Name))

	customSimbad = Simbad()
	customSimbad.add_votable_fields('ra(2;A;ICRS;J2000)', 'dec(2;D;ICRS;J2000)','pm', 'plx','parallax','rv_value', 'sptype',
	'flux(U)','flux(B)','flux(V)','flux(R)','flux(I)','flux(J)','flux(H)','flux(K)', 'ids')
	#Simbad.list_votable_fields()
	customSimbad.remove_votable_fields( 'coordinates')
	#Simbad.get_field_description('orv')
	obj = customSimbad.query_object(Name)
	if obj is None:
		raise ValueError('ERROR: {} target not found. Check target name or enter RA,Dec,PMRA,PMDec,Plx,RV,Epoch manually\n\n'.format(Name))
	else:
		warning += ['{} queried from SIMBAD.'.format(Name)]

	# Check for masked values
	if all([not x for x in [obj.mask[0][i] for i in obj.colnames]])==False:
		warning += ['Masked values present in output']


	obj = obj.filled(None)

	pos = SkyCoord(ra=obj['RA_2_A_ICRS_J2000'],dec=obj['DEC_2_D_ICRS_J2000'],unit=(u.hourangle, u.deg))
	ra = pos.ra.value[0]
	dec = pos.dec.value[0]
	pmra = obj['PMRA'][0]
	pmdec = obj['PMDEC'][0]
	plx = obj['PLX_VALUE'][0]
	rv = obj['RV_VALUE'][0] * 1000 #SIMBAD output is in km/s. Converting to m/s
	epoch = 2451545.0
	sp_type = obj['SP_TYPE'][0]

	star = {'ra':ra,'dec':dec,'pmra':pmra,'pmdec':pmdec,'px':plx,'rv':rv,'epoch':epoch, 'sp_type':sp_type,
	'Umag':obj['FLUX_U'][0],'Bmag':obj['FLUX_B'][0],'Vmag':obj['FLUX_V'][0],'Rmag':obj['FLUX_R'][0],
	'Imag':obj['FLUX_I'][0],'Jmag':obj['FLUX_J'][0],'Hmag':obj['FLUX_H'][0],'Kmag':obj['FLUX_K'][0]}

	if ReturnAll:
		return star, warning, obj
	else:
		return star,warning


def _QueryTIC(Name, Radius=2):
	"""
	Query the TIC Catalogue
	
	Name: Name to query catalogue. For example 'Proxima' or 'TIC 172370679'
	Radius: Radius in arcseconds to query
	"""
	_d = Catalogs.query_object(Name.replace(' ', '').replace('-', '').lower(), radius=Radius*u.arcsec, catalog="TIC").to_pandas()
	
	return _d


def _QueryGaia(coord, Radius=20):
	"""
	Query Gaia catalogue
	coord: Astropy SkyCoord object
	Radius: Radius in arcseconds to query
	"""
	from astroquery.gaia import Gaia

	j = Gaia.cone_search_async(coord, 20*u.arcsec)
	r = j.get_results()
	
	return r



def get_stellar_data_and_mag(name='',
					RA=None, Dec=None, PMRA=None, PMDec=None,PMEpoch = 2015.5, Equinox=2000.0, Vmag=None, Jmag=None,
					QueryTIC=True, QueryGaia=True, QuerySimbad=True):
	'''
	Function to query Simbad for following stellar information RA, Dec, PMRA, PMDec, Parallax Epoch
	INPUTS:
		name = Name of source. Example

	from pyastrotools.astro_tools
	Shubham Kanodia 27th April 2021

	'''
	warning = []

	star_input = {'ra':RA,'dec':Dec,'pmra':PMRA,'pmdec':PMDec,'PMEpoch':PMEpoch, 'Equinox':Equinox, 'Vmag': Vmag, 'Jmag':Jmag, 'rv':None, 'sp_type':None,
	'Umag':None,'Bmag':None,'Rmag':None,'Imag':None,'Hmag':None,'Kmag':None}
	star_simbad = {}
	star_tic = {}

	if QuerySimbad:
		star, warning = _QuerySimbad(name)

		star_simbad = star.copy()
	if QueryTIC:
		print('Querying TIC for {}'.format(name))
		_d = _QueryTIC(name)
		if len(_d)==0:
			print("Nothing found in TIC for {}".format(name))
		else:
			star_tic['ra'] = _d.ra[0]
			star_tic['dec'] = _d.dec[0]
			star_tic['pmra'] = _d['pmRA'][0]
			star_tic['pmdec'] = _d['pmDEC'][0]
			star_tic['Vmag'] = _d['Vmag'][0]
			star_tic['Jmag'] = _d['Jmag'][0]
			star_tic['Bmag'] = _d['Bmag'][0]

		star_output = star_tic.copy()
	else:
		star_output = star_input.copy()
	star_output.update({k:star_simbad[k] for k in star_output if star_output[k] is None})
	star_output.update({k:star_input[k] for k in star_input if star_input[k] is not None})

	coord = SkyCoord(ra=star_output['ra'], dec=star_output['dec'], unit=(u.degree, u.degree), frame='icrs')


	if QueryGaia:
		from astroquery.gaia import Gaia

		r = _QueryGaia(coord, Radius=50)
		star_output['pmra'] = r['pmra'][0]
		star_output['pmdec'] = r['pmdec'][0]
		print('Querying GAIA for {}'.format(name))

	# Fill Masked values with None. Again.
	for i in [k for k in star_output if star_output[k]==1e20]:
		star_output[i]=star_output

	return star_output

def wav_airtovac(la):
	""" Converts air wavelength (Angstoms) to vaccum wavelength (Angstoms).
	Input should be in Angstoms.
	Using the equation 3 in IAU standard accorind to Morton 1991
	http://adsabs.harvard.edu/abs/1991ApJS...77..119M
	
	from pyastrotools.astro_tools
	Shubham Kanodia 27th April 2021
	
	"""
	sig = 1e4/la
	# n - 1 =
	n_m_1 = 6.4328e-5 + 2.94981e-2/(146-sig**2) + 2.5540e-4/(41-sig**2)
	lv = la + la*n_m_1
	return lv

def wav_vactoair(lv):
	""" Converts vaccum wavelength (Angstoms) to air wavelength (Angstoms).
	Input should be in Angstoms.
	Using the equation 3 in IAU standard accorind to Morton 1991
	http://adsabs.harvard.edu/abs/1991ApJS...77..119M
	
	from pyastrotools.astro_tools
	Shubham Kanodia 27th April 2021
	
	"""
	sig = 1e4/lv
	# n - 1 =
	n_m_1 = 6.4328e-5 + 2.94981e-2/(146-sig**2) + 2.5540e-4/(41-sig**2)
	la = lv/(1 + n_m_1)
	return la

def mdwarf_r_from_teff(st_teff,plot=False):
	"""
	Get M-dwarf radius from effective temperature

	INPUT:
		effective temperature in K. Has to be between 2700K and 4200K

	OUTPUT:
		stellar radius in solar radii
		is between 0.125 R_sun and 0.68 R_sun

	NOTES:
		Figure 9 in Mann et al. 2015
		Equation 4
		Constants in Table 1
		
	from pyastrotools.astro_tools
	Shubham Kanodia 27th April 2021
		
	"""
	if np.any(st_teff>4200):
		print("Warning {} elements with TEFF>4200K".format(np.sum(st_teff>4200)))
	if np.any(st_teff>2700):
		print("Warning {} elements with TEFF<2700K".format(np.sum(st_teff<2700)))
	a = 10.5440
	b = -33.7546
	c = 35.1909
	d = -11.59280
	X = st_teff/3500.
	R = a + b*X + c*X**2. + d*X**3.
	if plot:
		fig, ax = plt.subplots()
		_T = np.linspace(2700,4200) # Mann sample
		_X = _T/3500.
		_R = a + b*_X + c*_X**2. + d*_X**3.
		ax.plot(st_teff,R,'k.')
		ax.plot(_T,_R,color='red')
	return R

def mdwarf_teff_from_r(st_rad,plot=False):
	"""
	Interpolates the mdwarf_r_from_teff function

	INPUT:
		radius in solar radii

	OUTPUT:
		Teff in K

	NOTES:
		only valid for 0.125 R_sun and 0.68 R_sun. If not, will return nan
		
	from pyastrotools.astro_tools
	Shubham Kanodia 27th April 2021
		
	"""
	m = (st_rad>0.125)&(st_rad<0.68)
	if np.sum(~m)>0:
		print('Using {} elements and {} are nans'.format(np.sum(m),np.sum(~m)))

	_T = np.linspace(2700,4200) # Mann sample
	_R = mdwarf_r_from_teff(_T)
	teffs = np.ones(np.size(st_rad))*np.nan
	teffs[m] = interp1d(_R,_T,kind='linear')(st_rad[m])
	if plot:
		fig, ax = plt.subplots()
		ax.plot(teffs,st_rad,'k.')
	return teffs

def Mann2015_mdwarf_r_from_ks(AbsK):
	"""
	Calculate the stellar radius from absolute K mag from eqn 4 in Mann 2015
	Typical scatter in radius is ~20%
	"""
	
	a = 1.9515
	b = -0.3520
	c = 0.01680
	
	radius = a + b*AbsK + c*(AbsK**2)
	return radius
	

def fgk_teff_from_mr_feh(st_mass, st_rad, FeH=0):
	"""
	Calculate Teff from Stellar Mass, Radius and Fe/H.
	Use M,R to calculate log(g), and then invert the Torres 2010 relation to find
	Teff.

	CAUTION: Approximate, check for precise applications.
	http://adsabs.harvard.edu/abs/2010A%26ARv..18...67T

	INPUTS:
		st_mass - Mass of the star in solar mass
		st_rad - Radius of the star in solar radii
		FeH - Stellar Metallicity
	OUTPUTS:
		teff - Effective temperature of the star in Kelvins

	from pyastrotools.astro_tools
	Shubham Kanodia 27th April 2021

	"""
	g = (ac.G*(st_mass * ac.M_sun)/((st_rad * ac.R_sun)**2)).to(u.cm/(u.s)**2)
	logg = np.log10(g.value)

	ai = [1.5689,1.3787,0.4243,1.139,-0.14250,0.01969,0.10100]
	bi = [2.4427,0.6679,0.1771,0.705,-0.21415,0.02306,0.04173]


	logM = np.log10(st_mass)
	logR = np.log10(st_rad)

	Mass_LHS = logM - ai[6]*FeH - ai[5]*(logg)**3 - ai[4]*(logg)**2 - ai[0]
	Radius_LHS = logR - bi[6]*FeH - bi[5]*(logg)**3 - bi[4]*(logg)**2 - bi[0]

	x_m = np.roots([ai[3], ai[2], ai[1], -Mass_LHS])
	T_m = 10 ** (x_m[np.isreal(x_m)].real + 4.1)

	x_r = np.roots([bi[3], bi[2], bi[1], -Radius_LHS])
	T_r = 10 ** (x_r[np.isreal(x_r)].real + 4.1)

	teff = T_m/2. + T_r/2.

	return teff

def calculate_stellar_luminosity(st_rad, st_teff, st_raderr1=0.0, st_tefferr1=0.0):
	"""
	Calculate the Stellar luminosity relative to solar luminosity.
	INPUT:
		st_rad: Radius of the star in units of solar radii
		Teff: Effective temperature of star in Kelvin. Assuming Teff_sol = 5777K
		Uncertainties in st_rad, Teff
	OUTPUT:
		L_star: In units of L_sol
		
	from pyastrotools.astro_tools
	Shubham Kanodia 27th April 2021
	"""
	from uncertainties import ufloat

	st_rad = ufloat(st_rad, st_raderr1)
	Teff = ufloat(st_teff, st_tefferr1)

	Teff_sol = 5777

	return (st_rad)**2 * (Teff/Teff_sol)**4


def calculate_insolation_flux(st_lum, pl_orbsmax, st_lumerr1=0.0, pl_orbsmaxerr1=0.0):
	"""
	Calculate the insolation flux incident on a planet.
	INPUT:
		st_lum: Luminosity of the star in units of L_sol
		semi_major_axis: Semi major axis in AU
	OUTPUT:
		S: Insolation flux in units of S_earth
		
	from pyastrotools.astro_tools
	Shubham Kanodia 27th April 2021
		
	"""

	from uncertainties import ufloat
	st_lum = ufloat(st_lum, st_lumerr1)
	pl_orbsmax = ufloat(pl_orbsmax, pl_orbsmaxerr1)

	return st_lum * (1/pl_orbsmax)**2


def calculate_semi_major_axis(st_mass, pl_orbper, st_masserr1=0.0, pl_orbpererr1=0.0):
	"""
	Calculate the Semi Major Axis given the orbital period and the stellar mass.
	INPUTS:
		st_mass = Stellar Mass in Solar Masses
		st_masserr1 = Error
		pl_orbper = Orbital Period in years
		pl_orbpererr1 = Error
	OUTPUTS:
		pl_orbsmax = In AU
		
	from pyastrotools.astro_tools
	Shubham Kanodia 27th April 2021
		
	"""
	from uncertainties import ufloat

	pl_orbper = (pl_orbper*u.yr).to(u.s).value
	pl_orbpererr1 = (pl_orbpererr1*u.yr).to(u.s).value
	pl_orbper = ufloat(pl_orbper, pl_orbpererr1)
	st_mass = (st_mass*ac.M_sun).to(u.kg).value
	st_masserr1 = (st_masserr1*ac.M_sun).to(u.kg).value
	st_mass = ufloat(st_mass, st_masserr1)

	AU = (ac.au).to(u.m).value

	pl_orbsmax = (((pl_orbper**2) * (ac.G.value*st_mass))/(4*np.pi**2))**(1/3) / AU

	return pl_orbsmax


def calculate_orbvelocity(st_mass, pl_orbper=None, pl_orbsmax=None):
	"""
	Calculate
	INPUTS:
		st_mass = Stellar Mass in solar masses
		pl_orbper = Orbital period in years
		pl_obrsmax = Semi Major axis in AU
			Need either period or semi major axis
			
	from pyastrotools.astro_tools
	Shubham Kanodia 27th April 2021
		  
	"""
	st_mass = (st_mass*ac.M_sun).to(u.M_sun)

	if pl_orbper is not None:
		pl_orbper = (pl_orbper*u.yr)
		pl_orbsmax = calculate_semi_major_axis(st_mass=st_mass.to(u.M_sun).value, pl_orbper=pl_orbper.value).n * u.au

	elif pl_orbsmax is not None:
		pl_orbsmax = (pl_orbsmax)*u.au
		pl_orbper = calculate_orbperiod(st_mass.value, pl_orbsmax.value) * u.yr

	pl_orbvel = np.sqrt(ac.G*st_mass/(pl_orbsmax)).to(u.km/u.s)

	return {pl_orbper, pl_orbsmax, pl_orbvel}




def calculate_orbperiod(st_mass, pl_orbsmax):
	"""
	Calculate the orbital period given the Semi Major Axis and the stellar mass.
	INPUTS:
		st_mass = Stellar Mass in Solar Masses
		pl_orbsmax = In AU
	OUTPUTS:
		pl_orbper = Orbital Period in years
		
	from pyastrotools.astro_tools
	Shubham Kanodia 27th April 2021
		
	"""
	pl_orbsmax = (pl_orbsmax*ac.au).to(u.m)
	# ~ pl_orbper = (pl_orbper*u.yr).to(u.s)
	st_mass = (st_mass*ac.M_sun).to(u.kg)

	pl_orbper = (((pl_orbsmax**3) * (4*np.pi**2)/(ac.G*st_mass)))**(1/2)

	return pl_orbper.to(u.yr).value

def calculate_eqtemperature(st_rad, st_teff, pl_orbsmax, st_raderr1=0.0, st_tefferr1=0.0, pl_orbsmaxerr1=0.0):
	"""
	Calculate the equilibrium temperature for planets.
	INPUTS:
		st_rad = Stellar Radius in Sol radi
		st_teff = Effective Temperature in Kelvin
		pl_orbsmax = Semi major axis in AU
	OUTPUTS:
		pl_eqt =
		

	from pyastrotools.astro_tools
	Shubham Kanodia 27th April 2021	  
	"""
	AU2SolarRadii = u.AU.to(u.R_sun)

	from uncertainties import ufloat
	st_rad = ufloat(st_rad, st_raderr1)
	st_teff = ufloat(st_teff, st_tefferr1)
	pl_orbsmax = ufloat(pl_orbsmax*AU2SolarRadii, pl_orbsmaxerr1*AU2SolarRadii)


	return st_teff * ((st_rad/pl_orbsmax/2)**(1/2))




def calculate_TSM(pl_rade, pl_eqt, pl_masse, st_rad, st_j, pl_radeerr1=0.0, pl_eqterr1=0.0, pl_masseerr1=0.0, st_raderr1=0.0):
	"""
	Calculate Transmission Spectroscopy Metric (TSM) from Kempton 2018 (for JWST)
	INPUTS:
		pl_rade = Planetary radius in Earth radii
		pl_eqt = Equilibrium Temperature in Kelvin
		pl_masse = Planetary mass in Earth masses
		st_rad = Stellar Radius in Sol radi
		st_j = J magnitude
	OUTPUTS:
		TSM

	from pyastrotools.astro_tools
	Shubham Kanodia 27th April 2021

	"""

	if pl_rade < 1.5:
		scale = 0.19
	elif pl_rade < 2.75:
		scale = 1.26
	elif pl_rade < 4.0:
		scale = 1.28
	else:
		scale = 1.15

	from uncertainties import ufloat
	pl_rade = ufloat(pl_rade, pl_radeerr1)
	pl_eqt = ufloat(pl_eqt, pl_eqterr1)
	pl_masse = ufloat(pl_masse, pl_masseerr1)
	st_rad = ufloat(st_rad, st_raderr1)

	TSM = scale * (((pl_rade**3)*pl_eqt)/(pl_masse*st_rad*st_rad)) * 10**(-st_j/5)

	return TSM

def CalculateSurfaceGravity(pl_masse, pl_rade, pl_masseerr1=0.0, pl_radeerr1=0.0):
	"""
	Calculate the surface gravity for given planetary mass and radius
	
	INPUTS:
		
		pl_masse : Planetary mass (Earth masses)
		pl_rade : Planetary radius (Earth radii)
		pl_masseerr1: Planetary mass 1 sigma error (Earth masses): Default = 0
		pl_radeerr1: Planetary radius 1 sigma error (Earth radius): Default = 0
	
	OUTPUTS:
		g: Surface gravity (cm/s2)
		
	from pyastrotools.astro_tools
	Shubham Kanodia 23rd Nov 2021
	
	"""
	
	pl_rade = (pl_rade*u.R_earth).to(u.cm).value
	pl_radeerr1 = (pl_radeerr1*u.R_earth).to(u.cm).value
	pl_masse = (pl_masse*u.M_earth).to(u.g).value
	pl_masseerr1 = (pl_masseerr1*u.M_earth).to(u.g).value

	pl_rade = ufloat(pl_rade, pl_radeerr1)
	pl_masse = ufloat(pl_masse, pl_masseerr1)
	
	g = ac.G.cgs.value * pl_masse / (pl_rade**2)
	
	return g

def CalculateScaleHeight(pl_masse, pl_rade, pl_eqt, pl_masseerr1=0.0, pl_radeerr1=0.0, pl_eqterr1=0.0):
	"""
	Calcualte the planetary scale height
	
	INPUTS:
		pl_masse : Planetary mass (Earth masses)
		pl_rade : Planetary radius (Earth radii)
		pl_eqt: Planetary equilibrium temperature (K)
		pl_masseerr1: Planetary mass 1 sigma error (Earth masses): Default = 0
		pl_radeerr1: Planetary radius 1 sigma error (Earth radius): Default = 0
		pl_eqterr1: Planetary equilibrium temperature 1 sigma error (K): Default = 0
	
	OUTPUTS:
		H: Scale Height (km)
	
	from pyastrotools.astro_tools
	Shubham Kanodia 23rd Nov 2021
	"""
	
	g = CalculateSurfaceGravity(pl_masse, pl_rade, pl_masseerr1=pl_masseerr1, pl_radeerr1=pl_radeerr1)
	
	surfacegravity = g/100 # Convert cm/s2 to m/s2
	eqt = ufloat(pl_eqt, pl_eqterr1) # Kelvin
	
	
	# Using the mean molecular weight to be 2.2 x m_p for hydrogen molecule
	H = ac.k_B.value * eqt / (ac.m_p.value  * 2.2 * surfacegravity)
	
	return H/1000
	

def CalculateCoreMass_Fortney2007(QueryMass, QueryRadiusE, QueryEqT, QueryAge, Plot=False):
	"""
	QueryMass: Total planetary mass (Earth mass)
	QueryRadiusE: Total planetary radius (Earth radius)
	QueryEqT: Equilibrium temperature for planet (K): [Has to lie between 78 K, 1960 K]
	QueryAge: In Gyr. Fortney 2007 options are 0.3 Gyr, 1 Gyr, 4.5 Gyr. Will find the one closest
	"""
		
	"""
	Table 2 = Giant planet radii at 300 Myr
	Table 3 = Giant planet radii at 1 Gyr
	Table 4 = Giant planet radii at 4.5 Gyr
	
	from pyastrotools.astro_tools
	Shubham Kanodia 23rd Nov 2021
	"""

	AgeArray = np.array([0.3, 1, 4.5])
	AgeIndex = np.argmin(np.abs(AgeArray-QueryAge))
	TablePath = os.path.join(os.path.dirname(CodeDir), 'Data', 'Fortney2007_Tab{}.txt'.format(int(AgeIndex+2)))

	Rjup2Rearth = 11.208
	QueryRadius = QueryRadiusE / Rjup2Rearth

	#https://ui.adsabs.harvard.edu/abs/2007ApJ...659.1661F/abstract
	# Grid axes from Fortney 2007
	pl_coremasse = np.array([0, 10, 25, 50, 100])
	pl_totalmasse = np.array([17, 28, 46, 77, 129, 215, 318, 464, 774, 1292, 2154, 3594])
	pl_orbsmax = np.array([0.02, 0.045, 0.1, 1.0, 9.5])
	pl_eqt = np.array([1960, 1300, 875, 260, 78])


	# Open Table and reshape it into 3D array
	df = pd.read_csv(TablePath, delimiter='\t', skiprows=1).replace('-', np.nan)
	tarray = np.array(df.iloc[:,2:]).astype(float)
	t3d = np.reshape(tarray, (5, 5, 12))
	# df.iloc[:,2:]

	x = pl_totalmasse
	y = pl_coremasse
	z = pl_eqt

	# Maskout nans and form mesh grid
	array = np.ma.masked_invalid(t3d)
	xx, yy, zz = np.meshgrid(x, y, z)
	xx = xx.swapaxes(0,1).T
	yy = yy.swapaxes(0,1).T
	zz = zz.swapaxes(0,1).T

	xnew = QueryMass
	ynew = np.arange(0, 100)
	znew = QueryEqT
	np.shape(zz)
	xx1, yy1, zz1 = np.meshgrid(xnew, ynew, znew)
	x1 = xx[~array.mask]
	y1 = yy[~array.mask]
	z1 = zz[~array.mask]
	newarr = array[~array.mask]

	# Interpolate onto grid at known planet mass and EqT with range of core masses
	# GD1 is the pl_radj as a function of core mass
	GD1 = griddata(points=(x1, y1, z1), values=newarr.ravel(),
							  xi=(xx1, yy1, zz1),
								 method='linear')

	CoreMass = interp1d(GD1[:,0,0], ynew, fill_value=(np.nan, 0), bounds_error=False)(QueryRadius)

	if Plot:
		plt.plot(ynew, GD1[:,0,0], label="Interpolated value")
		plt.xlabel("Core Mass $(M_{\oplus}$)")
		plt.ylabel("Planet Radius $(R_J$)")
		plt.axhline(QueryRadius, color='k', linestyle='dashed', label="Input planetary radius")
		plt.legend()
		
	return CoreMass
