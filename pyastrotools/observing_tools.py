import os
import sys
import numpy as np
import pytz
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import astropy.units as u
from astropy.time import Time
import datetime
from astroplan import Observer
from astroplan import EclipsingSystem, FixedTarget
from astroplan import (PrimaryEclipseConstraint, is_event_observable, AtNightConstraint, AltitudeConstraint, LocalTimeConstraint,MoonSeparationConstraint)
from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.coordinates import get_sun,get_moon

Location = os.path.dirname(os.path.abspath(__file__))
print(Location)


from .astro_tools import rv_magnitude_period, rv_magnitude_period_uncertainty, mdwarf_teff_from_r, fgk_teff_from_mr_feh, get_stellar_data_and_mag

try:
	ETCDirectory = r"C:\Users\shbhu\Documents\GitHub\TESS_MADNESS\src\ETC"
	sys.path.append(os.path.join(ETCDirectory, "NEID_ETC_20190329"))
	from neid_etcalc_public import NEID_RV_prec

	sys.path.append(os.path.join(ETCDirectory, "HPF"))
	from rvprec import HPF_ETC
except:
	print("Unable to load HPF and NEID ETCs")

try:
	from mrexo.predict import predict_from_measurement
except:
	print("Unable to import MRExo")


def find_location(obsname=None, lat=0., longi=0., alt=0.):
	'''
	Return Astropy EarthLocation object
	obsname : Name of Observatory as defined in Astropy EarthLocation routine. Can check list by EarthLocation.get_site_names().
			If obsname is not used, then can enter lat,long,alt.
						OR
	lat : Latitude of observatory in [degrees]. North (+ve) and South (-ve).
	longi : Longitude of observatory [degrees]. East (+ve) and West (-ve).
	alt : Altitude of observatory [m].
	
			
	from pyastrotools.observing_tools
	Shubham Kanodia 27th April 2021
	'''
	if obsname:
		location = EarthLocation.of_site(obsname)
		lat = location.lat.value
		longi = location.lon.value
		alt = location.height.value
		warning = [['Warning: Taking observatory coordinates from Astropy Observatory database. Verify precision. Latitude = %f  Longitude = %f  Altitude = %f'%(lat,longi,alt)]]
	else:
		location = EarthLocation.from_geodetic(longi, lat, height=alt)

	return location

def find_utc_offset(location, obstime, timezone=None):
	'''
	observatory - Astropy observatory object.
	time - Astropy Time object
	
	from pyastrotools.observing_tools
	Shubham Kanodia 27th April 2021	
	'''
	if timezone is None:
		timezone = location.info.meta['timezone']
	utcoffset = float(pytz.timezone(timezone).localize(obstime.datetime).strftime('%z'))/100
	return utcoffset*u.hour, timezone

def find_observatory(obstime, obsname, lat=0., longi=0., alt=0., timezone=None):
	'''
	Return Astropy Observer object

	obstime - Astropy Time object
	'''
	location = find_location(obsname, lat, longi, alt)
	utcoffset, timezone = find_utc_offset(location, obstime, timezone=timezone)
	observatory = Observer(location=location, name="", timezone=timezone)
	return observatory, utcoffset


def obs_planning_star(obstime, target_name=None, RA=None, Dec=None, obsname='', lat=None, longi=None, alt=None, title=None, utcoffset=None, timezone=None,
					QueryTIC=False, QueryGaia=False, QuerySimbad=True):
	'''
	Produce an altitude - azimuth - airmass plot for the target, Sun and the Moon.

	INPUT:
		pl_name : Planet name [String]. Eg. 'Kepler 225b'
		obstime : Astropy Time object for which to calculate
		obsname : Name of Observatory as defined in Astropy EarthLocation routine. Can check list by EarthLocation.get_site_names().
				If obsname is not used, then can enter lat,long,alt.
							OR
		lat : Latitude of observatory in [degrees]. North (+ve) and South (-ve).
		longi : Longitude of observatory [degrees]. East (+ve) and West (-ve).
		alt : Altitude of observatory [m].
		title : If title is specified, then will use for plot title
		utcoffset : Offset from UTC as -5 or -6 or -7, etc.
		timezone = 'US/Mountain' or 'US/Pacific'

	from pyastrotools.observing_tools
	Shubham Kanodia 27th April 2021
	'''
	from astropy.visualization import astropy_mpl_style
	plt.style.use(astropy_mpl_style)
	plt.figure()


	location = find_location(obsname=obsname, lat=lat, longi=longi, alt=alt)


	if utcoffset is None:
		utcoffset, timezone = find_utc_offset(location, obstime)
	else:
		utcoffset = utcoffset * u.h

	print(utcoffset)


	observatory = Observer(location=location, name="", timezone=timezone)


	print('Using {} as the UTC offset'.format(utcoffset))

	if target_name:
		try:
			targetobj = get_stellar_data_and_mag(name=target_name, RA=RA, Dec=Dec, QueryTIC=QueryTIC, QueryGaia=QueryGaia, QuerySimbad=QuerySimbad)
			# targetobj = Simbad.query_object(target_name)
			target = SkyCoord(ra=targetobj['ra'], dec=targetobj['dec'],unit=(u.deg, u.deg))
		except TypeError:
			print("Could not find on SIMBAD")
			target = SkyCoord(ra=RA, dec=Dec,unit=(u.hourangle, u.deg))

	else:
		target = SkyCoord(ra=RA, dec=Dec,unit=(u.hourangle, u.deg))
	print(target)

	# Find the alt,az coordinates of target at 100 times evenly spaced

	midnight = Time(np.floor(obstime.jd) + 0.5,format='jd',scale='utc') - utcoffset

	# Establish boundaries around transit to plot curve
	xboundary = []
	xboundary = [-6,18]


	delta_midnight = np.linspace(num=200, *xboundary)*u.hour
	times_obs = midnight+delta_midnight
	frame_obsnight = AltAz(obstime = times_obs, location=location)
	targetaltaz_obsnight = target.transform_to(frame_obsnight)

	BriefTimeArray = np.linspace(-8, 8, 17)*u.hour
	BriefTimeobs = midnight+BriefTimeArray
	BriefFrameObsNight = AltAz(obstime = BriefTimeobs, location=location)
	BriefTargetAltAz = target.transform_to(BriefFrameObsNight)

	_ = [print("UTC-{} - Alt={:.2f}, z={:.2f}, Az={:.2f}".format(str(BriefTimeobs[t].isot)[:-7], BriefTargetAltAz[t].alt, BriefTargetAltAz[t].secz, BriefTargetAltAz[t].az)) for t in range(len(BriefTimeobs))]

	sun_altaz = get_sun(times_obs).transform_to(frame_obsnight)
	moon_obj = get_moon(times_obs)
	moon_altaz = moon_obj.transform_to(frame_obsnight)
	moon_separation = np.min(moon_obj.separation(target)).deg
	moon_illumination = observatory.moon_illumination(obstime)


	fig = plt.figure()
	plt.grid(False)

	ax1 = fig.add_subplot(111)
	ax2 = ax1.twiny()

	ax1.grid(False); ax2.grid(False)


	plt.scatter(delta_midnight.value, targetaltaz_obsnight.alt,c=targetaltaz_obsnight.az, s=15, marker='.',
				cmap='viridis', label=target_name)

	# Plot the Sun, Moon and Airmass lines
	plt.plot(delta_midnight.value, sun_altaz.alt, color='orange', label='Sun')
	plt.plot(delta_midnight.value, moon_altaz.alt, color=[0.75]*3, ls='--', label='Moon')
	plt.hlines(48, *xboundary, 'r', 'dashed')
	plt.text(10, 49, 'Airmass 1.5')
	plt.hlines(30, *xboundary, 'r', 'dashed')
	plt.text(10, 31, 'Airmass 2')
	plt.legend(loc='upper left')

	#plt.title('{} -  {} \n Stellar st_teff : {} K \n Transit midpoint = UTC {} +- {} min \n JD {}'.format(pl_name,obsname, str(planet['st_teff'][0]),str(midpoint.datetime - datetime.timedelta(microseconds = midpoint.datetime.microsecond)),str(midpoint_err),str(midpoint.jd)),fontsize=15, pad = 5)
	#plt.title('{} -  {} \n Stellar st_teff : {} K \n Transit midpoint = UTC {} \n Transit midpoint = JD {}'.format(pl_name,obsname, str(planet['st_teff'][0]),str(midpoint.datetime - datetime.timedelta(microseconds = midpoint.datetime.microsecond)),str(midpoint.jd)),fontsize=15, pad = 5)

	if title:
		plt.title(title)
	else:
		plt.title('{}; {}; ObsName = {}\nLocal Midnight at UTC {}. JD {}. \nMoon Illum = {}%. Min. Moon Sep = {} deg'.format(target_name, target.to_string('hmsdms'), obsname, midnight.datetime,np.round(midnight.jd,2),
																										np.round(moon_illumination*100, 0), np.round(moon_separation, 2)), fontsize=15, pad = 5)
	plt.fill_between(delta_midnight.to('hr').value, 0, 90,
					sun_altaz.alt < -0*u.deg, color='0.7', zorder=0)
	plt.fill_between(delta_midnight.to('hr').value, 0, 90,
					sun_altaz.alt < -6*u.deg, color='0.6', zorder=0)
	plt.fill_between(delta_midnight.to('hr').value, 0, 90,
					sun_altaz.alt < -12*u.deg, color='0.5', zorder=0)
	plt.fill_between(delta_midnight.to('hr').value, 0, 90,
					sun_altaz.alt < -18*u.deg, color='k', zorder=0)
	LocalTicks = np.linspace(*xboundary, 9, dtype=int)
	LocalTickLabels = ["{}".format(s).zfill(2)+'00' for s in LocalTicks%24]
	UTCTicks = LocalTicks - int(utcoffset.value)
	ax1.set_xticks(LocalTicks)
	ax1.set_xticklabels(LocalTickLabels, rotation=90, size=12)
	ax1.set_xlabel('[Local Time]', size=12)
	ax2.set_xticks(LocalTicks)
	ax2.set_xticklabels(["{}".format(s).zfill(2)+'00' for s in UTCTicks%24], rotation=90, size=12)
	ax2.set_xlabel('[UTC]', size=12)
	plt.colorbar().set_label('Azimuth [deg]')
	plt.xlim(*xboundary)
	plt.ylim(0, 90)
	ax1.set_ylabel('Altitude [deg]')
	plt.tight_layout()
	plt.grid()
	# plt.show(block=False)


	return 1


def transiting_planet(pl_tranmid, pl_trandur, pl_orbper):
	'''
	Load transiting planet object as defined here -
	https://astroplan.readthedocs.io/en/latest/tutorials/periodic.html
	INPUT:
		pl_tranmid : Transit midpoint (JD)
		pl_trandur : Transit duration in days (Float)
		orb_per : Orbital period in days (Float)

	from pyastrotools.observing_tools
	Shubham Kanodia 27th April 2021
	'''

	epoch = Time(pl_tranmid, format = 'jd')
	transit_duration = pl_trandur * u.d
	orb_per = pl_orbper * u.d

	planet_transit_sys = EclipsingSystem(primary_eclipse_time = epoch, orbital_period = orb_per, duration = transit_duration)
	return planet_transit_sys


def TransitingSystem(pl_tranmid, pl_trandur, pl_orbper, start_query, pl_tranmiderr1=0.0, pl_trandurerr1=0.0, pl_orbpererr1=0.0, stop_query=None, n=1):
	"""
	Replacement to EclipsingSystem from Astroplan, since this includes uncertainties. Ignores eccentricities.

	Inputs:
		Ensure that all epochs are in UTC. Generally transit midpoints are reported in BJD
		pl_tranmid, pl_tranmiderr1 = Float in JD. Transit midpoint epoch and 1 sigma uncertainty
		pl_trandur, pl_trandurerr1 = Float in days. Transit duration in days and 1 sigma uncertainty
		pl_orbperr, pl_orbpererr1 = Float in days. Orbital Period and 1 sigma uncertainty

		start_query = Astropy Time object to start searching for transits
		stop_query = Optional, Astropy Time object to start searching for transits. If not defined, will look for 'n' transits.
		n = Number of transits after start_query, to look for. If undefined, default is n=1, and will return 1 transit for start_query.

	Outputs:
		All errors are 1 sigma and all outputs are Astropy Time objects for 'n' transits
		Returns the midpoint, midpoint error, ingress, ingress error, egress, egress error

		If there are no transits found

	https://arxiv.org/abs/2003.09046
	
	from pyastrotools.observing_tools
	Shubham Kanodia 27th April 2021
	"""
	n_start = np.ceil((start_query.jd - pl_tranmid)/pl_orbper)

	if stop_query:
		n_stop = np.ceil((stop_query.jd - pl_tranmid)/pl_orbper)
		n_array = np.arange(n_start, n_stop)
		if len(n_array) == 0:
			print("No transits found between {} and {}".format(start_query.isot, stop_query.isot))

	else:
		n_array = np.arange(n_start, n_start+n)




	nT_mid = Time(n_array*pl_orbper + pl_tranmid, format='jd')
	nT_miderr1 = np.sqrt(n_array**2 * pl_orbpererr1**2 + 2*n_array*pl_orbpererr1*pl_tranmiderr1 + pl_tranmiderr1**2)
	nT_ingress = nT_mid - ((pl_trandur/2) * u.d)
	nT_ingresserr1 = np.sqrt(nT_miderr1**2 + pl_trandurerr1**2)
	nT_egress = nT_mid + ((pl_trandur/2) * u.d)
	nT_egresserr1 = np.sqrt(nT_miderr1**2 + pl_trandurerr1**2)


	return nT_mid, nT_miderr1, nT_ingress, nT_ingresserr1, nT_egress, nT_egresserr1



def find_next_midtransit(pl_tranmid, pl_trandur, pl_orbper, start_query, stop_query='',n=1):
	'''
	Load transiting planet as defined here -
	https://astroplan.readthedocs.io/en/latest/tutorials/periodic.html
	Find the next n mid transit JDs
	INPUT:
		pl_tranmid : Transit midpoint (JD)
		pl_trandur : Transit duration in days (Float)
		orb_per : Orbital period in days (Float)		start_query : Astropy Time object after which to look for transits. In UTC
		stop_query : Astropy Time object until which to look for transits. In UTC
		n : Find 'n' transits

	from pyastrotools.observing_tools
	Shubham Kanodia 27th April 2021
	'''
	transit_sys = transiting_planet(pl_tranmid=pl_tranmid, pl_trandur=pl_trandur, pl_orbper=pl_orbper)

	if stop_query:
		mid_transit = transit_sys.next_primary_eclipse_time(start_query,1000)
		transits_in_range = mid_transit[np.where(mid_transit < stop_query)[0]]
		return transits_in_range
	else:
		mid_transit = transit_sys.next_primary_eclipse_time(start_query,n)
		return mid_transit


def find_next_fulltransit(pl_tranmid, pl_trandur, pl_orbper, start_query, stop_query='',n=1):
	'''
	Load transiting planet as defined here -
	https://astroplan.readthedocs.io/en/latest/tutorials/periodic.html
	Find the next n ingress-egress JDs (full transit) for transits
	INPUT:
		pl_tranmid : Transit midpoint (JD)
		pl_trandur : Transit duration in days (Float)
		orb_per : Orbital period in days (Float)		start_query : Astropy Time object after which to look for transits. In UTC
		stop_query : Astropy Time object until which to look for transits. In UTC
		n : Find 'n' transits
	OUTPUT:
		Transit ingress and egress in UTC
	'''

	transit_sys = transiting_planet(pl_tranmid=pl_tranmid, pl_trandur=pl_trandur, pl_orbper=pl_orbper)
	if stop_query:
		ing_egr = transit_sys.next_primary_ingress_egress_time(start_query,1000)
		result = ing_egr[np.where(ing_egr < stop_query)[0]][::2]
		if np.size(result)==0:
			result = transit_sys.next_primary_ingress_egress_time(start_query,1)

	else:
		result = transit_sys.next_primary_ingress_egress_time(start_query,n)
	return result


def find_next_fulltransit_observable(pl_name, RA, Dec,
									pl_tranmid, pl_trandur, pl_orbper,
									start_query, stop_query='', n = 1,
									obsname=None, lat=None, longi=None, alt=None, timezone=None,
									min_local_time=datetime.time(12,0), max_local_time = datetime.time(12,0),
									min_altitude = 30, min_moon_sep = 25,
									FindAtLeastOneTransit=True,
									pl_tranmiderr1=0.0, pl_trandurerr1=0.0, pl_orbpererr1=0.0):

	'''
	Not necessarily full transit, just ingress or egress would also be fine...
	Load transiting planet as defined here -
	https://astroplan.readthedocs.io/en/latest/tutorials/periodic.html
	Find the next n ingress-egress JDs for transits.
	Also specify constraints to check if full transit is observable
	INPUT:
		RA, Dec : In degrees
		pl_tranmid : Transit midpoint (JD)
		pl_trandur : Transit duration in days (Float)
		orb_per : Orbital period in days (Float)
		observatory : Astroplan observatory object.

					Example: observatory = Observer.at_site('APO', timezone='US/Mountain')
								observatory = Observer(longitude=-155.4761*u.deg, latitude=19.825*u.deg,
									elevation=0*u.m, name="Subaru", timezone="US/Hawaii")

		start_query : Astropy Time object after which to look for transits. In UTC
		stop_query : Astropy Time object until which to look for transits. In UTC
		n : Find 'n' transits
		min_local_time : Minimum local (observatory) time for observation. Default is 12 noon. Example: datetime.time(12,0)
		max_local_time : Maximum local (observatory) time for observation. Default is 12 noon. Example: datetime.time(12,0)
		min_altitude : Minimum altitude above the horizon for observation. [Degrees]
		min_moon_sep : Minimum separation from the moon. [Degrees]
		FindAtLeastOneTransit: If no observable transit is found between start and stop, then looks beyond.
		pl_trandurerr1, pl_tranmiderr1, pl_orbpererr1 : All in days

		Output:
			IN UTC
	from pyastrotools.observing_tools
	Shubham Kanodia 27th April 2021
	'''


	observatory, utcoffset = find_observatory(obstime=start_query, obsname=obsname, lat=lat, longi=longi, alt=alt, timezone=timezone)
	print('Using {} as the UTC offset'.format(utcoffset))

	target = FixedTarget(coord=SkyCoord(ra=RA*u.deg, dec=Dec*u.deg), name=pl_name)
	# Find the next n  Full Transit times
	# times = find_next_fulltransit(pl_tranmid=pl_tranmid, pl_trandur=pl_trandur, pl_orbper=pl_orbper, start_query = start_query, stop_query = stop_query, n=n)

	nT_mid, nT_miderr1, nT_ingress, nT_ingresserr1, nT_egress, nT_egresserr1 = TransitingSystem(pl_tranmid=pl_tranmid, pl_trandur=pl_trandur, pl_orbper=pl_orbper,
			start_query=start_query, stop_query=stop_query, n=n,
			pl_tranmiderr1=pl_tranmiderr1, pl_trandurerr1=pl_trandurerr1, pl_orbpererr1=pl_orbpererr1)

	# constraints = [AltitudeConstraint(min=min_altitude*u.deg), LocalTimeConstraint(min=min_local_time, max=max_local_time),MoonSeparationConstraint(min = min_moon_sep * u.deg)]
	constraints = [AtNightConstraint.twilight_civil(), AltitudeConstraint(min=min_altitude*u.deg), LocalTimeConstraint(min=min_local_time, max=max_local_time),MoonSeparationConstraint(min = min_moon_sep * u.deg)]

	results = []
	# Run only if at least one transit has been found in given time period between start and stop_query (if the latter is specified)
	if len(nT_mid) != 0:
		for i in range(len(nT_ingress)):
			if is_event_observable(constraints, observatory, target, nT_ingress[i]) or is_event_observable(constraints, observatory, target, nT_egress[i]):
				results.append([nT_mid[i], nT_miderr1[i], nT_ingress[i], nT_ingresserr1[i], nT_egress[i], nT_egresserr1[i]])


	# If cannot find a transit that satisfies the conditions between start and stop period, look at next 20 transits.
	if np.size(results)==0:
		c=0
		print('Could not find transit between start and stop period, finding next earliest available transit')
		if FindAtLeastOneTransit:
			while np.size(results)==0:
				nT_mid, nT_miderr1, nT_ingress, nT_ingresserr1, nT_egress, nT_egresserr1 = TransitingSystem(pl_tranmid=pl_tranmid, pl_trandur=pl_trandur, pl_orbper=pl_orbper,
						start_query=nT_egress[i], n=1,
						pl_tranmiderr1=pl_tranmiderr1, pl_trandurerr1=pl_trandurerr1, pl_orbpererr1=pl_orbpererr1)
				c+=1
				print(c, nT_ingress, nT_ingress)
				if is_event_observable(constraints, observatory, target, nT_ingress) or is_event_observable(constraints, observatory, target, nT_egress):
					results.append([nT_mid, nT_miderr1, nT_ingress, nT_ingresserr1, nT_egress, nT_egresserr1])
				if c==20:
					print("Could not find observable transit between {} and {}".format(start_query.isot, stop_query.isot))
					break
	return results


def find_next_midtransit_observable(pl_name, RA, Dec,
									pl_tranmid, pl_trandur, pl_orbper, obsname,
									start_query, stop_query='', n = 1,
									lat=None, longi=None, alt=None,
									min_local_time=datetime.time(12,0), max_local_time = datetime.time(12,0),
									min_altitude = 30, min_moon_sep = 25,
									FindAtLeastOneTransit=True,
									pl_tranmiderr1=0.0, pl_trandurerr1=0.0, pl_orbpererr1=0.0):
	'''
	Load transiting planet as defined here -
	https://astroplan.readthedocs.io/en/latest/tutorials/periodic.html
	Find the next n ingress-egress JDs for transits.
	Also specify constraints to check if mid transit is observable
	INPUT:
		pl_tranmid : Transit midpoint (JD)
		pl_trandur : Transit duration in days (Float)
		orb_per : Orbital period in days (Float)		observatory : Astroplan observatory object.

					Example: observatory = Observer.at_site('APO', timezone='US/Mountain')
								observatory = Observer(longitude=-155.4761*u.deg, latitude=19.825*u.deg,
									elevation=0*u.m, name="Subaru", timezone="US/Hawaii")

		start_query : Astropy Time object after which to look for transits. In UTC
		stop_query : Astropy Time object until which to look for transits. In UTC
		n : Find 'n' transits
		min_local_time : Minimum local (observatory) time for observation. Default is 12 noon. Example: datetime.time(12,0)
		max_local_time : Maximum local (observatory) time for observation. Default is 12 noon. Example: datetime.time(12,0)
		min_altitude : Minimum altitude above the horizon for observation. [Degrees]
		min_moon_sep : Minimum separation from the moon. [Degrees]
		FindAtLeastOneTransit: If no observable transit is found between start and stop, then looks beyond.
		pl_trandurerr1, pl_tranmiderr1, pl_orbpererr1 : All in days


	OUTPUT:
	Ingress, Egress JD times
	
	from pyastrotools.observing_tools
	Shubham Kanodia 27th April 2021	
	'''

	observatory, utcoffset = find_observatory(obstime=start_query, obsname=obsname, lat=lat, longi=longi, alt=alt)
	print('Using {} as the UTC offset'.format(utcoffset))

	target = FixedTarget(coord=SkyCoord(ra=RA*u.deg, dec=Dec*u.deg), name=pl_name)
	# Find the next n  Full Transit times
	# times = find_next_fulltransit(pl_tranmid=pl_tranmid, pl_trandur=pl_trandur, pl_orbper=pl_orbper, start_query = start_query, stop_query = stop_query, n=n)

	nT_mid, nT_miderr1, nT_ingress, nT_ingresserr1, nT_egress, nT_egresserr1 = TransitingSystem(pl_tranmid=pl_tranmid, pl_trandur=pl_trandur, pl_orbper=pl_orbper,
			start_query=start_query, stop_query=stop_query, n=n,
			pl_tranmiderr1=pl_tranmiderr1, pl_trandurerr1=pl_trandurerr1, pl_orbpererr1=pl_orbpererr1)

	# constraints = [AltitudeConstraint(min=min_altitude*u.deg), LocalTimeConstraint(min=min_local_time, max=max_local_time),MoonSeparationConstraint(min = min_moon_sep * u.deg)]
	constraints = [AtNightConstraint.twilight_civil(), AltitudeConstraint(min=min_altitude*u.deg), LocalTimeConstraint(min=min_local_time, max=max_local_time),MoonSeparationConstraint(min = min_moon_sep * u.deg)]

	results = []
	# Run only if at least one transit has been found in given time period between start and stop_query (if the latter is specified)
	if len(nT_mid) != 0:
		for i in range(len(nT_mid)):
			if is_event_observable(constraints, observatory, target, nT_mid[i]):
				results.append([nT_mid[i], nT_miderr1[i], nT_ingress[i], nT_ingresserr1[i], nT_egress[i], nT_egresserr1[i]])


	# If cannot find a transit that satisfies the conditions between start and stop period, look at next 20 transits.
	if np.size(results)==0:
		c=0
		print('Could not find transit between start and stop period, finding next earliest available transit')
		if FindAtLeastOneTransit:
			while np.size(results)==0:
				nT_mid, nT_miderr1, nT_ingress, nT_ingresserr1, nT_egress, nT_egresserr1 = TransitingSystem(pl_tranmid=pl_tranmid, pl_trandur=pl_trandur, pl_orbper=pl_orbper,
						start_query=nT_egress[i], n=1,
						pl_tranmiderr1=pl_tranmiderr1, pl_trandurerr1=pl_trandurerr1, pl_orbpererr1=pl_orbpererr1)
				c+=1
				print(c, nT_ingress, nT_ingress)
				if is_event_observable(constraints, observatory, target, nT_mid):
					results.append([nT_mid, nT_miderr1, nT_ingress, nT_ingresserr1, nT_egress, nT_egresserr1])
				if c==20:
					print("Could not find observable transit between {} and {}".format(start_query.isot, stop_query.isot))
					break


	return results



def obs_planning_transit(pl_name, RA, Dec,
						pl_tranmid, pl_trandur, pl_orbper,
						start_query, stop_query='', n = 1,
						pl_tranmiderr1=0.0, pl_trandurerr1=0.0, pl_orbpererr1=0.0,
						obsname=None, lat=None, longi=None, alt=None, timezone=None,
						min_local_time=datetime.time(12,0), max_local_time = datetime.time(12,0),
						title = None,
						min_altitude = 30, min_moon_sep = 25, FindAtLeastOneTransit=True,
						savedirectory=None):

	'''
	Produce an altitude - azimuth - airmass plot for the planet host, Sun and the Moon.

	INPUT:
		pl_tranmid : Transit midpoint (JD)
		pl_trandur : Transit duration in days (Float)
		orb_per : Orbital period in days (Float)		obsname : Astroplan observatory object.

					Example: obsname = Observer.at_site('APO', timezone='US/Mountain')
								obsname = Observer(longitude=-155.4761*u.deg, latitude=19.825*u.deg,
									elevation=0*u.m, name="Subaru", timezone="US/Hawaii")

		start_query : Astropy Time object after which to look for transits. In UTC
		stop_query : Astropy Time object until which to look for transits. In UTC
		n : Find 'n' transits
		title: For plot
		min_local_time : Minimum local (obsname) time for observation. Default is 7pm. Example: datetime.time(19,0)
		max_local_time : Maximum local (obsname) time for observation. Default is 12 midnight. Example: datetime.time(0,0)
		min_altitude : Minimum altitude above the horizon for observation. [Degrees]
		min_moon_sep : Minimum separation from the moon. [Degrees]
		FindAtLeastOneTransit: If no observable transit is found between start and stop, then looks beyond.
		pl_trandurerr1, pl_tranmiderr1, pl_orbpererr1 : All in days
		savedirectory: Default is None. If specified, then saves in that directory

	from pyastrotools.observing_tools
	Shubham Kanodia 27th April 2021
	'''

	from astropy.visualization import astropy_mpl_style
	plt.style.use(astropy_mpl_style)
	plt.figure()

	observatory, utcoffset = find_observatory(obstime=start_query, obsname=obsname, lat=lat, longi=longi, alt=alt, timezone=timezone)

	target = SkyCoord(ra=RA*u.deg, dec=Dec*u.deg)

	result = find_next_fulltransit_observable(pl_name=pl_name, RA=RA, Dec=Dec,
													pl_tranmid=pl_tranmid, pl_trandur=pl_trandur, pl_orbper=pl_orbper,
													obsname=obsname, lat=lat, longi=longi, alt=alt, timezone=timezone,
													start_query=start_query,stop_query=stop_query,
													min_local_time=min_local_time, max_local_time=max_local_time,
													min_altitude=min_altitude, min_moon_sep=min_moon_sep,
													FindAtLeastOneTransit=FindAtLeastOneTransit,
													pl_tranmiderr1=pl_tranmiderr1, pl_trandurerr1=pl_trandurerr1, pl_orbpererr1=pl_orbpererr1)
	if len(result)==0:
		print("No observable transits")
		return 1, 1, 1, 1
	else:
		MidPointTime = []
		IngressTime = []
		EgressTime = []
		MidnightTime = []
		for i in range(len(result)):
			midpoint, midpointerr1, ingress, ingresserr1, egress, egresserr1 = result[i]
			MidPointTime.append(midpoint)
			IngressTime.append(ingress)
			EgressTime.append(egress)

			ingress_er = ingress - ingresserr1
			egress_er = egress + egresserr1

			# Find the alt,az coordinates of target at 100 times evenly spaced
			midnight = Time(np.floor(ingress.value - 0.3) + 0.5, format='jd') - utcoffset

			MidnightTime.append(midnight)

			# Establish boundaries around transit to plot curve
			xboundary = []
			'''
			if (ingress.jd - midnight.jd)*24-6 < -12:
				xboundary.append((ingress.jd - midnight.jd)*24-6)
			else:
				xboundary.append(-12)
			'''
			xboundary = [-8]
			if (egress.jd - midnight.jd)*24+6 > 12:
				xboundary.append((egress.jd - midnight.jd)*24+6)
			else:
				xboundary.append(+8)
			#xboundary = [(ingress.jd - midnight.jd)*24-6,(egress.jd - midnight.jd)*24+6]
			# print(ingress, egress, midnight, xboundary)

			delta_midnight = np.linspace(num=200, *xboundary)*u.hour
			times_obs = midnight+delta_midnight
			frame_obsnight = AltAz(obstime=times_obs, location=observatory.location)
			targetaltaz_obsnight = target.transform_to(frame_obsnight)

			# Mark ingress, egress and transit midpoint
			transit_time = Time(np.linspace(ingress_er.jd,egress_er.jd, 100),format='jd',scale='utc')
			tmp_frame = AltAz(obstime=transit_time,location=observatory.location)
			tmp_C = target.transform_to(tmp_frame)
			# Find position for markers
			ing_er_pos = 0
			ing_pos = np.argmin(np.abs(transit_time.jd - ingress.jd))
			egr_er_pos = -1
			egr_pos = np.argmin(np.abs(transit_time.jd - egress.jd))
			mid_pos = int(len(transit_time)/2)

			slope_in_err = np.arctan(np.median(np.diff(tmp_C[ing_er_pos:ing_er_pos+5].alt.value)) / np.median(np.diff(transit_time[ing_er_pos:ing_er_pos+5].jd))) * 180/np.pi
			slope_in = np.arctan(np.median(np.diff(tmp_C[ing_pos:ing_pos+5].alt.value)) / np.median(np.diff(transit_time[ing_pos:ing_pos+5].jd))) * 180/np.pi
			slope_eg = np.arctan(np.median(np.diff(tmp_C[egr_pos-2:egr_pos+3].alt.value)) / np.median(np.diff(transit_time[egr_pos-2:egr_pos+3].jd))) * 180/np.pi
			slope_eg_err = np.arctan(np.median(np.diff(tmp_C[egr_er_pos-3:egr_er_pos].alt.value)) / np.median(np.diff(transit_time[egr_er_pos-3:egr_er_pos].jd))) * 180/np.pi


			fig = plt.figure(figsize=(10,8))
			plt.grid(False)

			ax1 = fig.add_subplot(111)
			ax2 = ax1.twiny()

			ax1.grid(False); ax2.grid(False)

			ing_er_plot, = plt.plot(((ingress_er.jd - midnight.jd)*24)*u.hour, tmp_C[ing_er_pos].alt, marker=(3,0,-np.abs(slope_in_err)),
							color='lime', markersize = 10, label='Ingress w/ uncert. = {}'.format(ingress_er.iso[:-4]))
			ing_plot, = plt.plot(((ingress.jd - midnight.jd)*24)*u.hour, tmp_C[ing_pos].alt, marker=(3,0,-np.abs(slope_in)), color='r', markersize = 10, label='Ingress = {}'.format(ingress.iso[:-4]))
			eg_plot, = plt.plot(((egress.jd - midnight.jd)*24)*u.hour,  tmp_C[egr_pos].alt, marker=(3,0,np.abs(slope_eg)), color='r', markersize = 10, label='Egress = {}'.format(egress.iso[:-4]))
			eng_er_plot, = plt.plot(((egress_er.jd - midnight.jd)*24)*u.hour, tmp_C[egr_er_pos].alt, marker=(3,0, np.abs(slope_eg_err)),
							color='lime', markersize = 10, label='Egress w/ uncert. = {}'.format(egress_er.iso[:-4]))
			mid_plot, = plt.plot(((midpoint.jd - midnight.jd)*24)*u.hour,tmp_C[mid_pos].alt, marker='o', color='r', markersize = 10, label='Midpoint = {}'.format(midpoint.iso[:-4]))

			plt.scatter(delta_midnight.value, targetaltaz_obsnight.alt, c = targetaltaz_obsnight.az, s = 15, marker = '.', cmap = 'viridis')

			sun_altaz = get_sun(times_obs).transform_to(frame_obsnight)
			moon_obj = get_moon(times_obs)
			moon_altaz = moon_obj.transform_to(frame_obsnight)
			moon_separation = np.min(moon_obj.separation(target)).deg
			moon_illumination = observatory.moon_illumination(midpoint)


			# Plot the Sun, Moon and Airmass lines
			sun_plot, = plt.plot(delta_midnight.value, sun_altaz.alt, color='orange', label='Sun')
			moon_plot, = plt.plot(delta_midnight.value, moon_altaz.alt, color=[0.75]*3, ls='--', label='Moon')
			plt.hlines(84, *xboundary, 'r', 'dashed')
			plt.text(-7, 85, '84 degree', size=15)
			plt.hlines(48, *xboundary, 'r', 'dashed')
			plt.text(-7, 49, 'Airmass 1.5', size=15)
			plt.hlines(30, *xboundary, 'r', 'dashed')
			plt.text(-7, 31, 'Airmass 2', size=15)
			legend1 = plt.legend(handles=[sun_plot, moon_plot], loc='upper left')
			legend2 = plt.legend(handles=[ mid_plot, ing_plot, eg_plot, ing_er_plot, eng_er_plot], loc='lower left', prop={'size':12})

			plt.gca().add_artist(legend1)
			plt.gca().add_artist(legend2)

			if title:
				plt.title(title)
			else:
				plt.title('{}; RA/Dec = {} \nTrans Dur. = {} h, Orb. Per. = {} d TranMid= JDUTC {}\nObsName = {}; Local Midnight at UTC {} = JD {}. \nMoon Illum = {}%. Min. Moon Sep = {} deg'.format(pl_name, target.to_string('hmsdms'),
																												np.round(pl_trandur*24, 2), np.round(pl_orbper, 3), np.round(midpoint.jd, 5), obsname,
																												midnight.iso[:-4], np.round(midnight.jd,2), np.round(moon_illumination*100, 0), np.round(moon_separation, 2)), fontsize=18, pad = 5)

			plt.fill_between(delta_midnight.to('hr').value, 0, 90,
							sun_altaz.alt < -0*u.deg, color='0.7', zorder=0)
			plt.fill_between(delta_midnight.to('hr').value, 0, 90,
							sun_altaz.alt < -6*u.deg, color='0.6', zorder=0)
			plt.fill_between(delta_midnight.to('hr').value, 0, 90,
							sun_altaz.alt < -12*u.deg, color='0.5', zorder=0)
			plt.fill_between(delta_midnight.to('hr').value, 0, 90,
							sun_altaz.alt < -18*u.deg, color='k', zorder=0)

			LocalTicks = np.linspace(*xboundary, 9, dtype=int)
			LocalTickLabels = ["{}".format(s).zfill(2)+'00' for s in LocalTicks%24]
			UTCTicks = LocalTicks - int(utcoffset.value)
			ax1.set_xticks(LocalTicks)
			ax1.set_xticklabels(LocalTickLabels, rotation=90, size=14)
			ax1.set_xlabel('[Local Time]', size=12)
			ax2.set_xticks(LocalTicks)
			ax2.set_xticklabels(["{}".format(s).zfill(2)+'00' for s in UTCTicks%24], rotation=90, size=14)
			ax2.set_xlabel('[UTC]', size=12)
			plt.colorbar().set_label('Azimuth [deg]')
			plt.xlim(*xboundary)
			plt.ylim(0, 90)
			ax1.set_ylabel('Altitude [deg]')
			plt.tight_layout()
			plt.grid()

			if savedirectory:
				print("Saving figure in {}\{}\{}".format(savedirectory, midnight.datetime.date(), pl_name))
				plt.savefig(os.path.join(savedirectory, 'Transit_{}{}_{}.png'.format(obsname, midnight.datetime.date(), pl_name)), dpi=240)
				plt.close()


	return MidnightTime, MidPointTime, IngressTime, EgressTime






def mdwarf_hpfneid_observability(pl_rade, Vmag=0, Jmag=0, pl_orbper=1, st_mass=1, pl_masse=None,
		  pl_orbpererr1=0.0, st_masserr1=0.0, pl_radeerr1=np.nan, st_rad=None, st_teff=None, exptime=1800, NEID_inst_precision = 0.3,
		  return_mass_only=False, simulation=True, set_seed=0):
	'''
	INPUTS:
		pl_rade = Planetary radius in Earth Radii (linear)
		pl_radeerr1 = Optional 1 sigma uncertainty in Earth Radii. Default=0.0, not used for simulation data.
		st_rad = Radius of Star in solar radii. Only required if st_teff is not given
		st_teff = Effective temperature of  star. If this is unknown enter st_rad, to use Mann 2015 scaling relation for M dwarfs.
				For FGK will use Torres 2010 relation.
		Vmag = Vmag of the star, required for NEID ETC.
		Jmag = Jmag of the star, required for the HPF ETC.
		pl_orbper = Orbital Period in days, required to calculate semi-amplitude.
		pl_orbpererr1 = Uncertainty in period.
		st_mass = Mass of Star in sol mass. Required to calculate semi-amplitude.
		st_masserr1 = Uncertainty in stellar mass.
		return_mass_only = If True, will not calculate semi amplitude ETC, and only return planetary mass.
		exptime = Exposure time for NEID (seconds). Default is 1800 seconds.
		NEID_inst_precision = NEID single visit instrument precision to add in quadrature to photon noise (Default: 30 cm/s)
		simulation = If function is being run for simulated data or real. If simulation, then will perform rejection sampling to estimate
				the mass of the input radius using MRExo, else will use direct predict function.
	OUTPUTS:
		pl_masse = Planetary mass predicted from MRExo, in Earth mass.
		pl_masseerr1 = Planetary mass uncertainty.
		st_teff = Stellar Effective temperature, either the user entered value or as estimated from Mann 2015.
		NEID_sigma = NEID single visit precision as calculated from ETC for given exptime, Vmag andst_teff.
		HPF_sigma = HPF single visit precision.
		K = Semi Amplitude of doppler signal
		K_sigma = Uncertainty in K due to uncertainty in planetary mass
		
		
	from pyastrotools.observing_tools
	Shubham Kanodia 27th April 2021

	'''

	if set_seed:
		np.random.seed(set_seed)

	qtl = np.linspace(0,1,101)

	if not st_teff:
	   st_teff =  mdwarf_teff_from_r(st_rad=st_rad)

	dataset='Mdwarf'


	if pl_masse is None:
		if simulation:
			# Random sampling
				results = predict_from_measurement(measurement=pl_rade, predict='Mass',
													use_lookup=True, qtl=qtl, dataset=dataset)
				try:
					result_dir = "C:/Users/shbhu/Documents/GitHub/mrexo/mrexo/datasets/M_dwarfs_20200520"
					output_location = os.path.join(result_dir, 'output')
					M_points = np.loadtxt(os.path.join(output_location, 'Y_points.txt'))
				except:
					result_dir = "C:/Users/shbhu/Documents/Git/mrexo/mrexo/datasets/M_dwarfs_20200520"
					output_location = os.path.join(result_dir, 'output')
					M_points = np.loadtxt(os.path.join(output_location, 'Y_points.txt'))

				cdf = np.log10(results[1])
				cdf_interp = interp1d(cdf, qtl, bounds_error=False, fill_value = "extrapolate")(M_points)

				# Conditional_plot. PDF is derivative of CDF
				y = np.diff(cdf_interp) / np.diff(M_points)
				x = M_points[:-1]

				y_interp = interp1d(x, y)


				accepted = 0
				iters=0
				while accepted == 0:
					iters+=1

					x_rand = np.random.uniform(min(x),max(x),1)
					y_rand = y_interp(x_rand)
					if y_rand == np.nan:
						break
					check_rand = np.random.uniform(0,1,1)

					# print(x_rand, y_rand, check_rand)
					if check_rand < y_rand:
						pl_masse = x_rand
						accepted = 1

					if iters==500:
						accepted = 1
						pl_masse = np.nan

				pl_masseerr1 = 0
				pl_masse = 10**pl_masse
		else:
			results = predict_from_measurement(measurement=pl_rade, measurement_sigma=pl_radeerr1, predict='Mass', use_lookup=False, qtl=qtl)
			pl_masse = results[0]
			pl_masseerr1 = results[0]-results[1][16]


		if return_mass_only:
			return pl_masse, pl_masseerr1
	else:
		pl_masseerr1 = 0

	NEID_sigma = np.sqrt(NEID_RV_prec(teff=st_teff, vmag = Vmag, exptime = exptime)**2 + NEID_inst_precision**2) # 30m NEID exposure is default
	HPF_sigma = HPF_ETC(jmag=Jmag,st_teff=st_teff-st_teff%100, exp_time=exptime)

	a = rv_magnitude_period_uncertainty(pl_masse=pl_masse , pl_masseerr1=pl_masseerr1, st_mass=st_mass, st_masserr1=st_masserr1, pl_orbper=pl_orbper/365., pl_orbpererr1=pl_orbpererr1)

	K = a.nominal_value
	K_sigma = a.std_dev

	return pl_masse, pl_masseerr1, st_teff, NEID_sigma, HPF_sigma, K, K_sigma

	
def fgk_hpfneid_observability(pl_rade, Vmag, Jmag, pl_orbper, st_mass,
		  pl_orbpererr1 = 0.0, st_masserr1=0.0, pl_radeerr1=np.nan, st_rad=None, st_teff=None, exptime=1800, NEID_inst_precision = 0.3,
		  return_mass_only=False, simulation=True, set_seed=0):
	'''
	INPUTS:
		pl_rade = Planetary radius in Earth Radii (linear)
		pl_radeerr1 = Optional 1 sigma uncertainty in Earth Radii. Default=0.0, not used for simulation data.
		st_rad = Radius of Star in solar radii. Only required if st_teff is not given
		st_teff = Effective temperature of  star. If this is unknown enter st_rad, to use Torres 2010 relation.
		Vmag = Vmag of the star, required for NEID ETC.
		Jmag = Jmag of the star, required for the HPF ETC.
		pl_orbper = Orbital Period in days, required to calculate semi-amplitude.
		pl_orbpererr1 = Uncertainty in period.
		st_mass = Mass of Star in sol mass. Required to calculate semi-amplitude.
		st_masserr1 = Uncertainty in stellar mass.
		return_mass_only = If True, will not calculate semi amplitude ETC, and only return planetary mass.
		exptime = Exposure time for NEID (seconds). Default is 1800 seconds.
		NEID_inst_precision = NEID single visit instrument precision to add in quadrature to photon noise (Default: 30 cm/s)
		simulation = If function is being run for simulated data or real. If simulation, then will perform rejection sampling to estimate
				the mass of the input radius using MRExo, else will use direct predict function.
	OUTPUTS:
		pl_masse = Planetary mass predicted from MRExo, in Earth mass.
		pl_masseerr1 = Planetary mass uncertainty.
		st_teff = Stellar Effective temperature, either the user entered value or as estimated from Mann 2015.
		NEID_sigma = NEID single visit precision as calculated from ETC for given exptime, Vmag and st_teff.
		HPF_sigma = HPF single visit precision.
		K = Semi Amplitude of doppler signal
		K_sigma = Uncertainty in K due to uncertainty in planetary mass
		
		
	from pyastrotools.observing_tools
	Shubham Kanodia 27th April 2021
		
	'''
	if set_seed:
		np.random.seed(set_seed)

	qtl = np.linspace(0,1,101)

	if not st_teff:
		st_teff =  fgk_teff_from_mr_feh(st_mass=st_mass, st_rad=st_rad, FeH=0)

	dataset='Kepler'

	if simulation:
			results = predict_from_measurement(measurement=pl_rade, predict='Mass',
												use_lookup=True, qtl=qtl, dataset=dataset)

			try:
				result_dir = "C:/Users/shbhu/Documents/GitHub/mrexo/mrexo/datasets/Kepler_Ning_etal_20170605"
				output_location = os.path.join(result_dir, 'output')
				M_points = np.loadtxt(os.path.join(output_location, 'Y_points.txt'))
			except:
				result_dir = "C:/Users/shbhu/Documents/Git/mrexo/mrexo/datasets/Kepler_Ning_etal_20170605"
				output_location = os.path.join(result_dir, 'output')
				M_points = np.loadtxt(os.path.join(output_location, 'Y_points.txt'))

			cdf = np.log10(results[1])
			cdf_interp = interp1d(cdf, qtl, bounds_error=False, fill_value = "extrapolate")(M_points)

			# Conditional_plot. PDF is derivative of CDF
			y = np.diff(cdf_interp) / np.diff(M_points)
			x = M_points[:-1]

			y_interp = interp1d(x, y)


			accepted = 0
			iters=0
			while accepted == 0:
				iters+=1

				x_rand = np.random.uniform(min(x),max(x),1)
				y_rand = y_interp(x_rand)
				if y_rand == np.nan:
					break
				check_rand = np.random.uniform(0,1,1)

				# print(x_rand, y_rand, check_rand)
				if check_rand < y_rand:
					pl_masse = x_rand
					accepted = 1

				if iters==500:
					accepted = 1
					pl_masse = np.nan

			pl_masseerr1 = 0
			pl_masse = 10**pl_masse
	else:
		results = predict_from_measurement(measurement=pl_rade, measurement_sigma=pl_radeerr1, predict='Mass', use_lookup=False, qtl=qtl)
		pl_masse = results[0]
		pl_masseerr1 = results[0]-results[1][16]

	if return_mass_only:
		return pl_masse, pl_masseerr1


	NEID_sigma = np.sqrt(NEID_RV_prec(teff = st_teff, vmag = Vmag, exptime = exptime)**2 + NEID_inst_precision**2) # 30m NEID exposure is default
	if st_teff < 5000:
		HPF_sigma = HPF_ETC(jmag=Jmag, st_teff=st_teff-st_teff%100, exp_time=exptime)
	else:
		HPF_sigma = np.nan


	a = rv_magnitude_period_uncertainty(pl_masse=pl_masse, pl_masseerr1=pl_masseerr1, st_mass=st_mass, st_masserr1=st_masserr1, pl_orbper=pl_orbper/365., pl_orbpererr1=pl_orbpererr1)

	K = a.nominal_value
	K_sigma = a.std_dev

	return pl_masse, pl_masseerr1, st_teff, NEID_sigma, HPF_sigma, K, K_sigma


def NoObservations(KoverSigma, SNR):
	'''
	Plavchan 2015, what are the number of observations required for a given SNR for K/sigma
	Eqn 4.2.5
	
		
	from pyastrotools.observing_tools
	Shubham Kanodia 27th April 2021	
	'''
	return 2*(SNR/KoverSigma)**2

def InvNoObservations(NoObservations, SNR):
	'''
	Plavchan 2015, what is the K/sigma with a given SNR and in NoObservations
	
			
	from pyastrotools.observing_tools
	Shubham Kanodia 27th April 2021

	'''
	return SNR/np.sqrt(NoObservations/2)


def QueryObjectPosition(ObsTime, 
		TargetName=None,
		RA=None, Dec=None, 
		ObsName='',  lat=None, longi=None, alt=None, 
		QueryTIC=False, QueryGaia=False, QuerySimbad=True):
	'''
	Produce an altitude - azimuth - airmass plot for the target, Sun and the Moon.

	INPUT:
		ObsTime : Astropy Time object for which to calculate (UTC time)
		obsname : Name of Observatory as defined in Astropy EarthLocation routine. Can check list by EarthLocation.get_site_names().
				If obsname is not used, then can enter lat,long,alt.
							OR
		lat : Latitude of observatory in [degrees]. North (+ve) and South (-ve).
		longi : Longitude of observatory [degrees]. East (+ve) and West (-ve).
		alt : Altitude of observatory [m].

	OUTPUT:
		targetaltaz_obsnight, sun_altaz, moon_altaz: AltAz object for the Target, Sun and Moon for given ObsTime and Observatory



	from pyastrotools.observing_tools
	Shubham Kanodia 22nd May 2021
	'''
	
	location = find_location(obsname=ObsName, lat=lat, longi=longi, alt=alt)

	observatory = Observer(location=location, name=ObsName)


	if TargetName:
		try:
			targetobj = get_stellar_data_and_mag(name=TargetName, RA=RA, Dec=Dec, QueryTIC=QueryTIC, QueryGaia=QueryGaia, QuerySimbad=QuerySimbad)
			# targetobj = Simbad.query_object(target_name)
			target = SkyCoord(ra=targetobj['ra'], dec=targetobj['dec'],unit=(u.deg, u.deg))
		except TypeError:
			print("Could not find on SIMBAD")
			target = SkyCoord(ra=RA, dec=Dec,unit=(u.hourangle, u.deg))
	else:
		target = SkyCoord(ra=RA, dec=Dec,unit=(u.hourangle, u.deg))
	print(target)

	frame_obsnight = AltAz(obstime = ObsTime, location=location)
	sun_altaz = get_sun(ObsTime).transform_to(frame_obsnight)
	targetaltaz_obsnight = target.transform_to(frame_obsnight)

	moon_obj = get_moon(ObsTime)
	moon_altaz = moon_obj.transform_to(frame_obsnight)
	moon_separation = np.min(moon_obj.separation(target)).deg
	moon_illumination = observatory.moon_illumination(ObsTime)
	
	if len(ObsTime) == 1:
		print("Object {} is at Alt = {:.2f}, Az = {:.2f}, with airmass = {:.3f}".format(TargetName, targetaltaz_obsnight.alt, targetaltaz_obsnight.az, targetaltaz_obsnight.secz))
		print("Sun is at Alt = {:.2f}, Az = {:.2f}, with airmass = {:.3f}".format(sun_altaz.alt, sun_altaz.az, sun_altaz.secz))
		print("Moon is at Alt = {:.2f}, Az = {:.2f}, with airmass = {:.3f}".format(moon_altaz.alt, moon_altaz.az, moon_altaz.secz))
		print("Moon is {:.2f} deg away from target with illumination = {:.2f} %".format(moon_separation, moon_illumination)) 

	return targetaltaz_obsnight, sun_altaz, moon_altaz
