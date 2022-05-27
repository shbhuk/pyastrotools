import os
from pyastrotools.general_tools import download_file_check_staleness, compactString
from pyastrotools.observing_tools import find_observatory, find_utc_offset
import datetime
import matplotlib.pyplot as plt
import astropy.units as u
import numpy as np
from astropy.table import Table,join,Column
import pandas as pd
from astropy.time import Time
from astropy.coordinates import SkyCoord,AltAz
try:
	from astroplan import EclipsingSystem, Observer, FixedTarget
	from astroplan import (PrimaryEclipseConstraint, is_event_observable, AtNightConstraint, AltitudeConstraint, LocalTimeConstraint,MoonSeparationConstraint)
except:
	print('Cannot import Astroplan')
from astroquery.simbad import Simbad
import math
import pytz





class NEA(object):
	def __init__(self, DownloadPS=False):
		"""
		DownloadPS: If True, will download the entire Planetary Systems CSV, this includes multiple rows for each system. Set default_flag=1 to pick default (smaller csv).
		If False, will download the Planetary Systems Composite Parameters, which consists of composite parameters from multiple sources for each planet (larger csv).
		
		"""
		# Download / Load the NASA Exoplanet Archive table and initialize
		location = os.path.dirname(__file__)
		save_file = os.path.join(location,'nasa_archive_exoplanets.csv')
		# url = 'https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=exoplanets&select=*'
		
		if DownloadPS:
			url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+ps&format=csv"
		else:
			url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+pscomppars&format=csv"

		_ = download_file_check_staleness(url=url, time_tolerance=14, save_file=save_file, file_name = 'NASA Exoplanet Archive')


		## Read into Astropy Table ##
		self.archive = pd.read_csv(save_file)
		
		if DownloadPS:
			self.archive = self.archive[self.archive['default_flag'] == 1]
		
		#Exoplanet_archive.mask = np.nan
		print('Reading the NASA Exoplanet Archive file into Table Object')
		self.colnames = self.archive.columns
		pl_hosts = np.array(self.archive['hostname'])
		pl_letter = np.array(self.archive['pl_letter'])
		archive_planets = np.array([compactString(pl_hosts[i]+pl_letter[i]) for i in range(0,len(pl_hosts))])
		self.archive['Planet_Name'] = archive_planets


	def query_planet(self,pl_name):
		'''
		Find planet when given planet name
		INPUT:
			pl_name : Planet name [String]. Eg. 'Kepler 225b'

		'''
		index = np.where(self.archive['Planet_Name'] == md.compactString(pl_name))[0]
		return self.archive[index]

	def transiting_planet(self,pl_name):
		'''
		Load transiting planet object as defined here -
		https://astroplan.readthedocs.io/en/latest/tutorials/periodic.html
		INPUT:
			pl_name : Planet name [String]. Eg. 'Kepler 225b'

		'''

		planet = self.query_planet(pl_name)
		epoch = Time(planet['pl_tranmid'], format = 'jd')
		transit_duration = planet['pl_trandur'] * u.d
		orb_per = planet['pl_orbper'] * u.d

		planet_transit_sys = EclipsingSystem(primary_eclipse_time = epoch, orbital_period = orb_per, duration = transit_duration)
		return planet_transit_sys

	def find_next_midtransit(self,pl_name,start_query,stop_query='',n=1):
		'''
		Load transiting planet as defined here -
		https://astroplan.readthedocs.io/en/latest/tutorials/periodic.html
		Find the next n mid transit JDs
		INPUT:
			pl_name : Planet name [String]. Eg. 'Kepler 225b'
			start_query : Astropy Time object after which to look for transits. In UTC
			stop_query : Astropy Time object until which to look for transits. In UTC
			n : Find 'n' transits

		'''
		transit_sys = self.transiting_planet(pl_name)

		if stop_query:
			mid_transit = transit_sys.next_primary_eclipse_time(start_query,1000)
			transits_in_range = mid_transit[np.where(mid_transit < stop_query)[0]]
			return transits_in_range
		else:
			mid_transit = transit_sys.next_primary_eclipse_time(start_query,n)
			return mid_transit


	def find_next_fulltransit(self, pl_name, start_query, stop_query='', n=1):
		'''
		Load transiting planet as defined here -
		https://astroplan.readthedocs.io/en/latest/tutorials/periodic.html
		Find the next n ingress-egress JDs (full transit) for transits
		INPUT:
			pl_name : Planet name [String]. Eg. 'Kepler 225b'
			start_query : Astropy Time object after which to look for transits. In UTC
			stop_query : Astropy Time object until which to look for transits. In UTC
			n : Find 'n' transits
		OUTPUT:
			Transit ingress and egress in UTC
		'''
		transit_sys = self.transiting_planet(pl_name)
		if stop_query:
			ing_egr = transit_sys.next_primary_ingress_egress_time(start_query,1000)
			result = ing_egr[np.where(ing_egr < stop_query)[0]][::2]
			if np.size(result)==0:
				result = transit_sys.next_primary_ingress_egress_time(start_query,1)

		else:
			result = transit_sys.next_primary_ingress_egress_time(start_query,n)
		return result

	def find_next_fulltransit_observable(self, pl_name, observatory, start_query, stop_query='', n = 1,
									 min_local_time=datetime.time(19,0), max_local_time = datetime.time(6,0),
									 min_altitude = 0, min_moon_sep = 0):
		'''
		Load transiting planet as defined here -
		https://astroplan.readthedocs.io/en/latest/tutorials/periodic.html
		Find the next n ingress-egress JDs for transits.
		Also specify constraints to check if full transit is observable
		INPUT:
			pl_name : Planet name [String]. Eg. 'Kepler 225b'
			observatory : Astroplan observatory object.

						Example: observatory = Observer.at_site('APO', timezone='US/Mountain')
								 observatory = Observer(longitude=-155.4761*u.deg, latitude=19.825*u.deg,
										elevation=0*u.m, name="Subaru", timezone="US/Hawaii")

			start_query : Astropy Time object after which to look for transits. In UTC
			stop_query : Astropy Time object until which to look for transits. In UTC
			n : Find 'n' transits
			min_local_time : Minimum local (observatory) time for observation. Default is 7pm. Example: datetime.time(19,0)
			max_local_time : Maximum local (observatory) time for observation. Default is 12 midnight. Example: datetime.time(0,0)
			min_altitude : Minimum altitude above the horizon for observation. [Degrees]
			min_moon_sep : Minimum separation from the moon. [Degrees]

			Output:
				IN UTC

		'''

		planet = self.query_planet(pl_name)
		# Initialize FixedTarget object from Astroplan with RA Dec from the Archive Table
		target = FixedTarget(coord=SkyCoord(ra=planet['ra']*u.deg, dec=planet['dec']*u.deg), name=pl_name)
		# Find the next n  Full Transit times
		times = self.find_next_fulltransit(pl_name = pl_name, start_query = start_query, stop_query = stop_query, n=n)

		constraints = [AltitudeConstraint(min=min_altitude*u.deg), LocalTimeConstraint(min=min_local_time, max=max_local_time),MoonSeparationConstraint(min = min_moon_sep * u.deg)]
		#constraints = [AtNightConstraint.twilight_astronomical(), AltitudeConstraint(min=min_altitude*u.deg), LocalTimeConstraint(min=min_local_time, max=max_local_time),MoonSeparationConstraint(min = min_moon_sep * u.deg)]

		results = []

		for time in times:
			if is_event_observable(constraints,observatory,target,time[0]) and is_event_observable(constraints,observatory,target,time[1]):
				results.append(time)

		# If cannot find a transit that satisfies the conditions between start and stop period, look at next 50 transits.
		if np.size(results)==0:
			c=0
			print('Could not find transit between start and stop period, finding next earliest available transit')
			while np.size(results)==0:
				time = self.find_next_fulltransit(pl_name=pl_name, start_query=time[1], n=1)[0]
				c+=1
				print(c,time)
				if is_event_observable(constraints, observatory, target, time[0]) and is_event_observable(constraints, observatory, target, time[1]):
					results.append(time)
				if c==10:
					break
		return results

	def find_next_midtransit_observable(self, pl_name, observatory, start_query, stop_query='', n=1,
									 min_local_time=datetime.time(19,0), max_local_time = datetime.time(6,0),
									 min_altitude=30, min_moon_sep=25):
		'''
		Load transiting planet as defined here -
		https://astroplan.readthedocs.io/en/latest/tutorials/periodic.html
		Find the next n ingress-egress JDs for transits.
		Also specify constraints to check if mid transit is observable
		INPUT:
			pl_name : Planet name [String]. Eg. 'Kepler 225b'
			observatory : Astroplan observatory object.

						Example: observatory = Observer.at_site('APO', timezone='US/Mountain')
								 observatory = Observer(longitude=-155.4761*u.deg, latitude=19.825*u.deg,
										elevation=0*u.m, name="Subaru", timezone="US/Hawaii")

			start_query : Astropy Time object after which to look for transits. In UTC
			stop_query : Astropy Time object until which to look for transits. In UTC
			n : Find 'n' transits
			min_local_time : Minimum local (observatory) time for observation. Default is 7pm. Example: datetime.time(19,0)
			max_local_time : Maximum local (observatory) time for observation. Default is 12 midnight. Example: datetime.time(0,0)
			min_altitude : Minimum altitude above the horizon for observation. [Degrees]
			min_moon_sep : Minimum separation from the moon. [Degrees]

		OUTPUT:
		Ingress, Egress JD times
		'''

		planet = self.query_planet(pl_name)
		# Initialize FixedTarget object from Astroplan with RA Dec from the Archive Table
		target = FixedTarget(coord=SkyCoord(ra=planet['ra']*u.deg, dec=planet['dec']*u.deg), name=pl_name)
		# Find the next n Mid Transit times
		times = self.find_next_midtransit(pl_name = pl_name, start_query = start_query, stop_query = stop_query, n=n)

		#constraints = [AtNightConstraint.twilight_astronomical(), AltitudeConstraint(min=min_altitude*u.deg), LocalTimeConstraint(min=min_local_time, max=max_local_time),MoonSeparationConstraint(min = min_moon_sep * u.deg)]
		constraints = [AltitudeConstraint(min=min_altitude*u.deg), LocalTimeConstraint(min=min_local_time, max=max_local_time),
						MoonSeparationConstraint(min = min_moon_sep * u.deg)]
		results = []

		for time in times:
			if is_event_observable(constraints,observatory,target,time):
				results.append(time)

		# If cannot find a transit that satisfies the conditions between start and stop period, look at next 50 transits.
		if np.size(results)==0:
			c=0
			print('Could not find transit between start and stop period, finding next earliest available transit')
			while np.size(results)==0:
				time = self.find_next_midtransit(pl_name = pl_name, start_query = time,n=1)
				c+=1
				if is_event_observable(constraints,observatory,target,time):
					results.append(time)
				if c==50:
					break


		return results



	def obs_planning_transit(self,pl_name,start_query = '',stop_query = '', obsname='', lat=0., longi=0., alt=0.):
		'''
		Produce an altitude - azimuth - airmass plot for the planet host, Sun and the Moon.

		INPUT:
			pl_name : Planet name [String]. Eg. 'Kepler 225b'
			start_query : Astropy Time object after which to look for transits
			stop_query : Astropy Time object until which to look for transits.
						 Will plot the first transit in this interval.
			obsname : Name of Observatory as defined in Astropy EarthLocation routine. Can check list by EarthLocation.get_site_names().
					If obsname is not used, then can enter lat,long,alt.
								OR
			lat : Latitude of observatory in [degrees]. North (+ve) and South (-ve).
			longi : Longitude of observatory [degrees]. East (+ve) and West (-ve).
			alt : Altitude of observatory [m].

		'''
		from astropy.visualization import astropy_mpl_style
		from astropy.coordinates import get_sun,get_moon
		plt.style.use(astropy_mpl_style)
		plt.figure()

		observatory, utcoffset = find_observatory(obstime=start_query, obsname=obsname, lat=lat, longi=longi, alt=alt)
		print('Using {} as the UTC offset'.format(utcoffset))


		planet = self.query_planet(pl_name)
		target = SkyCoord(ra=planet['ra']*u.deg, dec=planet['dec']*u.deg)

		ingress,egress = self.find_next_fulltransit_observable(pl_name=pl_name,start_query=start_query,stop_query=stop_query, observatory=observatory)[0]
		midpoint = Time((ingress.value + egress.value)/2,format='jd',scale = 'utc')
		midpoint_err = (np.sum(np.abs([planet['pl_tranmiderr1'],planet['pl_tranmiderr2']]))/2)*24*60 # Average midpoint error in minutes

		# Find the alt,az coordinates of target at 100 times evenly spaced
		midnight = Time(np.floor(ingress.value - 0.3) + 0.5,format='jd') - utcoffset

		# Establish boundaries around transit to plot curve
		xboundary = []
		'''
		if (ingress.jd - midnight.jd)*24-6 < -12:
			xboundary.append((ingress.jd - midnight.jd)*24-6)
		else:
			xboundary.append(-12)
		'''
		xboundary = [-12]
		if (egress.jd - midnight.jd)*24+6 > 12:
			xboundary.append((egress.jd - midnight.jd)*24+6)
		else:
			xboundary.append(+12)
		#xboundary = [(ingress.jd - midnight.jd)*24-6,(egress.jd - midnight.jd)*24+6]
		print(ingress, egress, midnight, xboundary)


		delta_midnight = np.linspace(-12,12, 200)*u.hour
		times_obs = midnight+delta_midnight
		frame_obsnight = AltAz(obstime=times_obs, location=observatory.location)
		targetaltaz_obsnight = target.transform_to(frame_obsnight)

		# Mark ingress, egress and transit midpoint
		transit_time = Time(np.linspace(ingress.jd,egress.jd,50),format='jd',scale='utc')
		tmp_frame = AltAz(obstime=transit_time,location=observatory.location)
		tmp_C = target.transform_to(tmp_frame)
		slope_in = math.atan(np.median(np.diff(tmp_C[0:10].alt.value)) / np.median(np.diff(transit_time[0:10].jd))) * 180/np.pi
		slope_eg = math.atan(np.median(np.diff(tmp_C[-10:].alt.value)) / np.median(np.diff(transit_time[-10:].jd))) * 180/np.pi

		plt.plot(((ingress.jd - midnight.jd)*24)%24*u.hour, tmp_C[0].alt, marker=(3,0,-slope_in), color='r', markersize = 10)
		plt.plot(((egress.jd - midnight.jd)*24)%24*u.hour,  tmp_C[-1].alt, marker=(3,0,slope_eg), color='r', markersize = 10)
		plt.plot(((midpoint.jd - midnight.jd)*24)%24*u.hour,tmp_C[25].alt, marker='o', color='r', markersize = 10)

		plt.scatter(delta_midnight.value%24, targetaltaz_obsnight.alt,c = targetaltaz_obsnight.az, s = 15,marker = '.',
					cmap = 'viridis')

		sun_altaz = get_sun(times_obs).transform_to(frame_obsnight)
		moon_altaz = get_moon(times_obs).transform_to(frame_obsnight)

		plt.plot(delta_midnight.value, sun_altaz.alt, color='orange', label='Sun')
		plt.plot(delta_midnight.value, moon_altaz.alt, color=[0.75]*3, ls='--', label='Moon')
		#plt.title('{} -  {} \n Stellar Teff : {} K \n Transit midpoint = UTC {} +- {} min \n JD {}'.format(pl_name,obsname, str(planet['st_teff'][0]),str(midpoint.datetime - datetime.timedelta(microseconds = midpoint.datetime.microsecond)),str(midpoint_err),str(midpoint.jd)),fontsize=15, pad = 5)
		plt.title('{} -  {}. Stellar Teff : {} K \n Transit midpoint = UTC {} \n Transit midpoint = JD {}'.format(pl_name,obsname, str(planet['st_teff'][0]),
					str(midpoint.datetime - datetime.timedelta(microseconds = midpoint.datetime.microsecond)),str(midpoint.jd)),fontsize=15, pad = 5)
		plt.fill_between(delta_midnight.to('hr').value, 0, 90,
						sun_altaz.alt < -0*u.deg, color='0.7', zorder=0)
		plt.fill_between(delta_midnight.to('hr').value, 0, 90,
						sun_altaz.alt < -6*u.deg, color='0.6', zorder=0)
		plt.fill_between(delta_midnight.to('hr').value, 0, 90,
						sun_altaz.alt < -12*u.deg, color='0.5', zorder=0)
		plt.fill_between(delta_midnight.to('hr').value, 0, 90,
						sun_altaz.alt < -18*u.deg, color='k', zorder=0)

		plt.colorbar().set_label('Azimuth [deg]')
		plt.legend(loc='upper left')
		plt.xlim(*xboundary)
		#plt.xticks(np.arange(*xboundary,2))
		plt.ylim(0, 90)
		plt.xlabel('Hours from Local Midnight')
		plt.ylabel('Altitude [deg]')
		plt.show()

		return midnight.datetime.date()









"""
	def obs_planning_planethost(self,pl_name,obstime,obsname='', lat=0., longi=0., alt=0.):
		'''
		Produce an altitude - azimuth - airmass plot for the planet host star, Sun and the Moon.

		INPUT:
			pl_name : Planet name [String]. Eg. 'Kepler 225b'
			obstime : Astropy Time object. Defined in UTC scale.
			obsname : Name of Observatory as defined in Astropy EarthLocation routine. Can check list by EarthLocation.get_site_names().
					If obsname is not used, then can enter lat,long,alt.
								OR
			lat : Latitude of observatory in [degrees]. North (+ve) and South (-ve).
			longi : Longitude of observatory [degrees]. East (+ve) and West (-ve).
			alt : Altitude of observatory [m].

		'''
		from astropy.visualization import astropy_mpl_style
		from astropy.coordinates import get_sun,get_moon
		plt.style.use(astropy_mpl_style)
		plt.figure()

		xboundary = [-6,18]

		location = find_location(obsname=obsname, lat=lat, longi=longi, alt=alt)

		utcoffset = find_utc_offset(location, obstime) * u.hour
		print('Using {} as the UTC offset'.format(utcoffset))

		# Find the alt,az coordinates of target at 100 times evenly spaced

		midnight = Time(np.floor(obstime.jd) + 0.5,format='jd') - utcoffset
		delta_midnight = np.linspace(*xboundary, 100)*u.hour
		times_obs = midnight+delta_midnight
		frame_obsnight = AltAz(obstime = times_obs,location = observatory)
		markers = ['.','D','o','s','*','p']

		if np.size(pl_name) > 1:
			for i in range(0,len(pl_name)):
				p = pl_name[i]
				planet = self.query_planet(p)
				target = SkyCoord(ra=planet['ra']*u.deg, dec=planet['dec']*u.deg)
				targetaltaz_obsnight = target.transform_to(frame_obsnight)
				plt.scatter(delta_midnight, targetaltaz_obsnight.alt,c=targetaltaz_obsnight.az, label= p, s=15,marker = markers[i],
						cmap='viridis')
		else:
			planet = self.query_planet(pl_name)
			target = SkyCoord(ra=planet['ra']*u.deg, dec=planet['dec']*u.deg)
			targetaltaz_obsnight = target.transform_to(frame_obsnight)
			plt.scatter(delta_midnight, targetaltaz_obsnight.alt,c=targetaltaz_obsnight.az, label= pl_name, s=15,marker = '.',
						cmap='viridis')



		sun_altaz = get_sun(times_obs).transform_to(frame_obsnight)
		moon_altaz = get_moon(times_obs).transform_to(frame_obsnight)


		plt.plot(delta_midnight, sun_altaz.alt, color='r', label='Sun')
		plt.plot(delta_midnight, moon_altaz.alt, color=[0.75]*3, ls='--', label='Moon')
		#plt.scatter(delta_midnight, moon_altaz.alt,c=moon_altaz.az, marker = 'D', label= 'Moon', cmap='viridis')
		#plt.scatter(delta_midnight, sun_altaz.alt,c=sun_altaz.az,marker= '_',label= 'Sun', cmap='viridis',s=50)

		plt.fill_between(delta_midnight.to('hr').value, 0, 90,
						sun_altaz.alt < -0*u.deg, color='0.7', zorder=0)
		plt.fill_between(delta_midnight.to('hr').value, 0, 90,
						sun_altaz.alt < -6*u.deg, color='0.6', zorder=0)
		plt.fill_between(delta_midnight.to('hr').value, 0, 90,
						sun_altaz.alt < -12*u.deg, color='0.5', zorder=0)
		plt.fill_between(delta_midnight.to('hr').value, 0, 90,
						sun_altaz.alt < -18*u.deg, color='k', zorder=0)

		plt.colorbar().set_label('Azimuth [deg]')
		plt.legend(loc='upper left')
		plt.xlim(*xboundary)
		plt.xticks(np.arange(*xboundary,2))
		plt.ylim(0, 90)
		plt.xlabel('Hours from Local Midnight')
		plt.ylabel('Altitude [deg]')
		plt.show()

		return 1


"""








'''
def find_transiting_planets(pl_name,start_JD,end_JD,obs_lat='',obs_lon='',phase='primary'):

	if obs_lat:
		base_url = 'https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TransitView/nph-visibletbls?dataset=transits&sname={}&getParams&bJD={}&eJD={}&lat={}&lon={}&phase={}'.format(str(pl_name),str(start_JD),str(end_JD),str(obs_lat),str(obs_lon),str(phase))
	else:
		base_url = 'https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TransitView/nph-visibletbls?dataset=transits&sname={}&getParams&bJD={}&eJD={}&phase={}'.format(str(pl_name),str(start_JD),str(end_JD),str(phase))


	webbrowser.get('windows-default').open_new(base_url)
	return base_url

'''
