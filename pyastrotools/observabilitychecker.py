# Generate plots for observability of targets across a semester

import os
import sys
import numpy as np
import pytz
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as pltfrom scipy.interpolate import interp1d
import astropy.units as u
from astropy.time import Time
import datetime
from astroplan import Observer
from astroplan import EclipsingSystem, FixedTarget
from astroplan import (PrimaryEclipseConstraint, is_event_observable, AtNightConstraint, AltitudeConstraint, LocalTimeConstraint,MoonSeparationConstraint)
from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.coordinates import get_sun,get_moon

from pyastrotools.observing_tools import find_location,find_utc_offset
from pyastrotools.astro_tools import get_stellar_data_and_mag

from astropy.visualization import astropy_mpl_style
plt.style.use(astropy_mpl_style)
plt.figure()

lat=None; longi=None; alt=None
obsname = 'LCO'
QueryTIC = True
QueryGaia = False
QuerySimbad = False

TargetNames = ['TIC-165227846', 'TIC-212554303', 'TIC-230982415', 'TIC-165202476',
       'TIC-142914323', 'TIC-334247565', 'TIC-8474795',
       'TIC-188043856', 'TIC-243222192', 'TIC-55441385', 'TIC-240768149']


TargetNames =['TIC-211405734', 'TIC-237405551', 'TIC-240768149', 'TIC-242567923', 'TIC-389723320', 'TIC-248047460', 'TIC-4153968', 'TIC-454847936',
	'TIC-284196024', 'TIC-306579755', 'TIC-343142609', 'TIC-326425513', 'TIC-433560866', 'TIC-179482603', 'TIC-38763923', 'TIC-394485253', 'TIC-364246668',
	'TIC-413283073', 'TIC-405636055', 'TIC-398314416', 'TIC-197979666', 'TIC-241307917', 'TIC-259574611', 'TIC-241686933', 'TIC-354776764', 'TIC-197953918',
	'TIC-206383230', 'TIC-225834368', 'TIC-141141249', 'TIC-216702551', 'TIC-151587263', 'TIC-23205270', 'TIC-148311961', 'TIC-113355522', 'TIC-41860736',
	'TIC-7586789', 'TIC-140071846', 'TIC-124307817', 'TIC-67598497', 'TIC-124988112', 'TIC-206683584', 'TIC-142064910', 'TIC-54354404', 'TIC-53548076',
	 'TIC-164249186', 'TIC-212347535', 'TIC-58464534', 'TIC-257556394', 'TIC-434482244', 'TIC-441492442']

RAs = [None]*len(TargetNames)
Decs = [None]*len(TargetNames)


QueryStartDate =  Time(datetime.datetime(2026, 10, 11), format='datetime')
QueryEndDate = Time(datetime.datetime(2027, 1, 15), format='datetime')


# QueryStartDate = Time(datetime.datetime(2024, 4, 1), format='datetime')
# QueryEndDate = Time(datetime.datetime(2024, 6, 1), format='datetime')

MinAltitude = 30
MaxAltitude = 89
MinMoonSeparation = 20

NumTargets = len(TargetNames)
DateRange = Time( np.linspace(QueryStartDate.jd, QueryEndDate.jd, round(QueryEndDate.jd - QueryStartDate.jd)+1), format='jd')

NumNights = len(DateRange)

df = pd.DataFrame()

ListofTargets = []
for t in range(NumTargets):
	if (RAs[t] is None) | (Decs[t] is None):
		TargetObj = get_stellar_data_and_mag(name=TargetNames[t], RA=None, Dec=None, QueryTIC=QueryTIC, QueryGaia=QueryGaia, QuerySimbad=QuerySimbad)
		df = df.append(TargetObj, ignore_index=True)
		Target = SkyCoord(ra=TargetObj['ra'], dec=TargetObj['dec'],unit=(u.deg, u.deg))
	else: Target = SkyCoord(ra=RAs[t], dec=Decs[t], unit=(u.deg, u.deg))
	ListofTargets.append(Target)

Location = find_location(obsname=obsname, lat=lat, longi=longi, alt=alt)

MaxAltitudeArray = np.zeros((NumTargets, NumNights))
NightlyDurationArray = np.zeros((NumTargets, NumNights))

for t in range(NumTargets):
	print(TargetNames[t])
	for i in range(NumNights):
		ObsTime = DateRange[i] # Cycle through each date

		UTCOffset, Timezone = find_utc_offset(Location, ObsTime)
		Observatory = Observer(location=Location, name="", timezone=Timezone)
		xboundary = [-8, 8]
		DeltaMidnight = np.linspace(num=50, *xboundary)*u.hour

		Midnight = Time(np.floor(ObsTime.jd) + 0.5,format='jd',scale='utc') - UTCOffset
		TimeObs = Midnight+DeltaMidnight
		FrameObsNight = AltAz(obstime=TimeObs, location=Location)
		TargetAltAz = ListofTargets[t].transform_to(FrameObsNight)

		# Moon and Sun AltAz
		SunAltAz = get_sun(TimeObs).transform_to(FrameObsNight)
		MoonObj = get_moon(TimeObs)
		MoonAltAz = MoonObj.transform_to(FrameObsNight)
		MoonSeparation = MoonObj.separation(ListofTargets[t]).deg
		MoonIllumination = Observatory.moon_illumination(ObsTime)

		SunMask = SunAltAz.alt.value < -18 # 18 degree Nautical Twilight and thereafter
		MoonMask = (MoonSeparation > MinMoonSeparation)
		AltitudeMask = (TargetAltAz.alt.value > MinAltitude) & (TargetAltAz.alt.value < MaxAltitude)
		MasterMask = MoonMask & SunMask & AltitudeMask

		if len(TargetAltAz.alt.value[MasterMask]) > 0:
			MaxAltitudeArray[t, i] = TargetAltAz.alt.value[MasterMask].max()
			NightlyDurationArray[t, i] = (TimeObs[MasterMask].jd.max() - TimeObs[MasterMask].jd.min())*24


plt.imshow(MaxAltitudeArray, aspect='auto', vmin=MinAltitude, vmax=MaxAltitude, cmap="YlGnBu")
cbar = plt.colorbar(label="Max Altitude")

plt.imshow(NightlyDurationArray, aspect='auto', vmin=0, vmax=10, cmap="YlGnBu")
cbar = plt.colorbar()
cbar.set_label("Duration in hours above {} degrees".format(MinAltitude), fontsize=30)

plt.yticks(np.arange(NumTargets), TargetNames, fontsize=10)
plt.xticks(np.arange(0, NumNights, 30), [d.datetime.strftime("%d %b %Y") for d in DateRange[np.arange(0, NumNights, 30)]], fontsize=20)
plt.xticks(rotation=45)
plt.title("GEMS @ LCO : 2026 B", size=30)
plt.tight_layout()
plt.show(block=True)



 # obs_planning_star(obstime, target_name=None, RA=None, Dec=None, obsname='', lat=None, longi=None, alt=None, title=None, utcoffset=None, timezone=None,
					# QueryTIC=False, QueryGaia=False, QuerySimbad=True)
