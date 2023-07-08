import os
import sys
import numpy as np
import pytz
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
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

from pyastrotools.observing_tools import find_location, find_utc_offset


pl_name = 'TOI-5176 b'
RA = 135.019493
Dec = 13.273706
pl_tranmid = 2459553.71670012
pl_orbper = 20.35829214

QueryStartDate = start_query = Time(datetime.datetime(2023, 12, 1), format='datetime')
QueryEndDate = start_query = Time(datetime.datetime(2023, 12, 31), format='datetime')
QueryPhase = [[0.20, 0.30]]

ObsName = "Gemini North"
MinMoonSeparation = 25
MinAltitude = 30
MaxAltitude = 90
PlotDirectory = r"C:\Users\skanodia\Documents\PSU\Proposals\GeminiNorth\2023B"
PlotAirmass = True
PlotPhase = True


###############################


SurvivingWindows = ''

Location = find_location(obsname=ObsName)
Target = SkyCoord(ra=RA, dec=Dec,unit=(u.deg, u.deg))

if PlotPhase: pp = PdfPages(os.path.join(PlotDirectory, '{}_{}_Phase{}_{}.pdf'.format(ObsName.replace(' ', ''), pl_name.replace(' ', ''), QueryPhase[0][0], QueryPhase[0][1])))

for i in range(round(QueryEndDate.jd - QueryStartDate.jd)):
	DateRange = np.linspace(QueryStartDate, QueryEndDate, round(QueryEndDate.jd - QueryStartDate.jd))
	ObsTime = DateRange[i] # Cycle through each date

	UTCOffset, Timezone = find_utc_offset(Location, ObsTime)
	Observatory = Observer(location=Location, name="", timezone=Timezone)

	xboundary = [-8, 8]
	DeltaMidnight = np.linspace(num=200, *xboundary)*u.hour

	Midnight = Time(np.floor(ObsTime.jd) + 0.5,format='jd',scale='utc') - UTCOffset
	TimeObs = Midnight+DeltaMidnight
	FrameObsNight = AltAz(obstime=TimeObs, location=Location)
	TargetAltAz = Target.transform_to(FrameObsNight)

	# Convert time to phase from 0 to 1 with transit at 0
	PhaseObs = (TimeObs.jd - pl_tranmid)%pl_orbper / pl_orbper

	# Select phases that are amenable
	PhaseMask = (PhaseObs > QueryPhase[0][0]) &  (PhaseObs < QueryPhase[0][1])
	for p in QueryPhase[1:]:
		PhaseMask = PhaseMask | ((PhaseObs > p[0]) &  (PhaseObs < p[1]))

	# Moon and Sun AltAz
	SunAltAz = get_sun(TimeObs).transform_to(FrameObsNight)
	MoonObj = get_moon(TimeObs)
	MoonAltAz = MoonObj.transform_to(FrameObsNight)
	MoonSeparation = MoonObj.separation(Target).deg
	MoonIllumination = Observatory.moon_illumination(ObsTime)

	SunMask = SunAltAz.alt.value < -18 # 18 degree Nautical Twilight and thereafter
	MoonMask = MoonSeparation > MinMoonSeparation
	AltitudeMask = (TargetAltAz.alt.value > MinAltitude) & (TargetAltAz.alt.value < MaxAltitude)

	MasterMask = PhaseMask & MoonMask & SunMask & AltitudeMask

	SurvivingTimes = TimeObs[MasterMask]

	if len(SurvivingTimes) == 0: continue # Requirements not met. Skip
	Interval = SurvivingTimes[-1] - SurvivingTimes[0]

	SurvivingWindows += SurvivingTimes[0].iso[:-6]+'00' + '\t' + "{:02d}:{:02d}\n".format(int(round(Interval.value*24//1)), int(round((Interval.value*24%1)*60)))

	if PlotAirmass:

		fig, ax1 = plt.subplots((1), figsize=(9, 6))
		# plt.grid(False)

		# ax1 = fig.add_subplot(111)
		ax2 = ax1.twiny()
		ax1.grid(False); #ax2.grid(False)

		s = ax1.scatter(DeltaMidnight.value, TargetAltAz.alt,c=TargetAltAz.az, s=15, marker='.',
					cmap='viridis', label=pl_name)

		# Plot the Sun, Moon and Airmass lines
		ax1.plot(DeltaMidnight.value, SunAltAz.alt, color='orange', label='Sun')
		ax1.plot(DeltaMidnight.value, MoonAltAz.alt, color=[0.75]*3, ls='--', label='Moon')


		if PlotPhase:
			EastMask = TargetAltAz.az.value < 180
			WestMask = TargetAltAz.az.value > 180
			_ = [plt.text(DeltaMidnight.value[EastMask & MasterMask][j] - 1.0, TargetAltAz.alt.value[EastMask & MasterMask][j], "{:.03f}".format(PhaseObs[EastMask & MasterMask][j]), c='white') for j in range(0, len(PhaseObs[EastMask & MasterMask]), 8)]
			_ = [plt.text(DeltaMidnight.value[WestMask & MasterMask][j] + 0.8, TargetAltAz.alt.value[WestMask & MasterMask][j], "{:.03f}".format(PhaseObs[WestMask & MasterMask][j]), c='white') for j in range(0, len(PhaseObs[WestMask & MasterMask]), 8)]

		ax1.hlines(48, *xboundary, 'r', 'dashed')
		ax1.text(-4, 49, 'Airmass 1.5', c='w', fontsize=15)
		ax1.hlines(30, *xboundary, 'r', 'dashed')
		ax1.text(-4, 31, 'Airmass 2.0', c='w', fontsize=15)
		ax1.legend(loc='upper left')

		ax1.set_title('{}; {}; ObsName = {}\nLocal Midnight at UTC {}. JD {}. \nMoon Illum = {}%. Min. Moon Sep = {} deg: Phase at Max Alt = {:.03f}'.format(pl_name, target.to_string('hmsdms'), ObsName, Midnight.datetime,np.round(Midnight.jd,2),
																											np.round(np.max(MoonIllumination)*100, 0), np.round(np.min(MoonSeparation), 2), PhaseObs[np.argmax(TargetAltAz.alt.value)]), fontsize=15, pad = 5)
		ax1.fill_between(DeltaMidnight.to('hr').value, 0, 90,
						SunAltAz.alt < -0*u.deg, color='0.7', zorder=0)
		ax1.fill_between(DeltaMidnight.to('hr').value, 0, 90,
						SunAltAz.alt < -6*u.deg, color='0.6', zorder=0)
		ax1.fill_between(DeltaMidnight.to('hr').value, 0, 90,
						SunAltAz.alt < -12*u.deg, color='0.5', zorder=0)
		ax1.fill_between(DeltaMidnight.to('hr').value, 0, 90,
						SunAltAz.alt < -18*u.deg, color='k', zorder=0)
		LocalTicks = np.linspace(*xboundary, 9, dtype=int)
		LocalTickLabels = ["{}".format(s).zfill(2)+'00' for s in LocalTicks%24]
		UTCTicks = LocalTicks - int(UTCOffset.value)
		ax1.set_xticks(LocalTicks)
		ax1.set_xticklabels(LocalTickLabels, rotation=90, size=15)
		ax1.set_yticks([15, 30, 60, 75, 90])
		ax1.set_yticklabels([15, 30, 60, 75, 90], fontsize=15)
		ax1.set_xlabel('[Local Time]', size=12)
		ax2.set_xticks(LocalTicks)
		ax2.set_xticklabels(["{}".format(s).zfill(2)+'00' for s in UTCTicks%24], rotation=90, size=15)
		ax2.set_xlabel('[UTC]', size=12)
		plt.colorbar(s).set_label('Azimuth [deg]', fontsize=20)
		ax1.set_xlim(*xboundary)
		ax1.set_ylim(0, 90)
		ax1.set_ylabel('Altitude [deg]', fontsize=20)
		plt.tight_layout()
		# plt.grid()
		# plt.show(block=False)
		pp.savefig(fig)
	# break
pp.close()
print(SurvivingWindows)


with open(os.path.join(PlotDirectory, '{}_{}_Phase{}_{}.tw'.format(ObsName.replace(' ', ''), pl_name.replace(' ', ''), QueryPhase[0][0], QueryPhase[0][1])), 'w') as f:
	f.write(SurvivingWindows)
