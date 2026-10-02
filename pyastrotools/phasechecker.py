# Script to generate airmass plots for a given orbiting system and marking phases. Can also generate .tw file for Gemini PIT

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


pl_name = "TIC-140834213 b"
RA = 76.009409
Dec = 74.502532
pl_tranmid = 2458994.0838406
pl_orbper = 8.979226879

pl_name = "TOI-7393  b"
RA = 227.359837
Dec = 55.981056
pl_tranmid = 2460369.40819655
pl_orbper =3.33708467


pl_name = "TOI-5873 b"
RA  = 324.986222
Dec = 1.29336
pl_tranmid = 2459797.48248823
pl_orbper = 0.41314288

# """

# pl_name = 'TIC-240768149 b'
# RA = 340.216226
# Dec = -17.139553
# pl_tranmid = 2458354.9636226
# pl_orbper = 0.6240740
# """

# pl_name = "TOI-5325 b"
# RA = 32.912647
# Dec = 4.192318
# pl_tranmid = 2459498.292378
# pl_orbper = 15.057596

# pl_name = "TOI-6848"
# RA = 24.418283
# Dec =-54.80939
# pl_tranmid = 2458361.1614476
# pl_orbper = 7.5957680

pl_name = "TIC-311276853 b"
RA = 247.739002
Dec = 26.106951
pl_tranmid = 2458987.47200494
pl_orbper = 5.16954679


# Run 1
QueryStartDate =  Time(datetime.datetime(2026, 5, 1), format='datetime')
QueryEndDate = Time(datetime.datetime(2026, 9, 1), format='datetime')


# QueryStartDate = Time(datetime.datetime(2023, 11, 9), format='datetime')
# QueryEndDate = Time(datetime.datetime(2023, 11, 30), format='datetime')


QueryPhases = [[0.05, 0.15], [0.20, 0.30], [0.30, 0.40], [0.40, 0.60], [0.60, 0.70], [0.70, 0.80], [0.85, 0.95]]
# QueryPhases = [[0.0,  0.999]]
# QueryPhases = [[0.001, 0.999]]
QueryPhases = [[0.64, 0.86]]

# Transit is 0
# RV min is 0.25
# RV max is 0.75

# ObsName = "keck"
# MinAltitude = 15
# MaxAltitude = 89

ObsName = "Keck"
MinMoonSeparation = 10
MinAltitude = 45
MaxAltitude = 85

ObsName = "McDonald"
MinAltitude = 45
MaxAltitude = 60

PlotDirectory = r"C:\Users\skanodia\Documents\PSU\Proposals\GeminiNorth\2023B\2023B_ObservingFiles"
# PlotDirectory = r"C:\Users\skanodia\Documents\PSU\M_dwarves\TSLs\UT23-3-010"
PlotDirectory = r"C:\Users\skanodia\Downloads"
PlotAirmass = True
PlotPhase = True



###############################
Location = find_location(obsname=ObsName)
Target = SkyCoord(ra=RA, dec=Dec,unit=(u.deg, u.deg))
plt.close("all")

for QueryPhase in QueryPhases:
	SurvivingWindows = ''
	FileName = os.path.join(PlotDirectory, '{}_{}_Phase{}_{}_{}.pdf'.format(ObsName.replace(' ', ''), pl_name.replace(' ', ''), QueryPhase[0], QueryPhase[1], QueryStartDate.isot[:-13].replace('-', '')))
	if PlotPhase: pp = PdfPages(FileName)

	# DateRange = np.linspace(QueryStartDate, QueryEndDate, round(QueryEndDate.jd - QueryStartDate.jd)+1)
	DateRange = Time(np.linspace(QueryStartDate.jd, QueryEndDate.jd, round(QueryEndDate.jd - QueryStartDate.jd)+1), format='jd')
	for i in range(round(QueryEndDate.jd - QueryStartDate.jd)+1):
		ObsTime = DateRange[i] # Cycle thrpough each date

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
		PhaseMask = (PhaseObs > QueryPhase[0]) &  (PhaseObs < QueryPhase[1])

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
		if (Interval.value*24*60) < 30: continue # Too short a visit.


		# SurvivingWindows += SurvivingTimes[0].iso[:-6]+'00' + '\t' + "{:02d}:{:02d}\n".format(int(round(Interval.value*24//1)), int(round((Interval.value*24%1)*60)))
		SurvivingWindows += SurvivingTimes[0].isot[:-6]+'00' + ' \t ' + "{:02d}:{:02d} \t {:.02f} \n".format(int(round(Interval.value*24//1)), int(round((Interval.value*24%1)*60)), PhaseObs[np.argmax(TargetAltAz.alt.value)])

		if PlotAirmass:

			fig, ax1 = plt.subplots((1), figsize=(9, 6))
			# plt.grid(False)

			# ax1 = fig.add_subplot(111)
			ax2 = ax1.twiny()
			ax1.grid(False); #ax2.grid(False)

			s = ax1.scatter(DeltaMidnight.value, TargetAltAz.alt.value,c=TargetAltAz.az.value, s=15, marker='.',
						cmap='viridis', label=pl_name)

			# Plot the Sun, Moon and Airmass lines
			ax1.plot(DeltaMidnight.value, SunAltAz.alt, color='orange', label='Sun')
			ax1.plot(DeltaMidnight.value, MoonAltAz.alt, color=[0.75]*3, ls='--', label='Moon')


			if PlotPhase:
				EastMask = TargetAltAz.az.value < 180
				WestMask = TargetAltAz.az.value > 180
				_ = [plt.text(DeltaMidnight.value[EastMask & MasterMask][j] - 1.0, TargetAltAz.alt.value[EastMask & MasterMask][j], "{:.03f}".format(PhaseObs[EastMask & MasterMask][j]), c='white') for j in range(0, len(PhaseObs[EastMask & MasterMask]), 8)]
				_ = [plt.text(DeltaMidnight.value[WestMask & MasterMask][j] + 0.8, TargetAltAz.alt.value[WestMask & MasterMask][j], "{:.03f}".format(PhaseObs[WestMask & MasterMask][j]), c='white') for j in range(0, len(PhaseObs[WestMask & MasterMask]), 8)]

			if ObsName == 'McDonald':
				ax1.hlines(50, *xboundary, 'r', 'dashed')
				ax1.text(-4, 51, 'Altitude  50', c='w', fontsize=12)
				ax1.hlines(60, *xboundary, 'r', 'dashed')
				ax1.text(-4, 61, 'Altitude  60', c='w', fontsize=12)
			else:
				ax1.hlines(48, *xboundary, 'r', 'dashed')
				ax1.text(-4, 49, 'Airmass 1.5', c='w', fontsize=15)
				ax1.hlines(30, *xboundary, 'r', 'dashed')
				ax1.text(-4, 31, 'Airmass 2.0', c='w', fontsize=15)
			ax1.legend(loc='upper left')

			ax1.set_title('{}; {}; ObsName = {}\nLocal Midnight at UTC {}. JD {}. \nMoon Illum = {}%. Min. Moon Sep = {} deg\nOrbital Period = {:.2f} d, Phase at Max Alt = {:.03f}'.format(pl_name, Target.to_string('hmsdms'), ObsName, Midnight.datetime,np.round(Midnight.jd,2),
																												np.round(np.max(MoonIllumination)*100, 0), np.round(np.min(MoonSeparation), 2), pl_orbper, PhaseObs[np.argmax(TargetAltAz.alt.value)]), fontsize=15, pad = 5)
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
			# plt.colorbar(s).set_label('Azimuth [deg]', fontsize=20)
			ax1.set_xlim(*xboundary)
			ax1.set_ylim(0, 90)
			ax1.set_ylabel('Altitude [deg]', fontsize=20)
			plt.tight_layout()
			# plt.grid()
			# plt.show(block=False)
			pp.savefig(fig)
		# break
	pp.close()
	plt.close("all")
	print(QueryPhase)
	print(SurvivingWindows)

	if len(SurvivingWindows)  > 0:
		with open(os.path.join(PlotDirectory, FileName[:-4]+'.tw'), 'w') as f:
			f.write(SurvivingWindows)
	else:
		os.remove(FileName)
