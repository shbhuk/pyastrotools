import pandas as pd
import re
from astroquery.gaia import Gaia
from pyastrotools.astro_tools import _QueryTIC, _QueryGaia

InFile =  r"C:\Users\skanodia\Downloads\TESS Planning - MASTER.csv"

df = pd.read_csv(InFile)
df = df.dropna(subset=["Rank", "TIC Name"])

TIC_IDs = df['TIC Name']
GaiaDR2_IDs = np.zeros(len(TIC_IDs), dtype='int64')

for i,t in enumerate(TIC_IDs):
    GaiaDR2_IDs[i] = int(_QueryTIC('TIC '+str(t))['GAIA'][0])


df['GaiaDR2'] = GaiaDR2_IDs
gaia_ids = ",".join(str(int(x)) for x in GaiaDR2_IDs)

adql = f"""
SELECT
    source_id,
    ra,
    dec,
    ruwe,
    astrometric_excess_noise,
    astrometric_excess_noise_sig,
    parallax,
    parallax_error,
    pmra,
    pmra_error,
    pmdec,
    pmdec_error,
    radial_velocity,        -- Radial Velocity
    radial_velocity_error    -- Radial Velocity Error
FROM
    gaiadr3.gaia_source
WHERE
    source_id IN ({gaia_ids}
    )

"""


print("Submitting query to Gaia Archive...")
job = Gaia.launch_job_async(adql)
gaia_res = job.get_results().to_pandas()



# --- Merge
df = df.merge(
    gaia_res,
    how="left",
    left_on="GaiaDR2",
    right_on="SOURCE_ID"
).drop(columns=["SOURCE_ID"])


df.to_csv( r"C:\Users\skanodia\Downloads\TESS Planning - MASTER_Gaia.csv", index=False)
