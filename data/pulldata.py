import fastf1
from pathlib import Path
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from config import TARGET_YEAR

CACHE_DIR = ROOT / 'cache'
CACHE_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = ROOT / 'data'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))

LEGACY_LAST_YEAR = 2021
LEGACY_CORNERS = 23
REFERENCE_YEAR = 2018

if TARGET_YEAR == 2021 or TARGET_YEAR == 2020:
    exit()
else:
    pass
def lap_length(telemetry_df, n=7):
   
    first = telemetry_df['Driver'].unique()[:n]
    return telemetry_df[telemetry_df['Driver'].isin(first)].groupby('Driver')['Distance'].max().median()


def find_telemetry(year):
    for name in (f'singapore_q_{year}_data.csv', f'singapore_q_top7_{year}_data.csv'):
        if (OUTPUT_DIR / name).exists():
            return OUTPUT_DIR / name
    return None


print(f"pulling Session Singapore {TARGET_YEAR}")
session = fastf1.get_session(TARGET_YEAR, 'Singapore', 'Q')
session.load()


all_drivers = session.results['Abbreviation'].tolist()
print(f"\ndrivers: {all_drivers}")

all_telemetry_data = pd.DataFrame()

for driver in all_drivers:
    try:
        fastest_lap = session.laps.pick_driver(driver).pick_fastest()
        if pd.isna(fastest_lap['Time']):
            print(f"skip {driver}: not found")
            continue

        telemetry = fastest_lap.get_telemetry()

        df = telemetry[['Distance', 'Speed', 'Throttle']].copy()
        df['Driver'] = driver

        all_telemetry_data = pd.concat([all_telemetry_data, df], ignore_index=True)
        print(f"pull {driver} succes amount: {len(df)} vector")

    except Exception as e:
        print(f"Error {driver}: {e}")

print("\n All done")
print(f"vectors: {len(all_telemetry_data)}")

telemetry_file_path = OUTPUT_DIR / f'singapore_q_{TARGET_YEAR}_data.csv'
all_telemetry_data.to_csv(telemetry_file_path, index=False)
print(f"save {telemetry_file_path} success")

print("\npulling corners")
corners_df = None
try:
    circuit_info = session.get_circuit_info()
    corners_df = pd.DataFrame({
        'Corner_Number': circuit_info.corners['Number'],
        'Distance': circuit_info.corners['Distance']
    })
except Exception as e:
    print(f"get_circuit_info failed: {e!r}")

if TARGET_YEAR <= LEGACY_LAST_YEAR and (corners_df is None or len(corners_df) != LEGACY_CORNERS):
    got = 'none' if corners_df is None else len(corners_df)
    ref_corners = OUTPUT_DIR / f'singapore_{REFERENCE_YEAR}_corners.csv'
    ref_telemetry = find_telemetry(REFERENCE_YEAR)
    if TARGET_YEAR == REFERENCE_YEAR or not (ref_corners.exists() and ref_telemetry is not None):
        sys.exit(f"FastF1 gave {got} corners for {TARGET_YEAR} (expected {LEGACY_CORNERS}) and the "
                 f"reference files for {REFERENCE_YEAR} are missing in {OUTPUT_DIR}. "
                 f"Run with TARGET_YEAR = {REFERENCE_YEAR} first.")
    scale = lap_length(all_telemetry_data) / lap_length(pd.read_csv(ref_telemetry))
    corners_df = pd.read_csv(ref_corners)
    corners_df['Distance'] = corners_df['Distance'] * scale
    print(f"FastF1 gave {got} corners for {TARGET_YEAR}; using the {LEGACY_CORNERS}-corner layout of "
          f"{REFERENCE_YEAR} scaled by {scale:.4f}")
elif corners_df is None:
    sys.exit("could not get the corner list from FastF1")

corners_file_path = OUTPUT_DIR / f'singapore_{TARGET_YEAR}_corners.csv'
corners_df.to_csv(corners_file_path, index=False)
print(f"save {corners_file_path} ({len(corners_df)} corners) success")
