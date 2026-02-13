import sys
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from pyproj import Transformer
from astropy.coordinates import EarthLocation, AltAz, get_sun
from astropy.time import Time
import astropy.units as u
from scipy.stats import binned_statistic
import tyro

# --- Constants ---
EARTH_RADIUS_METERS = 6371000.0
REFRACTION_COEFF = 0.13


@dataclass
class Args:
    """
    Calculate sun visibility, horizon line, and true sunshine hours from a DEM.
    """
    dem_path: Path
    """Path to the raster elevation file (GeoTIFF)."""
    lat: float
    """Observer Latitude (WGS84)."""
    lon: float
    """Observer Longitude (WGS84)."""
    height_above_ground: float = 1.7
    """Observer height above the ground surface in meters."""
    radius_km: float = 50.0
    """Maximum radius to scan for horizon in kilometers."""
    output_prefix: str = "sun_analysis"
    """Prefix for output files."""
    year: int = 2024
    """Year to simulate (affects leap years and slight orbital vars)."""


def calculate_horizon(dem_path: Path, lat: float, lon: float, observer_h_agl: float, max_radius_km: float):
    print(f"Loading DEM: {dem_path}...")
    with rasterio.open(dem_path) as src:
        transform = src.transform
        crs = src.crs
        nodata = src.nodata if src.nodata is not None else -9999

        transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        obs_x, obs_y = transformer.transform(lon, lat)

        row_obs, col_obs = src.index(obs_x, obs_y)

        try:
            # Read 1x1 window to get ground elev
            obs_ground_elev = src.read(1, window=((row_obs, row_obs + 1), (col_obs, col_obs + 1)))[0][0]
        except:
            raise ValueError("Location is outside the bounds of the provided DEM.")

        if obs_ground_elev == nodata:
            raise ValueError("Location falls on NoData value in DEM.")

        observer_elev_abs = obs_ground_elev + observer_h_agl
        print(
            f"Observer Elevation: {obs_ground_elev:.1f}m (Ground) + {observer_h_agl}m (Eye) = {observer_elev_abs:.1f}m")

        # Optimization: Read only relevant window
        res_x = transform[0]
        res_y = -transform[4]
        px_radius = int((max_radius_km * 1000) / min(abs(res_x), abs(res_y)))

        win_row_off = max(0, row_obs - px_radius)
        win_col_off = max(0, col_obs - px_radius)
        win_height = min(src.height - win_row_off, 2 * px_radius)
        win_width = min(src.width - win_col_off, 2 * px_radius)

        window = rasterio.windows.Window(win_col_off, win_row_off, win_width, win_height)
        data = src.read(1, window=window)

        # Grid generation
        win_transform = src.window_transform(window)
        cols, rows = np.meshgrid(np.arange(win_width), np.arange(win_height))
        xs, ys = rasterio.transform.xy(win_transform, rows, cols, offset='center')

        xs = np.array(xs).flatten()
        ys = np.array(ys).flatten()
        zs = data.flatten()

        valid_mask = zs != nodata
        xs = xs[valid_mask]
        ys = ys[valid_mask]
        zs = zs[valid_mask]

    print("Calculating terrain occlusion mask...")

    dx = xs - obs_x
    dy = ys - obs_y
    dist = np.sqrt(dx ** 2 + dy ** 2)

    mask = (dist > 0) & (dist <= max_radius_km * 1000)
    dist = dist[mask]
    zs = zs[mask]

    az_rad = np.arctan2(dy[mask], dx[mask])
    az_deg = (90 - np.degrees(az_rad)) % 360

    curvature_correction = (dist ** 2) / (2 * EARTH_RADIUS_METERS) * (1 - REFRACTION_COEFF)
    rel_height = zs - observer_elev_abs - curvature_correction

    angle_deg = np.degrees(np.arctan2(rel_height, dist))

    # 0.5 degree bins for smoother horizon
    bins = np.linspace(0, 360, 721)
    skyline_elev, _, _ = binned_statistic(az_deg, angle_deg, statistic='max', bins=bins)
    skyline_elev = np.nan_to_num(skyline_elev, nan=-1.0)  # -1 if no terrain

    azimuths = (bins[:-1] + bins[1:]) / 2

    return azimuths, skyline_elev


def get_horizon_at_az(az_target, az_skyline, alt_skyline):
    """Interpolate horizon elevation at specific azimuths."""
    # Handle wrapping for interpolation (359 -> 0)
    az_target = az_target % 360
    return np.interp(az_target, az_skyline, alt_skyline)


def get_sun_path_detailed(date_str, lat, lon, elevation_m):
    """Calculates high-res sun path for a specific day."""
    loc = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=elevation_m * u.m)

    # High res for intersection finding (1 min)
    time_start = Time(f"{date_str} 00:00:00")
    times = time_start + np.linspace(0, 24, 1440) * u.hour

    altaz_frame = AltAz(obstime=times, location=loc)
    sun_pos = get_sun(times).transform_to(altaz_frame)

    return sun_pos.az.deg, sun_pos.alt.deg, times


def find_visibility_events(az_sun, alt_sun, times, az_skyline, alt_skyline):
    """Finds first and last visibility times against the terrain."""
    horizon_at_sun = get_horizon_at_az(az_sun, az_skyline, alt_skyline)

    # Boolean mask where sun is visible above terrain
    is_visible = alt_sun > horizon_at_sun

    # If never visible
    if not np.any(is_visible):
        return None, None

    # Find indices where visibility changes
    # Use astype(int) to convert boolean to 0/1, then diff
    diff = np.diff(is_visible.astype(int))

    # +1 is rise (0 -> 1), -1 is set (1 -> 0)
    rise_indices = np.where(diff == 1)[0]
    set_indices = np.where(diff == -1)[0]

    events = {}

    # First Rise (add 1 because diff index i is between i and i+1)
    if len(rise_indices) > 0:
        idx = rise_indices[0] + 1
        events['first_rise'] = (times[idx], az_sun[idx], alt_sun[idx])
    elif is_visible[0]:  # Starts visible
        events['first_rise'] = (times[0], az_sun[0], alt_sun[0])

    # Last Set
    if len(set_indices) > 0:
        idx = set_indices[-1]
        events['last_set'] = (times[idx], az_sun[idx], alt_sun[idx])
    elif is_visible[-1]:  # Ends visible
        events['last_set'] = (times[-1], az_sun[-1], alt_sun[-1])

    return events


def calculate_full_year_stats(year, lat, lon, az_skyline, alt_skyline):
    """Calculates daily means by simulating the whole year."""
    print("Simulating full year sun positions (this takes a moment)...")

    loc = EarthLocation(lat=lat * u.deg, lon=lon * u.deg)

    # Create time array for entire year: 10 minute intervals
    # 365 days * 24 hours * 6 steps = 52,560 points
    start_time = Time(f"{year}-01-01 00:00:00")
    # Using pandas to generate timestamps then converting to astropy is often easier for ranges
    timestamps = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31 23:50:00", freq="10min")
    times = Time(timestamps)

    altaz_frame = AltAz(obstime=times, location=loc)
    sun_pos = get_sun(times).transform_to(altaz_frame)

    az_sun = sun_pos.az.deg
    alt_sun = sun_pos.alt.deg

    # Calculate Horizon for all sun points
    horizon_at_sun = get_horizon_at_az(az_sun, az_skyline, alt_skyline)

    # Masks
    # Theoretical: Sun simply above geometric horizon (0)
    mask_theo = alt_sun > 0
    # Actual: Sun above terrain
    mask_actual = alt_sun > horizon_at_sun

    # Create DataFrame to aggregate
    df = pd.DataFrame({
        'date': timestamps.date,
        'month': timestamps.month,
        'theo_visible': mask_theo,
        'actual_visible': mask_actual
    })

    # Group by date to get daily hours (sum of True * 10 mins / 60 mins)
    daily = df.groupby('date').agg({
        'theo_visible': 'sum',
        'actual_visible': 'sum',
        'month': 'first'
    })

    # Convert counts (10 min chunks) to hours
    daily['theo_hours'] = daily['theo_visible'] * (10 / 60)
    daily['actual_hours'] = daily['actual_visible'] * (10 / 60)

    # Group by month for final table
    monthly = daily.groupby('month').agg({
        'theo_hours': 'mean',
        'actual_hours': 'mean'
    }).reset_index()

    # Formatting
    month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                   7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}

    results = []
    for _, row in monthly.iterrows():
        m_name = month_names[int(row['month'])]
        t_hrs = row['theo_hours']
        a_hrs = row['actual_hours']
        results.append({
            "Month": m_name,
            "Theoretical Mean (hrs)": round(t_hrs, 2),
            "Actual Mean (hrs)": round(a_hrs, 2),
            "Mean Loss (hrs)": round(t_hrs - a_hrs, 2),
            "Mean Loss (%)": round((1 - a_hrs / t_hrs) * 100, 1) if t_hrs > 0 else 0
        })

    return pd.DataFrame(results)


def main(args: Args):
    # 1. Skyline
    print("--- 1. Terrain Analysis ---")
    az_skyline, alt_skyline = calculate_horizon(
        args.dem_path, args.lat, args.lon, args.height_above_ground, args.radius_km
    )

    # 2. Plotting (Solstices)
    print("--- 2. Calculating Solstice Paths ---")
    dates = {
        'Summer': f'{args.year}-06-21',
        'Winter': f'{args.year}-12-21'
    }
    colors = {'Summer': 'orange', 'Winter': 'blue'}

    plt.figure(figsize=(15, 7))

    # Fill Terrain
    plt.fill_between(az_skyline, -10, alt_skyline, color='#444444', alpha=0.6, label='Terrain Horizon')
    plt.plot(az_skyline, alt_skyline, color='black', linewidth=1)

    for season, date in dates.items():
        az_s, alt_s, times_s = get_sun_path_detailed(date, args.lat, args.lon, 0)

        # Plot path (daytime only)
        mask = alt_s > -5
        plt.plot(az_s[mask], alt_s[mask], color=colors[season], linewidth=2, label=f'{season} Solstice ({date})')

        # Find and annotate rise/set
        events = find_visibility_events(az_s, alt_s, times_s, az_skyline, alt_skyline)

        if events:
            # First Rise
            if 'first_rise' in events:
                t, az, alt = events['first_rise']
                time_str = t.datetime.strftime("%H:%M")
                plt.scatter(az, alt, color='red', zorder=5, s=50)
                plt.annotate(f"{season} Rise\n{time_str}", (az, alt), xytext=(0, 15),
                             textcoords='offset points', ha='center', fontsize=9,
                             bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

            # Last Set
            if 'last_set' in events:
                t, az, alt = events['last_set']
                time_str = t.datetime.strftime("%H:%M")
                plt.scatter(az, alt, color='red', zorder=5, s=50)
                plt.annotate(f"{season} Set\n{time_str}", (az, alt), xytext=(0, 15),
                             textcoords='offset points', ha='center', fontsize=9,
                             bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    plt.xlim(0, 360)
    plt.ylim(0, 90)
    plt.xlabel("Azimuth (Degrees)")
    plt.ylabel("Elevation (Degrees)")
    plt.title(f"Sun Visibility: {args.lat:.4f}, {args.lon:.4f} | Eye Height: {args.height_above_ground}m")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right')
    plt.xticks(np.arange(0, 361, 45), ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'N'])

    plot_file = f"{args.output_prefix}_plot.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {plot_file}")

    # 3. Annual Stats
    print("--- 3. Calculating Annual Statistics (Full Year Simulation) ---")
    df = calculate_full_year_stats(args.year, args.lat, args.lon, az_skyline, alt_skyline)

    csv_file = f"{args.output_prefix}_stats.csv"
    df.to_csv(csv_file, index=False)

    print("\n--- Monthly Mean Daily Sunshine (Hours) ---")
    print(df.to_string(index=False))
    print(f"\nStatistics saved to {csv_file}")


if __name__ == "__main__":
    tyro.cli(main)