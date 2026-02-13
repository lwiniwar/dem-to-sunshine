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
    Calculate sun visibility with memory-efficient adaptive sampling.
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
    """Year to simulate."""
    angular_res_deg: float = 0.1
    """Target angular resolution. Smaller values require more sampling close up."""


def get_horizon_at_az(az_target, az_skyline, alt_skyline):
    az_target = az_target % 360
    return np.interp(az_target, az_skyline, alt_skyline)


def process_chunk(dx, dy, dz, obs_elev):
    """
    Core vector math to convert relative cartesian coords to Az/El.
    Optimized for memory (in-place operations where possible).
    """
    dist_sq = dx ** 2 + dy ** 2
    dist = np.sqrt(dist_sq)

    # Filter very close points (observer self-occlusion)
    mask = dist > 0.05
    if not np.any(mask):
        return np.array([]), np.array([])

    dx = dx[mask]
    dy = dy[mask]
    dz = dz[mask]
    dist = dist[mask]

    # Azimuth
    az_rad = np.arctan2(dy, dx)
    az_deg = (90 - np.degrees(az_rad)) % 360

    # Curvature & Refraction
    curvature = (dist ** 2) / (2 * EARTH_RADIUS_METERS) * (1 - REFRACTION_COEFF)
    rel_height = dz - obs_elev - curvature

    angle_deg = np.degrees(np.arctan2(rel_height, dist))

    return az_deg, angle_deg


def calculate_horizon_adaptive(args: Args):
    print(f"Loading DEM Metadata: {args.dem_path}...")
    with rasterio.open(args.dem_path) as src:
        transform = src.transform
        crs = src.crs
        nodata = src.nodata if src.nodata is not None else -9999
        res = abs(transform[0])  # Assuming square pixels roughly

        # 1. Locate Observer
        transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        obs_x_proj, obs_y_proj = transformer.transform(args.lon, args.lat)
        row_obs, col_obs = src.index(obs_x_proj, obs_y_proj)

        # Get Ground Level
        try:
            obs_ground = src.read(1, window=((row_obs, row_obs + 1), (col_obs, col_obs + 1)))[0][0]
        except:
            raise ValueError("Observer outside DEM bounds.")

        if obs_ground == nodata:
            raise ValueError("Observer on NoData pixel.")

        obs_abs_elev = obs_ground + args.height_above_ground
        print(f"Observer Elev: {obs_abs_elev:.2f}m (Ground {obs_ground:.1f}m + Eye {args.height_above_ground}m)")
        print(f"Native Resolution: {res:.2f}m")

        # 2. Determine Critical Distance
        # Distance at which 1 pixel (res) equals angular_res
        # tan(theta) = opp/adj => adj = res / tan(theta)
        rad_limit_rad = np.deg2rad(args.angular_res_deg)
        crit_dist_m = res / np.tan(rad_limit_rad)
        crit_radius_px = int(np.ceil(crit_dist_m / res))

        print(f"Adaptive Sampling Analysis:")
        print(f"  Target Angular Res: {args.angular_res_deg}°")
        print(f"  Critical Distance:  {crit_dist_m:.1f}m ({crit_radius_px} pixels)")
        print(f"  (Inside this radius, pixels are too large and will be sub-sampled)")

        # Prepare Bins for Skyline (global accumulator)
        # We process chunks and immediately update the histogram stats to save memory
        # However, binned_statistic is not incremental.
        # Strategy: Accumulate all (az, el) pairs?
        # No, that's too much memory (billions of points).
        # Better: Accumulate (az, el) for chunks, bin them to a temp skyline,
        # and keep the MAX of the temp skylines.

        final_bins = np.linspace(0, 360, int(360 / args.angular_res_deg) + 1)
        global_skyline = np.full(len(final_bins) - 1, -90.0)

        def update_skyline(az_chunk, el_chunk):
            if len(az_chunk) == 0: return
            # Bin this chunk
            chunk_skyline, _, _ = binned_statistic(az_chunk, el_chunk, statistic='max', bins=final_bins)
            chunk_skyline = np.nan_to_num(chunk_skyline, nan=-90.0)
            # Update global max
            np.maximum(global_skyline, chunk_skyline, out=global_skyline)

        # --- PHASE 1: Adaptive "Onion Layers" (0 to Critical Distance) ---
        print("\n--- Phase 1: Adaptive Near-Field Processing ---")

        # We iterate through rings. To go fast, we group rings with similar Sampling Factors.
        # Factor F = res / (dist * tan(alpha))
        # We can process in blocks of N pixels radius.

        # Load the whole critical window into memory (it's usually small, e.g. 500x500 px)
        # Add a buffer
        safe_crit_px = crit_radius_px + 2

        w_row_off = max(0, row_obs - safe_crit_px)
        w_col_off = max(0, col_obs - safe_crit_px)
        w_h = min(src.height - w_row_off, 2 * safe_crit_px + 1)
        w_w = min(src.width - w_col_off, 2 * safe_crit_px + 1)

        window = rasterio.windows.Window(w_col_off, w_row_off, w_w, w_h)
        data_crit = src.read(1, window=window)

        # Offsets of the window relative to observer
        win_center_row = row_obs - w_row_off
        win_center_col = col_obs - w_col_off

        # Iterate rings from r=1 to crit_radius_px
        # We group them: r=1, r=2, r=3... processing ring by ring is actually fine for memory
        # and allows precise factor calculation.

        for r in range(1, safe_crit_px):
            # 1. Extract Ring Coordinates (Chebyshev distance r)
            # Top, Bottom, Left, Right segments
            # Array slicing: [row_start:row_end, col_start:col_end]

            # Bounds of the ring in the window array
            r_min, r_max = win_center_row - r, win_center_row + r
            c_min, c_max = win_center_col - r, win_center_col + r

            # Check bounds
            if r_min < 0 or r_max >= w_h or c_min < 0 or c_max >= w_w:
                continue

            # Extract pixels for this ring
            # Top & Bottom rows
            r_indices = np.concatenate([np.full(2 * r + 1, r_min), np.full(2 * r + 1, r_max)])
            c_indices = np.concatenate([np.arange(c_min, c_max + 1), np.arange(c_min, c_max + 1)])

            # Left & Right columns (excluding corners already taken)
            if r > 0:  # Avoid duplicating center if r=0
                r_side = np.concatenate([np.arange(r_min + 1, r_max), np.arange(r_min + 1, r_max)])
                c_side = np.concatenate([np.full(2 * r - 1, c_min), np.full(2 * r - 1, c_max)])
                r_indices = np.concatenate([r_indices, r_side])
                c_indices = np.concatenate([c_indices, c_side])

            # Get Z values
            zs = data_crit[r_indices, c_indices]
            valid = zs != nodata
            zs = zs[valid]
            r_indices = r_indices[valid]
            c_indices = c_indices[valid]

            if len(zs) == 0: continue

            # Calculate Distance and required Sampling Factor
            # Distance in pixels (approximate as r for factor calc, or Euclidean)
            # Use minimum distance of the ring (r pixels) to be safe/conservative
            dist_m = r * res

            # Calculate required sub-divisions (Linear density)
            # Arc length at this distance = dist_m * tan(alpha)
            # We need PixelSize / Factor < Arc
            # Factor > PixelSize / Arc
            arc = dist_m * np.tan(rad_limit_rad)
            factor = int(np.ceil(res / arc)) if arc > 0 else 100

            # Clamp factor to sane limits (e.g., max 50 subdivisions per pixel axis)
            factor = min(factor, 50)
            factor = max(factor, 1)

            # Generate Coordinates
            # Convert window indices back to relative meters
            # relative row/col
            dy_base = (r_indices - win_center_row) * -res  # -dy because row index increases down
            dx_base = (c_indices - win_center_col) * res

            if factor > 1:
                # Vectorized Supersampling
                # Create offset grid
                step = res / factor
                # Offsets centered around 0 (-res/2 to +res/2)
                off = np.linspace(-res / 2 + step / 2, res / 2 - step / 2, factor)
                g_x, g_y = np.meshgrid(off, off)
                g_x = g_x.flatten()
                g_y = g_y.flatten()

                # Broadcast: repeat each pixel 'factor^2' times and add offsets
                # n_pixels -> n_pixels * n_offsets
                dx_full = np.repeat(dx_base, len(g_x)) + np.tile(g_x, len(dx_base))
                dy_full = np.repeat(dy_base, len(g_y)) + np.tile(g_y, len(dy_base))
                dz_full = np.repeat(zs, len(g_x))  # Nearest Neighbor Height

            else:
                dx_full, dy_full, dz_full = dx_base, dy_base, zs

            # Process angles and update histogram
            az, el = process_chunk(dx_full, dy_full, dz_full, obs_abs_elev)
            update_skyline(az, el)

            # Progress marker for dense zone
            if r % 50 == 0:
                print(
                    f"  Ring {r}/{crit_radius_px} (Dist: {dist_m:.1f}m) | Sampling Factor: {factor}x{factor} ({factor ** 2} pts/px)")

        # --- PHASE 2: Far Field (Native Resolution) ---
        print(f"\n--- Phase 2: Far Field Processing (> {crit_dist_m:.1f}m) ---")

        # We define a window for the max radius
        max_px_radius = int((args.radius_km * 1000) / res)

        # We need to process the area between 'safe_crit_px' and 'max_px_radius'.
        # Loading 50km radius at 0.2m res is IMPOSSIBLE (500,000^2 pixels).
        # We MUST process in blocks.

        block_size = 2048  # Process 2k x 2k chunks

        # Window bounds
        full_w_row_off = max(0, row_obs - max_px_radius)
        full_w_col_off = max(0, col_obs - max_px_radius)
        full_w_h = min(src.height - full_w_row_off, 2 * max_px_radius)
        full_w_w = min(src.width - full_w_col_off, 2 * max_px_radius)

        # Iterate over blocks within the bounding box
        for r_start in range(0, full_w_h, block_size):
            for c_start in range(0, full_w_w, block_size):

                # Define block window
                w_r = full_w_row_off + r_start
                w_c = full_w_col_off + c_start
                h = min(block_size, full_w_h - r_start)
                w = min(block_size, full_w_w - c_start)

                # Determine distance to block center to skip far blocks quickly
                # (Simple check to avoid reading I/O if block is outside circular radius)
                # Global coords of block center
                blk_cent_r = w_r + h / 2
                blk_cent_c = w_c + w / 2
                dist_to_obs_px = np.sqrt((blk_cent_r - row_obs) ** 2 + (blk_cent_c - col_obs) ** 2)

                # Skip if block is completely outside max radius
                # block diagonal approx size ~ 1.4 * block_size
                if dist_to_obs_px - (block_size) > max_px_radius:
                    continue

                # Skip if block is completely INSIDE critical radius (already processed)
                if dist_to_obs_px + (block_size) < safe_crit_px:
                    continue

                # Read Block
                # Note: read() is thread-safe in recent rasterio, but we are single threaded here.
                # Use window
                try:
                    window = rasterio.windows.Window(w_c, w_r, w, h)
                    data = src.read(1, window=window)
                except:
                    continue

                # Generate Coordinates
                # Local coords relative to Observer
                # Global row indices
                rows, cols = np.indices(data.shape)
                global_rows = rows + w_r
                global_cols = cols + w_c

                dy = (global_rows - row_obs) * -res
                dx = (global_cols - col_obs) * res

                # Flatten
                dx = dx.flatten()
                dy = dy.flatten()
                zs = data.flatten()

                # Mask NoData
                valid = zs != nodata

                # Calculate Distances for Critical Zone Masking
                # We must mask out points < crit_dist (they were handled in Phase 1)
                dists = np.sqrt(dx ** 2 + dy ** 2)
                mask_far = (dists > crit_dist_m) & (dists <= args.radius_km * 1000) & valid

                if np.any(mask_far):
                    az, el = process_chunk(dx[mask_far], dy[mask_far], zs[mask_far], obs_abs_elev)
                    update_skyline(az, el)

            print(f"  Processed row block {r_start}/{full_w_h}...")

    # Finalize
    bin_centers = (final_bins[:-1] + final_bins[1:]) / 2
    return bin_centers, global_skyline


def get_sun_path_detailed(date_str, lat, lon, elevation_m):
    loc = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=elevation_m * u.m)
    time_start = Time(f"{date_str} 00:00:00")
    times = time_start + np.linspace(0, 24, 1440) * u.hour
    altaz_frame = AltAz(obstime=times, location=loc)
    sun_pos = get_sun(times).transform_to(altaz_frame)
    return sun_pos.az.deg, sun_pos.alt.deg, times


def find_visibility_events(az_sun, alt_sun, times, az_skyline, alt_skyline):
    horizon_at_sun = np.interp(az_sun % 360, az_skyline, alt_skyline)
    is_visible = alt_sun > horizon_at_sun

    if not np.any(is_visible): return {}

    diff = np.diff(is_visible.astype(int))
    rise_indices = np.where(diff == 1)[0]
    set_indices = np.where(diff == -1)[0]

    events = {}
    if len(rise_indices) > 0:
        idx = rise_indices[0] + 1
        events['first_rise'] = (times[idx], az_sun[idx], alt_sun[idx])
    elif is_visible[0]:
        events['first_rise'] = (times[0], az_sun[0], alt_sun[0])

    if len(set_indices) > 0:
        idx = set_indices[-1]
        events['last_set'] = (times[idx], az_sun[idx], alt_sun[idx])
    elif is_visible[-1]:
        events['last_set'] = (times[-1], az_sun[-1], alt_sun[-1])

    return events


def calculate_full_year_stats(year, lat, lon, az_skyline, alt_skyline):
    print("\n--- Calculating Annual Stats ---")
    loc = EarthLocation(lat=lat * u.deg, lon=lon * u.deg)

    timestamps = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31 23:50:00", freq="10min")
    times = Time(timestamps)

    altaz_frame = AltAz(obstime=times, location=loc)
    sun_pos = get_sun(times).transform_to(altaz_frame)

    az_sun = sun_pos.az.deg
    alt_sun = sun_pos.alt.deg

    horizon_at_sun = np.interp(az_sun % 360, az_skyline, alt_skyline)

    mask_theo = alt_sun > 0
    mask_actual = alt_sun > horizon_at_sun

    df = pd.DataFrame({'date': timestamps.date, 'month': timestamps.month,
                       'theo_visible': mask_theo, 'actual_visible': mask_actual})

    daily = df.groupby('date').agg({'theo_visible': 'sum', 'actual_visible': 'sum', 'month': 'first'})
    daily['theo_hours'] = daily['theo_visible'] * (10 / 60)
    daily['actual_hours'] = daily['actual_visible'] * (10 / 60)

    monthly = daily.groupby('month').agg({'theo_hours': 'mean', 'actual_hours': 'mean'}).reset_index()

    month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                   7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}

    results = []
    for _, row in monthly.iterrows():
        t_hrs = row['theo_hours']
        a_hrs = row['actual_hours']
        results.append({
            "Month": month_names[int(row['month'])],
            "Theoretical (hrs)": round(t_hrs, 2),
            "Actual (hrs)": round(a_hrs, 2),
            "Loss (hrs)": round(t_hrs - a_hrs, 2),
            "Loss (%)": round((1 - a_hrs / t_hrs) * 100, 1) if t_hrs > 0 else 0
        })
    return pd.DataFrame(results)


def main(args: Args):
    # 1. Adaptive Horizon Scan
    az_skyline, alt_skyline = calculate_horizon_adaptive(args)

    # 2. Plotting
    print("\n--- Generating Plot ---")
    dates = {'Summer': f'{args.year}-06-21', 'Winter': f'{args.year}-12-21'}
    colors = {'Summer': 'orange', 'Winter': 'blue'}

    plt.figure(figsize=(15, 7))
    plt.fill_between(az_skyline, -90, alt_skyline, color='#444444', alpha=0.6, label='Terrain Horizon')
    plt.plot(az_skyline, alt_skyline, color='black', linewidth=0.8)

    for season, date in dates.items():
        az_s, alt_s, times_s = get_sun_path_detailed(date, args.lat, args.lon, 0)
        mask = alt_s > -5
        plt.plot(az_s[mask], alt_s[mask], color=colors[season], linewidth=2, label=f'{season} ({date})')

        events = find_visibility_events(az_s, alt_s, times_s, az_skyline, alt_skyline)
        if events:
            for k, (t, az, alt) in events.items():
                lbl = "Rise" if "rise" in k else "Set"
                time_str = t.datetime.strftime("%H:%M")
                plt.scatter(az, alt, color='red', zorder=5, s=40)
                plt.annotate(f"{lbl}\n{time_str}", (az, alt), xytext=(0, 15),
                             textcoords='offset points', ha='center', fontsize=8,
                             bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

    plt.xlim(0, 360)
    plt.ylim(0, 90)
    plt.xlabel("Azimuth")
    plt.ylabel("Elevation")
    plt.title(f"Sun Visibility Analysis | Loc: {args.lat:.4f}, {args.lon:.4f}")
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.xticks(np.arange(0, 361, 45), ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'N'])

    plt.savefig(f"{args.output_prefix}_plot.png", dpi=150, bbox_inches='tight')

    # 3. Stats
    df = calculate_full_year_stats(args.year, args.lat, args.lon, az_skyline, alt_skyline)
    df.to_csv(f"{args.output_prefix}_stats.csv", index=False)
    print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    tyro.cli(main)