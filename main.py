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
    High-precision Sun Visibility Tool with Dense Bilinear Near-Field Interpolation.
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
    angular_res_deg: float = 0.05
    """Target angular resolution. For 0.2m raster, 0.05-0.1 is recommended."""


def process_chunk(dx, dy, dz, obs_elev):
    """
    Vectorized conversion from relative cartesian (dx, dy, dz) to Horizon (Az, El).
    """
    dist_sq = dx ** 2 + dy ** 2
    dist = np.sqrt(dist_sq)

    # Filter observer self-occlusion (points < 1cm away)
    mask = dist > 0.01
    if not np.any(mask):
        return np.array([]), np.array([])

    dx = dx[mask]
    dy = dy[mask]
    dz = dz[mask]
    dist = dist[mask]

    # Azimuth (Mathematical -> Geographic)
    az_rad = np.arctan2(dy, dx)
    az_deg = (90 - np.degrees(az_rad)) % 360

    # Curvature & Refraction Correction
    # Apparent Elevation = Geometric Elevation - Correction
    # Correction is positive (earth drops away), so we subtract it from target height?
    # Actually standard formula: H_target_apparent = H_target - (d^2/2R)(1-k)
    # Then angle = atan(H_apparent / d)
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
        res_x = abs(transform[0])
        res_y = abs(transform[4])
        res = (res_x + res_y) / 2  # Average resolution

        # 1. Locate Observer
        transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        obs_x_proj, obs_y_proj = transformer.transform(args.lon, args.lat)

        # Get float indices (exact position within pixel)
        r_float, c_float = src.index(obs_x_proj, obs_y_proj, op=float)
        row_obs_int, col_obs_int = int(r_float), int(c_float)

        # 2. Interpolate Observer Ground Height (Bilinear)
        try:
            # Read 2x2 window around the float index
            # integer index (r, c) refers to the pixel covering the coordinate.
            # For bilinear interpolation between centers, we need to know where we are
            # relative to the center.
            # Rasterio index returns floating point relative to top-left of image (0,0).
            # Pixel (0,0) center is at (0.5, 0.5).
            # If r_float = 10.5, we are exactly at center of pixel 10.
            # If r_float = 10.0, we are at the top edge of pixel 10 (midway between 9 and 10 centers).

            # We align to the top-left neighbor for interpolation
            r_nw = int(np.floor(r_float - 0.5))
            c_nw = int(np.floor(c_float - 0.5))

            w = rasterio.windows.Window(c_nw, r_nw, 2, 2)
            z_win = src.read(1, window=w)

            if z_win.shape == (2, 2) and np.all(z_win != nodata):
                # Calculate weights based on distance from center of NW pixel (r_nw+0.5)
                # Position of interest is (r_float, c_float)
                # Position of NW pixel center is (r_nw + 0.5, c_nw + 0.5)
                dr = r_float - (r_nw + 0.5)
                dc = c_float - (c_nw + 0.5)

                # Bilinear weights
                obs_ground = (1 - dr) * (1 - dc) * z_win[0, 0] + dr * (1 - dc) * z_win[1, 0] + \
                             (1 - dr) * dc * z_win[0, 1] + dr * dc * z_win[1, 1]
            else:
                # Fallback to nearest
                obs_ground = src.read(1, window=((row_obs_int, row_obs_int + 1), (col_obs_int, col_obs_int + 1)))[0][0]

        except Exception as e:
            print(f"Warning: Could not interpolate observer exact height ({e}), using pixel center.")
            obs_ground = src.read(1, window=((row_obs_int, row_obs_int + 1), (col_obs_int, col_obs_int + 1)))[0][0]

        if obs_ground == nodata:
            raise ValueError("Observer is located on NoData pixel.")

        obs_abs_elev = obs_ground + args.height_above_ground
        print(f"Observer Location (Grid): {r_float:.2f}, {c_float:.2f}")
        print(f"Observer Elevation: {obs_abs_elev:.2f}m (Ground {obs_ground:.2f}m + Eye {args.height_above_ground}m)")

        # 3. Critical Distance Calculation
        # Distance where pixel size subtends the target angular resolution
        # d = res / tan(alpha)
        rad_limit = np.deg2rad(args.angular_res_deg)
        crit_dist_m = res / np.tan(rad_limit)
        crit_radius_px = int(np.ceil(crit_dist_m / res))

        print(f"\n--- Sampling Strategy ---")
        print(f"Target Resolution: {args.angular_res_deg}°")
        print(f"Native Pixel Size: {res:.2f}m")
        print(f"Critical Near-Field Radius: {crit_dist_m:.1f}m ({crit_radius_px} pixels)")
        print(f"Method: Dense Bilinear Interpolation (Inner) -> Native Grid (Outer)")

        # Prepare Histogram Bins
        final_bins = np.linspace(0, 360, int(360 / args.angular_res_deg) + 1)
        global_skyline = np.full(len(final_bins) - 1, -90.0)

        def update_skyline(az_chunk, el_chunk):
            if len(az_chunk) == 0: return
            chunk_skyline, _, _ = binned_statistic(az_chunk, el_chunk, statistic='max', bins=final_bins)
            chunk_skyline = np.nan_to_num(chunk_skyline, nan=-90.0)
            np.maximum(global_skyline, chunk_skyline, out=global_skyline)

        # --- PHASE 1: Dense Bilinear Interpolation (The "Inner Rings") ---
        print("\n--- Phase 1: Processing Near Field ---")

        # Load window covering critical radius + buffer
        # We center the window on the "Nearest Top-Left" integer index to align with the grid
        w_pad = crit_radius_px + 2
        w_r_start = max(0, row_obs_int - w_pad)
        w_c_start = max(0, col_obs_int - w_pad)
        w_h = min(src.height - w_r_start, 2 * w_pad + 2)
        w_w = min(src.width - w_c_start, 2 * w_pad + 2)

        window = rasterio.windows.Window(w_c_start, w_r_start, w_w, w_h)
        data_crit = src.read(1, window=window)

        # Window offset relative to the float observer position
        # Grid index (0,0) in 'data_crit' corresponds to global index (w_r_start, w_c_start)
        # Its center is at (w_r_start + 0.5, w_c_start + 0.5)
        # Observer is at (r_float, c_float)

        # We iterate through integer rings relative to the observer's integer pixel
        # Center of the "Onion" is (row_obs_int, col_obs_int)
        win_obs_r = row_obs_int - w_r_start
        win_obs_c = col_obs_int - w_c_start

        # Iterate rings 1..N
        # We start at 0 to capture the immediate surroundings (the 4 pixels around the observer)
        for r in range(0, w_pad):

            # 1. Determine Sampling Density for this ring
            # Distance from observer to ring ~ r * res
            dist_m = max(r * res, res * 0.5)  # Avoid 0

            # Required Arc Length = dist * tan(alpha)
            # Pixels per Arc = res / Arc
            arc = dist_m * np.tan(rad_limit)

            # Factor: How many samples per pixel edge?
            # Cap at 300 (very dense) for innermost ring to ensure < 1mm precision if needed
            if arc > 0:
                factor = int(np.ceil(res / arc))
            else:
                factor = 300

            # Clamp limits
            factor = max(min(factor, 300), 2)  # Always at least 2x2 for bilinear continuity

            # 2. Extract Ring Indices (Top-Left corners of 2x2 blocks)
            # Ring 0: The single pixel (win_obs_r, win_obs_c) - actually we need neighbors
            # Let's use a box approach: range [center-r, center+r]
            r_min = win_obs_r - r
            r_max = win_obs_r + r
            c_min = win_obs_c - r
            c_max = win_obs_c + r

            # Generate outline coordinates
            if r == 0:
                # Just the center pixel block
                r_ind = np.array([win_obs_r])
                c_ind = np.array([win_obs_c])
            else:
                # Perimeter
                # Top & Bottom
                rt = np.full(2 * r + 1, r_min)
                ct = np.arange(c_min, c_max + 1)
                rb = np.full(2 * r + 1, r_max)
                cb = np.arange(c_min, c_max + 1)
                # Sides (excluding corners)
                rl = np.arange(r_min + 1, r_max)
                cl = np.full(2 * r - 1, c_min)
                rr = np.arange(r_min + 1, r_max)
                cr = np.full(2 * r - 1, c_max)

                r_ind = np.concatenate([rt, rb, rl, rr])
                c_ind = np.concatenate([ct, cb, cl, cr])

            # 3. Safety Bounds Check
            # We need +1 for neighbors
            valid = (r_ind >= 0) & (r_ind < w_h - 1) & (c_ind >= 0) & (c_ind < w_w - 1)
            r_ind = r_ind[valid]
            c_ind = c_ind[valid]

            if len(r_ind) == 0: continue

            # 4. Fetch 2x2 Neighborhoods
            # Z00 at (r,c), Z01 at (r,c+1)...
            z00 = data_crit[r_ind, c_ind]
            z01 = data_crit[r_ind, c_ind + 1]
            z10 = data_crit[r_ind + 1, c_ind]
            z11 = data_crit[r_ind + 1, c_ind + 1]

            # Skip blocks with NoData
            mask_nodata = (z00 != nodata) & (z01 != nodata) & (z10 != nodata) & (z11 != nodata)
            if not np.any(mask_nodata): continue

            z00, z01, z10, z11 = z00[mask_nodata], z01[mask_nodata], z10[mask_nodata], z11[mask_nodata]
            r_base, c_base = r_ind[mask_nodata], c_ind[mask_nodata]

            # 5. Generate Dense Sub-Pixel Coordinates
            # We sample the area covered by the pixel (r_base, c_base)
            # The area spans from center of (r,c) to center of (r+1, c+1) if we interpret strictly?
            # Standard Bilinear: The surface is defined on the square [0,1]x[0,1] between grid points.
            # Grid points are at integer indices + 0.5 (centers).
            # Here we treat the grid indices as the sample points.

            # Generate U, V steps
            step = 1.0 / factor
            # Linspace from 0 to 1 (inclusive? usually exclusive of 1 for next pixel)
            # We want to cover the interval [0, 1).
            # To catch peaks exactly at grid points, we include 0.
            offsets = np.linspace(0, 1.0 - step, factor)

            u_grid, v_grid = np.meshgrid(offsets, offsets)
            u_flat = u_grid.flatten()
            v_flat = v_grid.flatten()

            # 6. Broadcast and Interpolate
            n_blocks = len(z00)
            n_samps = len(u_flat)

            # (N_blocks * N_samples)
            Z00 = np.repeat(z00, n_samps)
            Z01 = np.repeat(z01, n_samps)
            Z10 = np.repeat(z10, n_samps)
            Z11 = np.repeat(z11, n_samps)

            U = np.tile(u_flat, n_blocks)
            V = np.tile(v_flat, n_blocks)

            # Bilinear Formula
            # Z = (1-v)(1-u)Z00 + (1-v)uZ01 + v(1-u)Z10 + vuZ11
            w00 = (1 - V) * (1 - U)
            w01 = (1 - V) * U
            w10 = V * (1 - U)
            w11 = V * U

            z_interp = w00 * Z00 + w01 * Z01 + w10 * Z10 + w11 * Z11

            # 7. Convert to Relative Coordinates (Meters)
            # Global float index of sample = (r_base + V) + global_offset
            # Global float index of observer = r_float
            # Delta_rows = (r_base + V + w_r_start) - r_float
            # But remember: Index 0 center is 0.5.
            # Grid points (data sources) are at (Integer + 0.5).
            # We interpolated between (r_base+0.5) and (r_base+1.5).
            # So actual position = (r_base + 0.5) + V.

            abs_r = (np.repeat(r_base, n_samps) + w_r_start + 0.5) + V
            abs_c = (np.repeat(c_base, n_samps) + w_c_start + 0.5) + U

            dy = (r_float - abs_r) * res  # North (+y) if obs is "above" (lower index) target?
            # If obs=10, target=12. target is south. dy should be neg.
            # (10 - 12) = -2. Correct.
            dx = (abs_c - c_float) * res  # East (+x)

            # 8. Update Skyline
            az, el = process_chunk(dx, dy, z_interp, obs_abs_elev)
            update_skyline(az, el)

            if r % 10 == 0:
                print(
                    f"  Ring {r}/{crit_radius_px} | Density: {factor}x{factor} ({n_samps} pts/px) | Pts: {len(z_interp)}")

        # --- PHASE 2: Far Field (Native Resolution) ---
        print(f"\n--- Phase 2: Processing Far Field (> {crit_dist_m:.1f}m) ---")

        max_px_radius = int((args.radius_km * 1000) / res)
        block_size = 2048

        full_w_r_start = max(0, row_obs_int - max_px_radius)
        full_w_c_start = max(0, col_obs_int - max_px_radius)
        full_w_h = min(src.height - full_w_r_start, 2 * max_px_radius)
        full_w_w = min(src.width - full_w_c_start, 2 * max_px_radius)

        for r_start in range(0, full_w_h, block_size):
            for c_start in range(0, full_w_w, block_size):
                w_r = full_w_r_start + r_start
                w_c = full_w_c_start + c_start
                h = min(block_size, full_w_h - r_start)
                w = min(block_size, full_w_w - c_start)

                # Check distance of block center
                blk_cent_r = w_r + h / 2
                blk_cent_c = w_c + w / 2
                dist_to_obs_px = np.sqrt((blk_cent_r - row_obs_int) ** 2 + (blk_cent_c - col_obs_int) ** 2)

                # Skip if outside max radius or inside critical radius
                if dist_to_obs_px - block_size > max_px_radius: continue
                if dist_to_obs_px + block_size < crit_radius_px: continue

                try:
                    window = rasterio.windows.Window(w_c, w_r, w, h)
                    data = src.read(1, window=window)
                except:
                    continue

                rows, cols = np.indices(data.shape)
                # Convert to global pixel centers (Int + 0.5)
                global_r = (rows + w_r) + 0.5
                global_c = (cols + w_c) + 0.5

                dy = (r_float - global_r) * res
                dx = (global_c - c_float) * res

                dx = dx.flatten()
                dy = dy.flatten()
                zs = data.flatten()

                valid = zs != nodata
                dists = np.sqrt(dx ** 2 + dy ** 2)

                # Mask out the critical zone we already processed
                mask_far = (dists > crit_dist_m) & (dists <= args.radius_km * 1000) & valid

                if np.any(mask_far):
                    az, el = process_chunk(dx[mask_far], dy[mask_far], zs[mask_far], obs_abs_elev)
                    update_skyline(az, el)

            if r_start % (block_size * 2) == 0:
                print(f"  Processed block row {r_start}")

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
    az_skyline, alt_skyline = calculate_horizon_adaptive(args)

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

    # --- Additional: Hemisphere Plot ---
    print("--- Generating Hemisphere Plot ---")
    plt.figure(figsize=(10, 10))
    ax_polar = plt.subplot(111, projection='polar')

    # 1. Setup Direction (N=Up, Clockwise)
    ax_polar.set_theta_zero_location('N')
    ax_polar.set_theta_direction(-1)

    # 2. Convert Terrain to Polar Coordinates
    # Azimuth -> Radians
    # Elevation -> Zenith Distance (90 - Elev)
    # We define r=0 as Zenith (90deg elev) and r=90 as Horizon (0deg elev)
    az_rad_sky = np.radians(az_skyline)
    zenith_sky = 90 - alt_skyline

    # 3. Plot Terrain (Fill from Skyline out to Horizon/90)
    ax_polar.fill_between(az_rad_sky, zenith_sky, 90, color='#444444', alpha=0.6, label='Terrain')
    ax_polar.plot(az_rad_sky, zenith_sky, color='black', linewidth=0.8)

    # 4. Plot Sun Paths
    for season, date in dates.items():
        az_s, alt_s, times_s = get_sun_path_detailed(date, args.lat, args.lon, 0)

        # Filter visible sun (above geometric horizon)
        mask = alt_s > -5

        s_az_rad = np.radians(az_s[mask])
        s_zenith = 90 - alt_s[mask]

        ax_polar.plot(s_az_rad, s_zenith, color=colors[season], linewidth=2, label=f'{season}')

        # Re-calculate events for annotation
        events = find_visibility_events(az_s, alt_s, times_s, az_skyline, alt_skyline)
        if events:
            for k, (t, az, alt) in events.items():
                ev_az_rad = np.radians(az)
                ev_zenith = 90 - alt
                lbl = "Rise" if "rise" in k else "Set"
                time_str = t.datetime.strftime("%H:%M")

                ax_polar.scatter(ev_az_rad, ev_zenith, color='red', zorder=5, s=40)
                # Annotate
                ax_polar.annotate(f"{lbl}\n{time_str}", (ev_az_rad, ev_zenith),
                                  xytext=(0, 10), textcoords='offset points',
                                  ha='center', fontsize=8,
                                  bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

    # 5. Styling
    ax_polar.set_rlim(0, 90)  # Center=0, Edge=90
    # Custom Grid Labels (Show Elevation instead of Zenith Distance)
    ax_polar.set_yticks(np.arange(0, 91, 15))
    ax_polar.set_yticklabels([f"{90 - y}°" for y in np.arange(0, 91, 15)])

    ax_polar.set_title(f"Sky View (Hemisphere)\nLoc: {args.lat:.4f}, {args.lon:.4f}")
    ax_polar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    hemi_file = f"{args.output_prefix}_hemisphere.png"
    plt.savefig(hemi_file, dpi=150, bbox_inches='tight')
    print(f"Hemisphere plot saved to {hemi_file}")

    df = calculate_full_year_stats(args.year, args.lat, args.lon, az_skyline, alt_skyline)
    df.to_csv(f"{args.output_prefix}_stats.csv", index=False)
    print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    tyro.cli(main)