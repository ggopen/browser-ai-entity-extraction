// Coordinate transforms: WGS84 lon/lat/alt <-> ECEF <-> local ENU frame.

const a = 6378137.0;             // WGS84 semi-major axis
const f = 1 / 298.257223563;     // WGS84 flattening
const b = a * (1 - f);           // semi-minor axis
const e2 = (a * a - b * b) / (a * a);
const e_prime2 = (a * a - b * b) / (b * b);

export function lonLatAltToECEF(lon, lat, alt) {
  const lonRad = (lon * Math.PI) / 180;
  const latRad = (lat * Math.PI) / 180;
  const sinLat = Math.sin(latRad);
  const cosLat = Math.cos(latRad);
  const N = a / Math.sqrt(1 - e2 * sinLat * sinLat);
  const x = (N + alt) * cosLat * Math.cos(lonRad);
  const y = (N + alt) * cosLat * Math.sin(lonRad);
  const z = (N * (1 - e2) + alt) * sinLat;
  return { x, y, z };
}

export function ecefToLonLatAlt(x, y, z) {
  const p = Math.sqrt(x * x + y * y);
  const lon = Math.atan2(y, x);
  let lat = Math.atan2(z, p * (1 - e2));
  for (let i = 0; i < 10; i++) {
    const sinLat = Math.sin(lat);
    const N = a / Math.sqrt(1 - e2 * sinLat * sinLat);
    lat = Math.atan2(z + e2 * N * sinLat, p);
  }
  const sinLat = Math.sin(lat);
  const N = a / Math.sqrt(1 - e2 * sinLat * sinLat);
  const alt = p / Math.cos(lat) - N;
  return {
    lon: (lon * 180) / Math.PI,
    lat: (lat * 180) / Math.PI,
    alt
  };
}

export function buildLocalFrame(lon, lat, alt) {
  // Build a 4x4 column-major matrix converting local ENU coords to ECEF.
  const lonRad = (lon * Math.PI) / 180;
  const latRad = (lat * Math.PI) / 180;
  const sinLon = Math.sin(lonRad), cosLon = Math.cos(lonRad);
  const sinLat = Math.sin(latRad), cosLat = Math.cos(latRad);

  // East unit vector in ECEF
  const east = [-sinLon, cosLon, 0];
  // North unit vector in ECEF
  const north = [-sinLat * cosLon, -sinLat * sinLon, cosLat];
  // Up unit vector in ECEF
  const up = [cosLat * cosLon, cosLat * sinLon, sinLat];

  const origin = lonLatAltToECEF(lon, lat, alt);
  return {
    origin,
    east,
    north,
    up,
    // convert local (e, n, u) to ECEF
    toECEF(e, n, u) {
      return {
        x: origin.x + e * east[0] + n * north[0] + u * up[0],
        y: origin.y + e * east[1] + n * north[1] + u * up[1],
        z: origin.z + e * east[2] + n * north[2] + u * up[2]
      };
    },
    // convert ECEF to local (e, n, u)
    toLocal(x, y, z) {
      const dx = x - origin.x;
      const dy = y - origin.y;
      const dz = z - origin.z;
      return {
        e: dx * east[0] + dy * east[1] + dz * east[2],
        n: dx * north[0] + dy * north[1] + dz * north[2],
        u: dx * up[0] + dy * up[1] + dz * up[2]
      };
    }
  };
}

// Estimate the lon/lat of a Cartesian point in ECEF (without alt)
export function ecefToLonLat(x, y, z) {
  const r = ecefToLonLatAlt(x, y, z);
  return { lon: r.lon, lat: r.lat };
}

// Compute distance in meters between two ECEF points
export function ecefDistance(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  const dz = a.z - b.z;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}
