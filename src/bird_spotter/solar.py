"""Solar Dawn and Sunset Scheduler for Bird Spotter.

Automatically determines daily dawn/sunrise and sunset times from the internet
(with offline NOAA astronomical math fallback) to sleep 30 minutes after sunset
and wake 30 minutes before dawn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
import json
import math
import threading
import time
from typing import Any
import urllib.error
import urllib.request


def calculate_solar_times_noaa(
    target_date: date,
    lat: float = 52.3555,
    lon: float = -1.1743,
    tz_offset_hours: float | None = None,
) -> tuple[datetime, datetime, datetime]:
    """Calculate dawn (civil twilight begin), sunrise, and sunset using standard NOAA solar calculations."""
    if tz_offset_hours is None:
        now = datetime.now().astimezone()
        tz_offset = now.utcoffset()
        tz_offset_hours = tz_offset.total_seconds() / 3600.0 if tz_offset else 0.0

    def _calc_time(zenith: float, is_rise: bool) -> tuple[int, int] | None:
        n = target_date.timetuple().tm_yday
        h0 = 6.0 if is_rise else 18.0
        t = n + ((h0 - (lon / 15.0)) / 24.0)

        # Sun's mean anomaly
        m = (0.9856 * t) - 3.289
        m_rad = math.radians(m)

        # Sun's true longitude
        l = (m + (1.916 * math.sin(m_rad)) + (0.020 * math.sin(2.0 * m_rad)) + 282.634) % 360.0
        l_rad = math.radians(l)

        # Sun's right ascension
        ra = math.degrees(math.atan(0.91764 * math.tan(l_rad))) % 360.0
        l_quadrant = (math.floor(l / 90.0)) * 90.0
        ra_quadrant = (math.floor(ra / 90.0)) * 90.0
        ra = (ra + (l_quadrant - ra_quadrant)) / 15.0

        # Sun's declination
        sin_dec = 0.39782 * math.sin(l_rad)
        cos_dec = math.cos(math.asin(sin_dec))

        # Local hour angle
        lat_rad = math.radians(lat)
        zenith_rad = math.radians(zenith)
        cos_h = (math.cos(zenith_rad) - (sin_dec * math.sin(lat_rad))) / (
            cos_dec * math.cos(lat_rad)
        )

        if cos_h > 1.0 or cos_h < -1.0:
            return None

        h = (360.0 - math.degrees(math.acos(cos_h))) if is_rise else math.degrees(math.acos(cos_h))
        h = h / 15.0

        t_val = h + ra - (0.06571 * t) - 6.622
        ut = (t_val - (lon / 15.0)) % 24.0
        local_hours = (ut + tz_offset_hours) % 24.0

        hh = int(local_hours)
        mm = int(round((local_hours - hh) * 60.0))
        if mm >= 60:
            hh = (hh + 1) % 24
            mm = 0
        return hh, mm

    # Zenith: 90.833 for sunrise/sunset (90 deg 50 min), 96.0 for civil twilight begin
    sr = _calc_time(90.833, is_rise=True) or (7, 30)
    ss = _calc_time(90.833, is_rise=False) or (16, 30)
    dawn = _calc_time(96.0, is_rise=True) or (6, 50)

    local_tz = datetime.now().astimezone().tzinfo
    dawn_dt = datetime(
        target_date.year, target_date.month, target_date.day, dawn[0], dawn[1], tzinfo=local_tz
    )
    sr_dt = datetime(
        target_date.year, target_date.month, target_date.day, sr[0], sr[1], tzinfo=local_tz
    )
    ss_dt = datetime(
        target_date.year, target_date.month, target_date.day, ss[0], ss[1], tzinfo=local_tz
    )
    return dawn_dt, sr_dt, ss_dt


@dataclass
class SolarScheduler:
    """Thread-safe scheduler for sunset sleep (+30m) and dawn wake (-30m)."""

    enabled: bool = True
    auto_location: bool = True
    latitude: float = 52.3555
    longitude: float = -1.1743
    location_name: str = "UK"
    wake_before_dawn_min: int = 30
    sleep_after_sunset_min: int = 30

    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _cached_date: date | None = field(default=None, init=False, repr=False)
    _dawn_dt: datetime | None = field(default=None, init=False, repr=False)
    _sunrise_dt: datetime | None = field(default=None, init=False, repr=False)
    _sunset_dt: datetime | None = field(default=None, init=False, repr=False)
    _wake_dt: datetime | None = field(default=None, init=False, repr=False)
    _sleep_dt: datetime | None = field(default=None, init=False, repr=False)
    _source: str = field(default="init", init=False, repr=False)
    _last_window_is_night: bool | None = field(default=None, init=False, repr=False)
    _is_refreshing: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        # Initialize with local astronomical calculation immediately so schedule is never empty
        today = datetime.now().astimezone().date()
        self._calculate_fallback(today)

    def _calculate_fallback(self, target_date: date) -> None:
        """Calculate solar schedule using local NOAA astronomical formulas."""
        dawn, sr, ss = calculate_solar_times_noaa(
            target_date, lat=self.latitude, lon=self.longitude
        )
        with self._lock:
            self._cached_date = target_date
            self._dawn_dt = dawn
            self._sunrise_dt = sr
            self._sunset_dt = ss
            self._wake_dt = dawn - timedelta(minutes=self.wake_before_dawn_min)
            self._sleep_dt = ss + timedelta(minutes=self.sleep_after_sunset_min)
            self._source = "astronomical_fallback"

    def detect_location(self) -> bool:
        """Attempt to detect latitude and longitude from IP geolocation services."""
        if not self.auto_location:
            return False

        endpoints = [
            ("http://ip-api.com/json/?fields=status,lat,lon,city,country", 3.5),
            ("https://ipapi.co/json/", 3.5),
        ]
        for url, timeout in endpoints:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "BirdSpotter/1.0"})
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    if resp.status == 200:
                        data = json.loads(resp.read().decode("utf-8"))
                        lat = data.get("lat") or data.get("latitude")
                        lon = data.get("lon") or data.get("longitude")
                        city = data.get("city") or data.get("country") or "Auto"
                        if lat is not None and lon is not None:
                            with self._lock:
                                self.latitude = float(lat)
                                self.longitude = float(lon)
                                self.location_name = str(city)
                            return True
            except Exception:
                continue
        return False

    def fetch_internet_solar_times(self, target_date: date) -> bool:
        """Fetch accurate solar times from api.sunrise-sunset.org."""
        with self._lock:
            lat = self.latitude
            lon = self.longitude

        url = (
            f"https://api.sunrise-sunset.org/json?lat={lat:.4f}&lng={lon:.4f}"
            f"&date={target_date.isoformat()}&formatted=0"
        )
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "BirdSpotter/1.0"})
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    if data.get("status") == "OK" and "results" in data:
                        res = data["results"]
                        # Parse UTC timestamps and convert to local timezone
                        sr_str = res.get("sunrise")
                        ss_str = res.get("sunset")
                        dawn_str = res.get("civil_twilight_begin") or sr_str

                        if sr_str and ss_str and dawn_str:
                            sr_dt = datetime.fromisoformat(sr_str).astimezone()
                            ss_dt = datetime.fromisoformat(ss_str).astimezone()
                            dawn_dt = datetime.fromisoformat(dawn_str).astimezone()

                            with self._lock:
                                self._cached_date = target_date
                                self._dawn_dt = dawn_dt
                                self._sunrise_dt = sr_dt
                                self._sunset_dt = ss_dt
                                self._wake_dt = dawn_dt - timedelta(
                                    minutes=self.wake_before_dawn_min
                                )
                                self._sleep_dt = ss_dt + timedelta(
                                    minutes=self.sleep_after_sunset_min
                                )
                                self._source = "internet"
                            return True
        except Exception:
            pass
        return False

    def refresh(self, target_date: date | None = None) -> bool:
        """Refresh solar times for target date (trying internet then fallback)."""
        dt_target = target_date or datetime.now().astimezone().date()
        if self.auto_location:
            self.detect_location()

        ok = self.fetch_internet_solar_times(dt_target)
        if not ok:
            self._calculate_fallback(dt_target)
        return ok

    def refresh_async(self) -> None:
        """Run non-blocking background refresh so video streaming loop is never delayed."""
        with self._lock:
            if self._is_refreshing:
                return
            self._is_refreshing = True

        def _worker() -> None:
            try:
                self.refresh()
            finally:
                with self._lock:
                    self._is_refreshing = False

        threading.Thread(target=_worker, daemon=True).start()

    def _ensure_fresh(self, now_dt: datetime) -> None:
        today = now_dt.date()
        with self._lock:
            cached = self._cached_date
        if cached != today:
            self._calculate_fallback(today)
            self.refresh_async()

    def is_night_sleep_window(self, now_dt: datetime | None = None) -> bool:
        """Return True if now_dt falls inside the night sleep window (sleep_dt <= now or now < wake_dt)."""
        now = datetime.now().astimezone() if now_dt is None else now_dt
        if now.tzinfo is None:
            now = now.replace(tzinfo=datetime.now().astimezone().tzinfo)

        self._ensure_fresh(now)
        with self._lock:
            wake_dt = self._wake_dt
            sleep_dt = self._sleep_dt

        if wake_dt is None or sleep_dt is None:
            return False

        # Sleep window is from sleep_dt (evening) until wake_dt (next morning)
        # Active day window is wake_dt <= now < sleep_dt
        is_day = wake_dt <= now < sleep_dt
        return not is_day

    def check_transition(self, now_dt: datetime | None = None) -> str | None:
        """Check if a boundary between day and night was crossed.

        Returns 'sleep' when entering the night window,
        'wake' when entering the daytime window, or None if no transition.
        """
        if not self.enabled:
            return None

        now = datetime.now().astimezone() if now_dt is None else now_dt
        if now.tzinfo is None:
            now = now.replace(tzinfo=datetime.now().astimezone().tzinfo)

        is_night = self.is_night_sleep_window(now)

        with self._lock:
            last = self._last_window_is_night
            if last is None or is_night != last:
                self._last_window_is_night = is_night
                return "sleep" if is_night else "wake"
        return None

    def force_recheck(self) -> None:
        """Reset transition tracker to force immediate transition evaluation."""
        with self._lock:
            self._last_window_is_night = None

    def get_status_dict(self, now_dt: datetime | None = None) -> dict[str, Any]:
        """Return status dictionary for dashboard telemetry and UI."""
        now = datetime.now().astimezone() if now_dt is None else now_dt
        if now.tzinfo is None:
            now = now.replace(tzinfo=datetime.now().astimezone().tzinfo)

        self._ensure_fresh(now)
        is_night = self.is_night_sleep_window(now)

        with self._lock:
            dawn = self._dawn_dt.strftime("%H:%M") if self._dawn_dt else "--:--"
            sunrise = self._sunrise_dt.strftime("%H:%M") if self._sunrise_dt else "--:--"
            sunset = self._sunset_dt.strftime("%H:%M") if self._sunset_dt else "--:--"
            wake = self._wake_dt.strftime("%H:%M") if self._wake_dt else "--:--"
            sleep = self._sleep_dt.strftime("%H:%M") if self._sleep_dt else "--:--"
            return {
                "enabled": self.enabled,
                "is_night": is_night,
                "location": self.location_name,
                "dawn": dawn,
                "sunrise": sunrise,
                "sunset": sunset,
                "wake_time": wake,
                "sleep_time": sleep,
                "source": self._source,
            }
