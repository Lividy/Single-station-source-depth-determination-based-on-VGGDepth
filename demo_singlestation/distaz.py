import math

class DistAz:
    """Class to calculate the Great Circle Arc distance
    between two sets of geographic coordinates"""

    def __init__(self, stalon, stalat, evtlon, evtlat):
        """
        Initialize with the coordinates of two points:
        stalon => Longitude of first point (+E, -W) in degrees
        stalat => Latitude of first point (+N, -S) in degrees
        evtlon => Longitude of second point (+E, -W) in degrees
        evtlat => Latitude of second point (+N, -S) in degrees
        """
        self.stalat = stalat
        self.stalon = stalon
        self.evtlat = evtlat
        self.evtlon = evtlon

        # If the two points are identical, return 0 for all parameters
        if (stalat == evtlat) and (stalon == evtlon):
            self.delta = 0.0
            self.az = 0.0
            self.baz = 0.0
            return

        # Convert degrees to radians
        rad = 2. * math.pi / 360.0
        sph = 1.0 / 298.257  # Earth Flattening

        # Compute colatitudes
        scolat = math.pi / 2.0 - math.atan((1. - sph) * (1. - sph) * math.tan(stalat * rad))
        ecolat = math.pi / 2.0 - math.atan((1. - sph) * (1. - sph) * math.tan(evtlat * rad))
        
        # Convert longitudes to radians
        slon = stalon * rad
        elon = evtlon * rad

        # Calculate terms for first point
        a = math.sin(scolat) * math.cos(slon)
        b = math.sin(scolat) * math.sin(slon)
        c = math.cos(scolat)
        d = math.sin(slon)
        e = -math.cos(slon)
        g = -c * e
        h = c * d
        k = -math.sin(scolat)

        # Calculate terms for second point
        aa = math.sin(ecolat) * math.cos(elon)
        bb = math.sin(ecolat) * math.sin(elon)
        cc = math.cos(ecolat)
        dd = math.sin(elon)
        ee = -math.cos(elon)
        gg = -cc * ee
        hh = cc * dd
        kk = -math.sin(ecolat)

        # Bullen's equation for great circle arc distance
        delrad = math.acos(a * aa + b * bb + c * cc)
        self.delta = delrad / rad

        # Calculate back azimuth
        rhs1 = (aa - d) * (aa - d) + (bb - e) * (bb - e) + cc * cc - 2.
        rhs2 = (aa - g) * (aa - g) + (bb - h) * (bb - h) + (cc - k) * (cc - k) - 2.
        dbaz = math.atan2(rhs1, rhs2)
        if dbaz < 0.0:
            dbaz = dbaz + 2 * math.pi
        self.baz = dbaz / rad

        # Calculate azimuth
        rhs1 = (a - dd) * (a - dd) + (b - ee) * (b - ee) + c * c - 2.
        rhs2 = (a - gg) * (a - gg) + (b - hh) * (b - hh) + (c - kk) * (c - kk) - 2.
        daz = math.atan2(rhs1, rhs2)
        if daz < 0.0:
            daz = daz + 2 * math.pi
        self.az = daz / rad

        # Ensure azimuth and back azimuth are in range [0, 360)
        if abs(self.baz - 360.0) < 0.00001:
            self.baz = 0.0
        if abs(self.az - 360.0) < 0.00001:
            self.az = 0.0

    def getDelta(self):
        """Returns the Great Circle Arc distance in degrees"""
        return self.delta

    def getAz(self):
        """Returns the azimuth (degrees) from point 1 to point 2"""
        return self.az

    def getBaz(self):
        """Returns the back azimuth (degrees) from point 2 to point 1"""
        return self.baz

    def degreesToKilometers(self,degrees):
        """Converts degrees to kilometers (1 degree ~ 111.19 km)"""
        return degrees * 111.19
    def getDistanceKm(self):
        """Converts degrees to kilometers (1 degree ~ 111.19 km)"""
        return self.delta * 111.19    

    def kilometersToDegrees(self, kilometers):
        """Converts kilometers to degrees"""
        return kilometers / 111.19

