import contextlib
import os
from functools import lru_cache

import ipywidgets

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import ipyvolume as ipv

import galsim
import batoid


@lru_cache
def get_constellations_xyz():
    return fits.getdata(
        os.path.join(batoid.datadir, 'misc', 'constellations.fits')
    )


@lru_cache
def get_stars_xyzs(maglim=5.5):
    table = fits.getdata(
        os.path.join(batoid.datadir, 'misc', 'BSC5.fits')
    )
    table = table[table['mag'] < maglim]
    x = np.cos(table['ra']) * np.cos(table['dec'])
    y = np.sin(table['ra']) * np.cos(table['dec'])
    z = np.sin(table['dec'])
    return x, y, z, 10**(3-0.2*table['mag'])


def get_zk(opd):
    xs = np.linspace(-1, 1, opd.shape[0])
    ys = np.linspace(-1, 1, opd.shape[1])
    xs, ys = np.meshgrid(xs, ys)
    w = ~opd.mask
    basis = galsim.zernike.zernikeBasis(22, xs[w], ys[w], R_inner=0.61)
    zk, *_ = np.linalg.lstsq(basis.T, opd[w], rcond=None)
    return zk


def sub_ptt(opd):
    xs = np.linspace(-1, 1, opd.shape[0])
    ys = np.linspace(-1, 1, opd.shape[1])
    xs, ys = np.meshgrid(xs, ys)
    zk = get_zk(opd)
    opd -= galsim.zernike.Zernike(zk[:4], R_inner=0.61)(xs, ys)
    return opd


def lst(jd, lon_deg):
    """Local Sidereal Time from julian day and longitude"""
    T = jd / 36525
    return (
        280.46061837
        + 360.98564736629*(jd - 2451545)
        + 0.000387933*T*T
        - T*T*T/38710000
        + lon_deg
    ) % 360


def eq_to_az(ra_deg, dec_deg, jd, lat_deg, lon_deg):
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    ha = np.deg2rad(lst(jd, lon_deg)) - ra
    lat = np.deg2rad(lat_deg)
    sinlat = np.sin(lat)
    coslat = np.cos(lat)
    sindec = np.sin(dec)

    sinalt = sindec*sinlat + np.cos(dec)*coslat*np.cos(ha)
    alt = np.arcsin(sinalt)

    sinaz = -np.sin(ha)*np.cos(dec) / np.cos(alt)
    cosaz = (sindec - sinalt*sinlat)/(np.cos(alt)*coslat)
    az = np.arctan2(sinaz, cosaz)
    return np.rad2deg(alt), np.rad2deg(az)


def az_to_eq(alt_deg, az_deg, jd, lat_deg, lon_deg):
    alt = np.deg2rad(alt_deg)
    az = np.deg2rad(az_deg)
    lat = np.deg2rad(lat_deg)
    sindec = np.sin(alt)*np.sin(lat) + np.cos(alt)*np.cos(lat)*np.cos(az)
    dec = np.arcsin(sindec)
    sinha = -np.sin(az)*np.cos(alt)/np.cos(dec)
    cosha = (np.sin(alt) - np.sin(dec)*np.sin(lat))/(np.cos(dec)*np.cos(lat))
    ha = np.arctan2(sinha, cosha)
    ra = np.deg2rad(lst(jd, lon_deg)) - ha
    return np.rad2deg(ra), np.rad2deg(dec)


class RubinCSApp:
    def __init__(self, maglim=5.5, debug=None):
        self.maglim = maglim
        self.sky_dist = 15000
        self.lat = -30.2446
        self.lon = -70.7494
        # self.fiducial_telescope = batoid.Optic.fromYaml("LSST_r.yaml")
        self.fiducial_telescope = batoid.Optic.fromYaml("LSST_r_align_holes.yaml")
        self.actual_telescope = self.fiducial_telescope
        if debug is None:
            debug = contextlib.redirect_stdout(None)
        self.debug = debug

        # widget variables
        self.clip_horizon = False
        self.show_telescope = True
        self.ra = 186.65  # Approximately Acrux
        self.dec = -63.1
        self.jd = 2460676.25  # Start of 2025
        self.rtp = 0.0

        # compute
        self.lst = lst(self.jd, self.lon)
        self.alt, self.az = eq_to_az(self.ra, self.dec, self.jd, self.lat, self.lon)

        self.thx = 0.0
        self.thy = 0.0
        self.z_offset = 0.0
        self.show_rays = True
        self.show_CCS = False
        self.show_OCS = False
        self.show_ZCS = False
        self.show_DVCS = False
        self.show_EDCS = True
        self.noll = 4
        self.perturb = 0.0

        # IPV Scatters
        self.constellations = self._constellations_view()
        self.stars = self._stars_view()
        self.telescope = self._telescope_view()
        self.fp = self._fp_view()
        self.azimuth_ring = self._azimuth_ring_view()
        self.elevation_bearings = self._elevation_bearings_view()
        self.rays = self._rays_view()
        self.CCS = self._ccs_views()
        self.OCS = self._ocs_views()
        self.ZCS = self._zcs_views()
        self.EDCS = self._edcs_views()
        self.DVCS = self._dvcs_views()

        # Matplotlib
        self.mpl = self._mpl_view()

        # Controls
        kwargs = {'layout':{'width':'190px'}}
        self.ra_control = ipywidgets.FloatText(value=self.ra, step=3.0, description='ra (deg)', **kwargs)
        self.dec_control = ipywidgets.FloatText(value=self.dec, step=3.0, description='dec (deg)', **kwargs)
        self.jd_control = ipywidgets.FloatText(value=self.jd, step=0.01, description="JD", **kwargs)
        self.alt_control = ipywidgets.FloatText(value=self.alt, step=3.0, description='alt (deg)', **kwargs)
        self.az_control = ipywidgets.FloatText(value=self.az, step=3.0, description='az (deg)', **kwargs)

        self.rtp_control = ipywidgets.FloatText(value=0.0, step=3.0, description='RTP (deg)', **kwargs)
        self.thx_control = ipywidgets.FloatText(value=0.0, step=0.25, description='Field x (deg)', **kwargs)
        self.thy_control = ipywidgets.FloatText(value=0.0, step=0.25, description='Field y (deg)', **kwargs)
        self.z_control = ipywidgets.FloatText(value=0.0, step=0.1, description="Det z (mm)", **kwargs)
        self.telescope_control = ipywidgets.Checkbox(value=self.show_telescope, description="telescope", **kwargs)
        self.horizon_control = ipywidgets.Checkbox(value=self.clip_horizon, description='horizon', **kwargs)
        self.rays_control = ipywidgets.Checkbox(value=self.show_rays, description="rays", **kwargs)
        self.CCS_control = ipywidgets.Checkbox(value=self.show_CCS, description='CCS', **kwargs)
        self.OCS_control = ipywidgets.Checkbox(value=self.show_OCS, description='OCS', **kwargs)
        self.ZCS_control = ipywidgets.Checkbox(value=self.show_ZCS, description='ZCS', **kwargs)
        self.EDCS_control = ipywidgets.Checkbox(value=self.show_EDCS, description='EDCS', **kwargs)
        self.DVCS_control = ipywidgets.Checkbox(value=self.show_DVCS, description='DVCS', **kwargs)
        self.noll_control = ipywidgets.IntText(value=self.noll, description="Noll idx", **kwargs)
        self.perturb_control = ipywidgets.FloatText(value=self.perturb, step=0.1, description="Pert (µm)", **kwargs)

        # observe
        self.ra_control.observe(self.handle_ra, 'value')
        self.dec_control.observe(self.handle_dec, 'value')
        self.jd_control.observe(self.handle_jd, 'value')
        self.alt_control.observe(self.handle_alt, 'value')
        self.az_control.observe(self.handle_az, 'value')
        self.rtp_control.observe(self.handle_rtp, 'value')
        self.thx_control.observe(self.handle_thx, 'value')
        self.thy_control.observe(self.handle_thy, 'value')
        self.z_control.observe(self.handle_z, 'value')
        self.telescope_control.observe(self.handle_telescope, 'value')
        self.horizon_control.observe(self.handle_horizon, 'value')
        self.rays_control.observe(self.handle_rays, 'value')
        self.CCS_control.observe(self.handle_CCS, 'value')
        self.OCS_control.observe(self.handle_OCS, 'value')
        self.ZCS_control.observe(self.handle_ZCS, 'value')
        self.EDCS_control.observe(self.handle_EDCS, 'value')
        self.DVCS_control.observe(self.handle_DVCS, 'value')
        self.noll_control.observe(self.handle_noll, 'value')
        self.perturb_control.observe(self.handle_perturb, 'value')

        self.scatters = [
            self.constellations,
            self.stars,
            self.telescope,
            self.fp,
            self.azimuth_ring,
            self.elevation_bearings,
            self.rays,
            *self.CCS,
            *self.OCS,
            *self.ZCS,
            *self.EDCS,
            *self.DVCS,
        ]

        self.controls = [
            self.ra_control,
            self.dec_control,
            self.jd_control,
            self.alt_control,
            self.az_control,
            self.rtp_control,
            self.thx_control,
            self.thy_control,
            self.z_control,
            self.horizon_control,
            self.telescope_control,
            self.rays_control,
            self.CCS_control,
            self.OCS_control,
            self.ZCS_control,
            self.EDCS_control,
            self.DVCS_control,
            self.noll_control,
            self.perturb_control
        ]

    def _constellations_xyz(self, lst):
        ctf = batoid.CoordTransform(
            batoid.globalCoordSys,
            batoid.CoordSys(
                (0, 0, 0),
                batoid.RotZ(np.deg2rad(lst+180)) @ batoid.RotY(-np.deg2rad(90-self.lat))
            )
        )
        return ctf.applyForwardArray(*get_constellations_xyz())

    def _constellations_view(self):
        x, y, z = self._constellations_xyz(0.0)
        return ipv.Scatter(
            x=x, y=y, z=z,
            color="blue",
            visible_lines=True,
            color_selected=None,
            size_selected=1,
            size=0,
            connected=True,
            visible_markers=False,
            cast_shadow=True,
            receive_shadow=True
        )

    def _stars_xyzs(self, lst):
        ctf = batoid.CoordTransform(
            batoid.globalCoordSys,
            batoid.CoordSys(
                (0, 0, 0),
                batoid.RotZ(np.deg2rad(lst+180)) @ batoid.RotY(-np.deg2rad(90-self.lat))
            )
        )
        x, y, z, s = get_stars_xyzs(self.maglim)
        x, y, z = ctf.applyForwardArray(x, y, z)
        return x, y, z, s

    def _stars_view(self):
        x, y, z, s = self._stars_xyzs(0.0)
        x *= self.sky_dist
        y *= self.sky_dist
        z *= self.sky_dist
        return ipv.Scatter(
            x=x, y=y, z=z,
            color="white",
            visible_lines=True,
            color_selected=None,
            size_selected=1,
            size=s,
            connected=False,
            visible_markers=False,
            cast_shadow=True,
            receive_shadow=True,
            geo='sphere'
        )

    def _telescope_xyz(self, alt, az, rtp):
        M1_height = 3.53  # LTS-213
        elev_axis_height = 5.425  # LTS-213
        telescope = self.fiducial_telescope
        # Shift upwards to place M1 vertex prescribed distance above the ground
        telescope = telescope.withGlobalShift([0, 0, M1_height])
        # Apply camera rotator
        telescope = telescope.withLocallyRotatedOptic("LSSTCamera", batoid.RotZ(np.deg2rad(rtp)))
        # Apply Azimuth
        telescope = telescope.withLocalRotation(batoid.RotZ(np.deg2rad(90-az)))
        # Apply elevation.  Note height of elevation axis.
        telescope = telescope.withLocalRotation(
            batoid.RotX(np.deg2rad(90-alt)),
            rotCenter=[0, 0, elev_axis_height],
            coordSys=batoid.globalCoordSys
        )
        self.actual_telescope = telescope
        return self.actual_telescope.get3dmesh()

    def _telescope_view(self):
        x, y, z = self._telescope_xyz(self.alt, self.az, self.rtp)
        return ipv.Scatter(
            x=x, y=y, z=z,
            color="white",
            visible_lines=True,
            color_selected=None,
            size_selected=1,
            size=0,
            connected=True,
            visible_markers=False,
            cast_shadow=True,
            receive_shadow=True
        )

    def _azimuth_ring_view(self):
        th = np.linspace(0, 2*np.pi, 100)
        x_, y_ = np.cos(th), np.sin(th)
        xs = []
        ys = []
        zs = []
        for d in [-0.1, 0.1]:
            for r in [4.5, 5.0]:
                x = x_*r
                y = y_*r
                z = np.full_like(x, d)
                xs.append(x)
                ys.append(y)
                zs.append(z)
                xs.append([np.nan])
                ys.append([np.nan])
                zs.append([np.nan])
        xs = np.hstack(xs)
        ys = np.hstack(ys)
        zs = np.hstack(zs)
        return ipv.Scatter(
            x=xs, y=ys, z=zs,
            color="white",
            visible_lines=True,
            color_selected=None,
            size_selected=0,
            size=0,
            connected=True,
            visible_markers=False,
            cast_shadow=True,
            receive_shadow=True
        )

    def _elevation_bearings_xyz(self, az):
        th = np.linspace(0, 2*np.pi, 100)
        x_, y_ = np.cos(th), np.sin(th)
        xs = []
        ys = []
        zs = []
        for d in [4.4, 4.5]:
            for r in [1.5, 1.75]:
                x = np.full_like(th, -d)
                y = x_*r
                z = y_*r
                z += 5.425  # height of elevation axis above azimuth ring
                c = np.cos(np.deg2rad(90-az))
                s = np.sin(np.deg2rad(90-az))
                x, y = c*x - s*y, s*x + c*y
                xs.append(x)
                ys.append(y)
                zs.append(z)
                xs.append([np.nan])
                ys.append([np.nan])
                zs.append([np.nan])
                xs.append(-x)
                ys.append(-y)
                zs.append(z)
                xs.append([np.nan])
                ys.append([np.nan])
                zs.append([np.nan])
        return np.hstack(xs), np.hstack(ys), np.hstack(zs)

    def _elevation_bearings_view(self):
        x, y, z = self._elevation_bearings_xyz(45.0)
        return ipv.Scatter(
            x=x, y=y, z=z,
            color="white",
            visible_lines=True,
            color_selected=None,
            size_selected=0,
            size=0,
            connected=True,
            visible_markers=False,
            cast_shadow=True,
            receive_shadow=True
        )

    def _rays_xyz(self, thx, thy):
        # Assume that self.actual_telescope is correct;
        # so need to update telescope before rays when changing alt,az,rtp
        rays = batoid.RayVector.asFan(
            optic=self.actual_telescope,
            wavelength=620e-9,
            theta_x=np.deg2rad(thx), theta_y=np.deg2rad(thy),
            nx=30, ny=30,
            lx=8.359, ly=8.359
        )
        tf = self.actual_telescope.traceFull(rays)
        xyz = []
        for key, surface in tf.items():
            if len(xyz) == 0:
                xyz.append(surface['in'].propagate(-self.sky_dist).toCoordSys(batoid.globalCoordSys).r)
            out = surface['out'].toCoordSys(batoid.globalCoordSys)
            r = out.r
            r[out.vignetted] = np.nan
            xyz.append(r)

        xyz = np.array(xyz)
        xyz = np.concatenate([xyz, np.full((1,)+xyz.shape[1:3], np.nan)])
        xyz = np.array(xyz)
        xyz = np.transpose(xyz, axes=[1,0,2])
        xyz = np.vstack(xyz).T
        return xyz

    def _rays_view(self):
        x, y, z = self._rays_xyz(0.0, 0.0)
        return ipv.Scatter(
            x=x, y=y, z=z,
            color="yellow",
            visible_lines=True,
            color_selected=None,
            size_selected=0,
            size=0,
            connected=True,
            visible_markers=False,
            cast_shadow=True,
            receive_shadow=True
        )

    def _fp_xyz(self):
        # Assume self.actual_telescope is up-to-date.
        raft_det = 0.31527*2/5
        raft_sky = self.sky_dist * np.deg2rad(0.7)
        detector_ctf = batoid.CoordTransform(
            self.actual_telescope['Detector'].coordSys,
            batoid.globalCoordSys
        )
        # Sky CoordSys can be obtained from detector coordsys by
        # rotating 180 degrees about the detector Z, and then
        # placing the origin along Z out sky_dist meters.
        sky_coordSys = self.actual_telescope['Detector'].coordSys
        sky_coordSys = sky_coordSys.rotateLocal(batoid.RotZ(np.pi))
        sky_coordSys.origin = sky_coordSys.zhat * self.sky_dist
        sky_ctf = batoid.CoordTransform(
            sky_coordSys,
            batoid.globalCoordSys
        )

        # Build in xy-plane in units of rafts to start
        xs = []
        ys = []

        # top hline
        xs.extend([-1.5, 1.5, np.nan])
        ys.extend([2.5, 2.5, np.nan])
        # Middle hlines
        for y in [1.5, 0.5, -0.5, -1.5]:
            xs.extend([-2.5, 2.5, np.nan])
            ys.extend([y, y, np.nan])
        # bottom hline
        xs.extend([-1.5, 1.5, np.nan])
        ys.extend([-2.5, -2.5, np.nan])
        # left vline
        xs.extend([-2.5, -2.5, np.nan])
        ys.extend([-1.5, 1.5, np.nan])
        # middle vlines
        for x in [1.5, 0.5, -0.5, -1.5]:
            xs.extend([x, x, np.nan])
            ys.extend([-2.5, 2.5, np.nan])
        # right vline
        xs.extend([2.5, 2.5, np.nan])
        ys.extend([-1.5, 1.5, np.nan])
        xs = np.array(xs)
        ys = np.array(ys)
        zs = np.zeros_like(xs)

        # Now do the projection on the sky.  It's technically a 180-degree rotation,
        # but we'll ignore that for now since the pattern is rotationally symmetric.
        xfp, yfp, zfp = detector_ctf.applyForwardArray(xs*raft_det, ys*raft_det, zs)
        xsky, ysky, zsky = sky_ctf.applyForwardArray(xs*raft_sky, ys*raft_sky, zs)

        return (
            np.hstack([xfp, [np.nan], xsky]),
            np.hstack([yfp, [np.nan], ysky]),
            np.hstack([zfp, [np.nan], zsky])
        )

    def _fp_view(self):
        x, y, z = self._fp_xyz()
        return ipv.Scatter(
            x=x, y=y, z=z,
            color="red",
            visible_lines=True,
            color_selected=None,
            size_selected=0,
            size=0,
            connected=True,
            visible_markers=False,
            cast_shadow=True,
            receive_shadow=True
        )

    def _cs_xyz(self, coordSys, length=2):
        p0 = coordSys.origin
        px = p0 + coordSys.xhat * length
        py = p0 + coordSys.yhat * length
        pz = p0 + coordSys.zhat * length
        return (
            np.vstack([p0, px]).T,
            np.vstack([p0, py]).T,
            np.vstack([p0, pz]).T
        )

    def _cs_views(self, coordSys, length=2):
        x_xyz, y_xyz, z_xyz = self._cs_xyz(coordSys, length=length)

        xscat = ipv.Scatter(
            x=x_xyz[0], y=x_xyz[1], z=x_xyz[2],
            color="cyan",
            visible_lines=True,
            color_selected=None,
            size_selected=1,
            size=0,
            connected=True,
            visible_markers=False,
            cast_shadow=True,
            receive_shadow=True
        )
        yscat = ipv.Scatter(
            x=y_xyz[0], y=y_xyz[1], z=y_xyz[2],
            color="magenta",
            visible_lines=True,
            color_selected=None,
            size_selected=1,
            size=0,
            connected=True,
            visible_markers=False,
            cast_shadow=True,
            receive_shadow=True
        )
        zscat = ipv.Scatter(
            x=z_xyz[0], y=z_xyz[1], z=z_xyz[2],
            color="yellow",
            visible_lines=True,
            color_selected=None,
            size_selected=1,
            size=0,
            connected=True,
            visible_markers=False,
            cast_shadow=True,
            receive_shadow=True
        )
        return xscat, yscat, zscat

    def _ccs_views(self):
        # Assume actual_telescope is up-to-date
        return self._cs_views(self.actual_telescope['LSSTCamera'].coordSys, 2)

    def _ocs_views(self):
        # Assume actual_telescope is up-to-date
        return self._cs_views(self.actual_telescope['M1'].coordSys, 3)

    def _zcs_views(self):
        # ZCS is OCS rotated 180 about y
        rot = batoid.RotY(np.pi)
        cs = self.actual_telescope['M1'].coordSys.rotateLocal(rot)
        return self._cs_views(cs, 3)

    def _edcs_views(self):
        # Use two views here, one for detector plane and one for sky
        det_coordSys = self.actual_telescope['Detector'].coordSys
        sky_coordSys = det_coordSys.rotateLocal(batoid.RotZ(np.pi))
        sky_coordSys.origin = sky_coordSys.zhat * self.sky_dist

        edcs_detector = self._cs_views(det_coordSys, 2)
        edcs_sky = self._cs_views(sky_coordSys, np.deg2rad(5.0)*self.sky_dist)
        return edcs_detector + edcs_sky

    def _dvcs_views(self):
        # Use two views here, one for detector plane and one for sky
        # Start with same coordSys as EDCS.
        det_coordSys = self.actual_telescope['Detector'].coordSys
        sky_coordSys = det_coordSys.rotateLocal(batoid.RotZ(np.pi))
        sky_coordSys.origin = sky_coordSys.zhat * self.sky_dist
        # Then transpose x, y, which inverts z too.  We can implement this as a
        # 180 degree rotation around x, followed by 90 rotation around z.
        rot = batoid.RotZ(np.pi/2)@batoid.RotX(np.pi)
        det_coordSys = det_coordSys.rotateLocal(rot)
        sky_coordSys = sky_coordSys.rotateLocal(rot)

        dvcs_detector = self._cs_views(det_coordSys, 2)
        dvcs_sky = self._cs_views(sky_coordSys, np.deg2rad(5.0)*self.sky_dist)
        return dvcs_detector + dvcs_sky

    def _mpl_view(self):
        self._mpl_fig, axes = plt.subplots(
            nrows=3, ncols=1,
            figsize=(2, 6.2), dpi=100,
            facecolor='k'
        )
        self._mpl_canvas = self._mpl_fig.canvas
        self._mpl_canvas.header_visible=False

        # Spot diagram
        self.spot_ax = axes[0]
        self.spot_scatter = self.spot_ax.scatter(
            [], [], s=0.1, c=[], cmap='plasma',
            vmin=0.0, vmax=1.0, # remember to scale data to this range
        )
        self.spot_ax.set_xlim(1, -1)
        self.spot_ax.set_ylim(-1, 1)

        # Add EDCS rose to spot diagram
        edcs_x = self.spot_ax.arrow(-0.9, -0.9, 1.0, 0.0, width=0.02, color='cyan')
        edcs_y = self.spot_ax.arrow(-0.9, -0.9, 0.0, 1.0, width=0.02, color='magenta')
        edcs_x_text = self.spot_ax.text(1.0, -0.9, "EDCS +x", color='cyan', fontsize=8)
        edcs_y_text = self.spot_ax.text(-0.9, 0.3, "EDCS +y", color='magenta', fontsize=8, rotation='vertical')
        self.EDCS_mpl = (edcs_x, edcs_y, edcs_x_text, edcs_y_text)

        # Add DVCS rose to spot diagram.  This is just x/y transpose of EDCS
        dvcs_y = self.spot_ax.arrow(-0.9, -0.9, 1.0, 0.0, width=0.02, color='magenta')
        dvcs_x = self.spot_ax.arrow(-0.9, -0.9, 0.0, 1.0, width=0.02, color='cyan')
        dvcs_y_text = self.spot_ax.text(1.0, -0.9, "DVCS +y", color='magenta', fontsize=8)
        dvcs_x_text = self.spot_ax.text(-0.9, 0.3, "DVCS +x", color='cyan', fontsize=8, rotation='vertical')
        self.DVCS_mpl = (dvcs_x, dvcs_y, dvcs_x_text, dvcs_y_text)

        # Wavefront
        self.wf_ax = axes[1]
        self.wf_imshow = self.wf_ax.imshow(
            np.zeros((256, 256)),
            vmin=-1.0, vmax=1.0,
            cmap='seismic',
            extent=[4.18, -4.18, -4.18, 4.18]  # +x goes to the left in OCS
        )
        # Add OCS rose.
        ocs_x = self.wf_ax.arrow(-4.0, -4.0, 1.0, 0.0, width=0.02, color='cyan')
        ocs_y = self.wf_ax.arrow(-4.0, -4.0, 0.0, 1.0, width=0.02, color='magenta')
        ocs_x_text = self.wf_ax.text(0.0, -4.0, "OCS +x", color='cyan', fontsize=8)
        ocs_y_text = self.wf_ax.text(-4.0, -2.5, "OCS +y", color='magenta', fontsize=8, rotation='vertical')
        self.OCS_mpl = (ocs_x, ocs_y, ocs_x_text, ocs_y_text)

        # Add ZCS rose.
        zcs_x = self.wf_ax.arrow(4.0, -4.0, -1.0, 0.0, width=0.02, color='cyan')
        zcs_y = self.wf_ax.arrow(4.0, -4.0, 0.0, 1.0, width=0.02, color='magenta')
        zcs_x_text = self.wf_ax.text(2.5, -4.0, "ZCS +x", color='cyan', fontsize=8)
        zcs_y_text = self.wf_ax.text(4.0, -2.5, "ZCS +y", color='magenta', fontsize=8, rotation='vertical')
        self.ZCS_mpl = (zcs_x, zcs_y, zcs_x_text, zcs_y_text)

        for ax in axes:
            ax.set_aspect('equal')
            ax.set_facecolor('k')

        out = ipywidgets.Output()
        with out:
            plt.show(self._mpl_fig)
        return out

    def update_altaz(self):
        self.alt, self.az = eq_to_az(
            self.ra, self.dec, self.jd, self.lat, self.lon
        )
        self.alt_control.value = self.alt
        self.az_control.value = self.az

    def update_eq(self):
        self.ra, self.dec = az_to_eq(
            self.alt, self.az, self.jd, self.lat, self.lon
        )
        self.ra_control.value = self.ra
        self.dec_control.value = self.dec

    def handle_ra(self, change):
        self.ra = change['new']
        self.update_altaz()
        self.update_telescope()

    def handle_dec(self, change):
        self.dec = change['new']
        self.update_altaz()
        self.update_telescope()

    def handle_jd(self, change):
        self.jd = change['new']
        self.lst = lst(self.jd, self.lon)
        self.update_eq()
        self.update_constellations()
        self.update_stars()

    def handle_alt(self, change):
        self.alt = change['new']
        self.update_eq()
        self.update_telescope()

    def handle_az(self, change):
        self.az = change['new']
        self.update_eq()
        self.update_telescope()
        self.update_elevation_bearings()

    def handle_rtp(self, change):
        self.rtp = change['new']
        self.update_telescope()
        self.update_spot()
        self.update_wf()
        self.update_CCS()
        self.update_EDCS()
        self.update_DVCS()
        self._mpl_canvas.draw()

    def handle_thx(self, change):
        self.thx = change['new']
        self.update_rays()
        self.update_spot()
        self.update_wf()
        self._mpl_canvas.draw()

    def handle_thy(self, change):
        self.thy = change['new']
        self.update_rays()
        self.update_spot()
        self.update_wf()
        self._mpl_canvas.draw()

    def handle_z(self, change):
        self.z_offset = change['new']
        self.update_spot()
        self._mpl_canvas.draw()

    def handle_telescope(self, change):
        self.show_telescope = not self.show_telescope
        self.update_telescope()

    def handle_horizon(self, change):
        self.clip_horizon = not self.clip_horizon
        self.update_constellations()
        self.update_stars()

    def handle_rays(self, change):
        self.show_rays = not self.show_rays
        self.update_rays()

    def handle_CCS(self, change):
        self.show_CCS = not self.show_CCS
        self.update_CCS()

    def handle_EDCS(self, change):
        self.show_EDCS = not self.show_EDCS
        self.update_EDCS()
        self._mpl_canvas.draw()

    def handle_DVCS(self, change):
        self.show_DVCS = not self.show_DVCS
        self.update_DVCS()
        self._mpl_canvas.draw()

    def handle_OCS(self, change):
        self.show_OCS = not self.show_OCS
        self.update_OCS()
        self._mpl_canvas.draw()

    def handle_ZCS(self, change):
        self.show_ZCS = not self.show_ZCS
        self.update_ZCS()
        self._mpl_canvas.draw()

    def handle_noll(self, change):
        self.noll = change['new']
        self.update_spot()
        self.update_wf()
        self._mpl_canvas.draw()

    def handle_perturb(self, change):
        self.perturb = change['new']
        self.update_spot()
        self.update_wf()
        self._mpl_canvas.draw()

    def update_CCS(self):
        if self.show_CCS:
            coordSys = self.actual_telescope['LSSTCamera'].coordSys
            for axis, xyz in zip(self.CCS, self._cs_xyz(coordSys, length=2)):
                axis.x = xyz[0]
                axis.y = xyz[1]
                axis.z = xyz[2]
        for axis in self.CCS:
            axis.visible = self.show_CCS

    def update_OCS(self):
        if self.show_OCS:
            coordSys = self.actual_telescope['M1'].coordSys
            for axis, xyz in zip(self.OCS, self._cs_xyz(coordSys, length=2)):
                axis.x = xyz[0]
                axis.y = xyz[1]
                axis.z = xyz[2]
        for axis in self.OCS:
            axis.visible = self.show_OCS
        for item in self.OCS_mpl:
            item.set_visible(self.show_OCS)

    def update_ZCS(self):
        if self.show_ZCS:
            coordSys = self.actual_telescope['M1'].coordSys
            coordSys = coordSys.rotateLocal(batoid.RotY(np.pi))
            for axis, xyz in zip(self.ZCS, self._cs_xyz(coordSys, length=2)):
                axis.x = xyz[0]
                axis.y = xyz[1]
                axis.z = xyz[2]
        for axis in self.ZCS:
            axis.visible = self.show_ZCS
        for item in self.ZCS_mpl:
            item.set_visible(self.show_ZCS)

    def update_EDCS(self):
        if self.show_EDCS:
            det_coordSys = self.actual_telescope['Detector'].coordSys
            # Detector
            for axis, xyz in zip(
                self.EDCS[:3],
                self._cs_xyz(det_coordSys, length=2)
            ):
                axis.x = xyz[0]
                axis.y = xyz[1]
                axis.z = xyz[2]

            # Sky
            sky_coordSys = det_coordSys.rotateLocal(batoid.RotZ(np.pi))
            sky_coordSys.origin = sky_coordSys.zhat * self.sky_dist
            for axis, xyz in zip(
                self.EDCS[3:],
                self._cs_xyz(sky_coordSys, length=np.deg2rad(5.0)*self.sky_dist)
            ):
                axis.x = xyz[0]
                axis.y = xyz[1]
                axis.z = xyz[2]

        for item in self.EDCS_mpl:
            item.set_visible(self.show_EDCS)
        for axis in self.EDCS:
            axis.visible = self.show_EDCS

    def update_DVCS(self):
        if self.show_DVCS:
            det_coordSys = self.actual_telescope['Detector'].coordSys
            sky_coordSys = det_coordSys.rotateLocal(batoid.RotZ(np.pi))
            sky_coordSys.origin = sky_coordSys.zhat * self.sky_dist
            rot = batoid.RotZ(np.pi/2)@batoid.RotX(np.pi)
            det_coordSys = det_coordSys.rotateLocal(rot)
            sky_coordSys = sky_coordSys.rotateLocal(rot)

            for axis, xyz in zip(
                self.DVCS[:3],
                self._cs_xyz(det_coordSys, length=2)
            ):
                axis.x = xyz[0]
                axis.y = xyz[1]
                axis.z = xyz[2]

            for axis, xyz in zip(
                self.DVCS[3:],
                self._cs_xyz(sky_coordSys, length=np.deg2rad(5.0)*self.sky_dist)
            ):
                axis.x = xyz[0]
                axis.y = xyz[1]
                axis.z = xyz[2]

        for item in self.DVCS_mpl:
            item.set_visible(self.show_DVCS)
        for axis in self.DVCS:
            axis.visible = self.show_DVCS

    def update_constellations(self):
        x, y, z = self._constellations_xyz(self.lst)*self.sky_dist
        if self.clip_horizon:
            w = z<0
            x[w] = np.nan
            y[w] = np.nan
            z[w] = np.nan
        self.constellations.x = x
        self.constellations.y = y
        self.constellations.z = z

    def update_stars(self):
        x, y, z, s = self._stars_xyzs(self.lst)
        x *= self.sky_dist
        y *= self.sky_dist
        z *= self.sky_dist
        if self.clip_horizon:
            w = z<0
            x[w] = np.nan
            y[w] = np.nan
            z[w] = np.nan
        self.stars.x = x
        self.stars.y = y
        self.stars.z = z

    def update_telescope(self):
        x, y, z = self._telescope_xyz(self.alt, self.az, self.rtp)
        self.telescope.x = x
        self.telescope.y = y
        self.telescope.z = z
        self.update_rays()
        self.update_fp()
        self.update_EDCS()
        self.update_DVCS()
        self.update_OCS()
        self.update_ZCS()
        self.telescope.visible = self.show_telescope

    def update_elevation_bearings(self):
        x, y, z = self._elevation_bearings_xyz(self.az)
        self.elevation_bearings.x = x
        self.elevation_bearings.y = y
        self.elevation_bearings.z = z

    def update_rays(self):
        if self.show_rays:
            x, y, z = self._rays_xyz(self.thx, self.thy)
            self.rays.x = x
            self.rays.y = y
            self.rays.z = z
        self.rays.visible = self.show_rays

    def update_fp(self):
        x, y, z = self._fp_xyz()
        self.fp.x = x
        self.fp.y = y
        self.fp.z = z

    def update_spot(self):
        # Spot respects z_offset, wf does not.
        perturbed = self.actual_telescope.withLocallyShiftedOptic(
            "Detector",
            (0, 0, self.z_offset*1e-3)
        )
        if np.abs(self.perturb) > 1e-6:
            # Add in phase screen by hand for a moment
            zern = batoid.Zernike(
                np.array([0]*self.noll+[self.perturb*1e-6]),
                R_outer=4.18, R_inner=0.61*4.18
            )
            perturbed = batoid.CompoundOptic(
                (
                    batoid.optic.OPDScreen(
                        batoid.Plane(),
                        zern,
                        name='PS',
                        obscuration=batoid.ObscNegation(batoid.ObscCircle(5.0)),
                        coordSys=perturbed.stopSurface.coordSys
                    ),
                    *perturbed.items
                ),
                name='PS0',
                backDist=perturbed.backDist,
                pupilSize=perturbed.pupilSize,
                inMedium=perturbed.inMedium,
                stopSurface=perturbed.stopSurface,
                sphereRadius=perturbed.sphereRadius,
                pupilObscuration=perturbed.pupilObscuration,
                coordSys=perturbed.coordSys
            )

        rays = batoid.RayVector.asPolar(
            optic=perturbed, wavelength=620e-9,
            nrad=50, naz=300,
            theta_x=np.deg2rad(self.thx),
            theta_y=np.deg2rad(self.thy)
        )
        rv = batoid.intersect(
            self.actual_telescope.stopSurface.surface, rays.copy()
        )
        r_pupil = np.hypot(rv.x, rv.y)
        perturbed.trace(rays)
        r_focal = np.hypot(rays.x, rays.y)
        chief = batoid.RayVector.fromStop(
            optic=perturbed, wavelength=620e-9,
            x=0, y=0,
            theta_x=np.deg2rad(self.thx),
            theta_y=np.deg2rad(self.thy)
        )
        perturbed.trace(chief)
        point = chief.r[0]
        targetCoordSys = rays.coordSys.shiftLocal(point)
        rays = rays.toCoordSys(targetCoordSys)

        visible = ~rays.vignetted
        if not np.any(visible):
            self.spot_scatter.set_offsets(np.array([[],[]]).T)
            return
        x = rays.x * 1e6  # microns
        y = rays.y * 1e6  # microns

        xmax = 1.5*np.quantile(np.abs(np.array([x[visible], y[visible]])), 0.98)

        # Keep the axis limits fixed, rescale x,y instead.
        x /= xmax
        y /= xmax

        r = r_pupil
        # r = r_focal
        r = (r - np.min(r[visible]))/np.ptp(r[visible])

        self.spot_scatter.set_array(r)
        self.spot_scatter.set_alpha(visible.astype(float)*0.5)
        self.spot_scatter.set_offsets(
            np.ma.masked_array(
                np.array([x, y]).T
            )
        )

    def update_wf(self):
        perturbed = self.actual_telescope
        if np.abs(self.perturb) > 1e-6:
            # Add in phase screen by hand for a moment
            zern = batoid.Zernike(
                np.array([0]*self.noll+[self.perturb*1e-6]),
                R_outer=4.18, R_inner=0.61*4.18
            )
            perturbed = batoid.CompoundOptic(
                (
                    batoid.optic.OPDScreen(
                        batoid.Plane(),
                        zern,
                        name='PS',
                        obscuration=batoid.ObscNegation(batoid.ObscCircle(5.0)),
                        coordSys=perturbed.stopSurface.coordSys
                    ),
                    *perturbed.items
                ),
                name='PS0',
                backDist=perturbed.backDist,
                pupilSize=perturbed.pupilSize,
                inMedium=perturbed.inMedium,
                stopSurface=perturbed.stopSurface,
                sphereRadius=perturbed.sphereRadius,
                pupilObscuration=perturbed.pupilObscuration,
                coordSys=perturbed.coordSys
            )

        nx = 256
        wf = batoid.wavefront(
            perturbed,
            np.deg2rad(self.thx),
            np.deg2rad(self.thy),
            620e-9,
            nx=nx
        )
        arr = np.fliplr(sub_ptt(wf.array))  # FLIP LR b/c +x is _left_
        self.wf_imshow.set_array(arr)

    def display(self):
        from IPython.display import display
        ipvfig = ipv.Figure(width=800, height=600)
        ipvfig.camera.far = 100000
        ipvfig.camera.near = 0.01
        ipvfig.camera.position=(2, -0.5, 0.0)
        ipvfig.camera.up=(0, 0, 1)
        ipvfig.camera_center=(0, 0, 0.4)
        ipvfig.style = ipv.styles.dark
        ipvfig.style['box'] = dict(visible=False)
        ipvfig.style['axes']['visible'] = False
        ipvfig.xlim = (-10, 10)
        ipvfig.ylim = (-10, 10)
        ipvfig.zlim = (-10, 10)
        ipvfig.animation = 100
        ipvfig.scatters = self.scatters

        self.app = ipywidgets.HBox([
            self.mpl,
            ipvfig,
            ipywidgets.VBox(self.controls)
        ])
        self.update_stars()
        self.update_constellations()
        self.update_telescope()
        self.update_fp()
        self.update_elevation_bearings()
        self.update_rays()
        self.update_CCS()
        self.update_spot()
        self.update_wf()
        self.update_OCS()
        self.update_ZCS()
        self.update_EDCS()
        self.update_DVCS()

        display(self.app)
