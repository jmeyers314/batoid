import ipywidgets
from tables import Description
import batoid
import numpy as np
import ipyvolume as ipv
from functools import lru_cache
import matplotlib.pyplot as plt
import galsim
import contextlib

@lru_cache
def get_constellations_xyz():
    from astropy.io import fits
    try:
        xyz = fits.getdata("../notebook/constellations.fits")
    except:
        from astroquery.simbad import Simbad

        def ten(s):
            ss = s.split()
            h = float(ss[0])
            sign = +1
            if h < 0:
                sign = -1
                h *= -1
            m = float(ss[1])
            s = float(ss[2])
            return sign * (h + m/60 + s/3600)

        Simbad.add_votable_fields('typed_id')
        HIPset = set()
        with open("../notebook/constellationship.fab") as f:
            lines = f.readlines()
        for line in lines:
            HIPset.update([int(s) for s in line.split()[2:]])
        HIPlist = list(HIPset)
        table = Simbad.query_objects(
            [f"HIP {s}" for s in HIPlist]
        )
        table['HIPID'] = HIPlist

        xs = []
        ys = []
        zs = []
        for line in lines:
            xs.append(np.nan)
            ys.append(np.nan)
            zs.append(np.nan)
            prev_second = -1
            endpoints = iter(line.split()[2:])
            for first in endpoints:
                second = next(endpoints)
                first = int(first)
                second = int(second)

                secondrow = table[np.nonzero(table['HIPID'] == int(second))]
                ra1 = np.deg2rad(15*ten(secondrow['RA'][0]))
                dec1 = np.deg2rad(ten(secondrow['DEC'][0]))
                x1 = np.cos(ra1)*np.cos(dec1)
                y1 = np.sin(ra1)*np.cos(dec1)
                z1 = np.sin(dec1)

                if first == prev_second:
                    # just append new second
                    xs.append(x1)
                    ys.append(y1)
                    zs.append(z1)
                else:
                    firstrow = table[np.nonzero(table['HIPID'] == int(first))]
                    ra0 = np.deg2rad(15*ten(firstrow['RA'][0]))
                    dec0 = np.deg2rad(ten(firstrow['DEC'][0]))
                    x0 = np.cos(ra0)*np.cos(dec0)
                    y0 = np.sin(ra0)*np.cos(dec0)
                    z0 = np.sin(dec0)
                    xs.extend([np.nan, x0, x1])
                    ys.extend([np.nan, y0, y1])
                    zs.extend([np.nan, z0, z1])
                prev_second = second
        xyz = np.array([xs, ys, zs])
        fits.writeto("constellations.fits", xyz)
    return xyz

@lru_cache
def get_stars_xyzs():
    #http://tdc-www.harvard.edu/catalogs/bsc5.dat.gz
    from astropy.io import fits
    table = fits.getdata("../notebook/BSC5.fits")
    table = table[table['mag'] < 5.5]
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

class RubinCSApp:
    def __init__(self, debug=None):
        self.sky_dist = 15000
        self.lat = -30.2446
        self.fiducial_telescope = batoid.Optic.fromYaml("LSST_r.yaml")
        self.actual_telescope = self.fiducial_telescope # for now
        if debug is None:
            debug = contextlib.redirect_stdout(None)
        self.debug = debug

        # widget variables
        self.clip_horizon = False
        self.show_telescope = True
        self.lst = 0.0
        self.rtp = 0.0
        self.alt = 30.0
        self.az = 45.0
        self.thx = 0.0
        self.thy = 0.0
        self.z_offset = 0.0
        self.show_rays = True
        self.show_CCS = False
        self.show_OCS = False
        self.show_ZCS = False
        self.show_DVCS = False
        self.show_EDCS = True

        # IPV Scatters
        self.constellations = self._constellations_view()
        self.stars = self._stars_view()
        self.telescope = self._telescope_view()
        self.fp = self._fp_view()
        self.azimuth_ring = self._azimuth_ring_view()
        self.elevation_bearings = self._elevation_bearings_view()
        self.rays = self._rays_view()
        self.CCS = self._ccs_views()
        self.EDCS = self._edcs_views()
        self.DVCS = self._dvcs_views()

        # Matplotlib
        self.mpl = self._mpl_view()

        # Controls
        self.alt_control = ipywidgets.FloatText(value=self.alt, step=3.0, description='alt (deg)')
        self.az_control = ipywidgets.FloatText(value=self.az, step=3.0, description='az (deg)')
        self.rtp_control = ipywidgets.FloatText(value=0.0, step=3.0, description='RTP (deg)')
        self.thx_control = ipywidgets.FloatText(value=0.0, step=0.25, description='Field x (deg)')
        self.thy_control = ipywidgets.FloatText(value=0.0, step=0.25, description='Field y (deg)')
        self.lst_control = ipywidgets.FloatText(value=0.0, step=0.01, description='LST (hr)')
        self.z_control = ipywidgets.FloatText(value=0.0, step=0.1, description="Det z (mm)")
        self.telescope_control = ipywidgets.Checkbox(value=self.show_telescope, description="show telescope")
        self.horizon_control = ipywidgets.Checkbox(value=self.clip_horizon, description='horizon')
        self.rays_control = ipywidgets.Checkbox(value=self.show_rays, description="Show rays")
        self.CCS_control = ipywidgets.Checkbox(value=self.show_CCS, description='CCS')
        self.EDCS_control = ipywidgets.Checkbox(value=self.show_EDCS, description='EDCS')
        self.DVCS_control = ipywidgets.Checkbox(value=self.show_DVCS, description='DVCS')

        # observe
        self.alt_control.observe(self.handle_alt, 'value')
        self.az_control.observe(self.handle_az, 'value')
        self.rtp_control.observe(self.handle_rtp, 'value')
        self.thx_control.observe(self.handle_thx, 'value')
        self.thy_control.observe(self.handle_thy, 'value')
        self.lst_control.observe(self.handle_lst, 'value')
        self.z_control.observe(self.handle_z, 'value')
        self.telescope_control.observe(self.handle_telescope, 'value')
        self.horizon_control.observe(self.handle_horizon, 'value')
        self.rays_control.observe(self.handle_rays, 'value')
        self.CCS_control.observe(self.handle_CCS, 'value')
        self.EDCS_control.observe(self.handle_EDCS, 'value')
        self.DVCS_control.observe(self.handle_DVCS, 'value')

        self.update_constellations()
        self.update_telescope()
        self.update_fp()
        self.update_elevation_bearings()
        self.update_rays()
        self.update_CCS()
        self.update_spot()
        self.update_wf()
        self.update_EDCS()
        self.update_DVCS()

        self.scatters = [
            self.constellations,
            self.stars,
            self.telescope,
            self.fp,
            self.azimuth_ring,
            self.elevation_bearings,
            self.rays,
            *self.CCS,
            *self.EDCS,
            *self.DVCS
        ]

        self.controls = [
            self.alt_control,
            self.az_control,
            self.rtp_control,
            self.thx_control,
            self.thy_control,
            self.lst_control,
            self.z_control,
            self.horizon_control,
            self.telescope_control,
            self.rays_control,
            self.CCS_control,
            self.EDCS_control,
            self.DVCS_control
        ]

    def _constellations_xyz(self, lst):
        ctf = batoid.CoordTransform(
            batoid.globalCoordSys,
            batoid.CoordSys(
                (0, 0, 0),
                batoid.RotZ(np.deg2rad(lst*15)) @ batoid.RotY(-np.deg2rad(90-self.lat))
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
                batoid.RotZ(np.deg2rad(lst*15)) @ batoid.RotY(-np.deg2rad(90-self.lat))
            )
        )
        x, y, z, s = get_stars_xyzs()
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
        telescope = self.fiducial_telescope
        telescope = telescope.withGlobalShift([0, 0, 3.53])  # Height of M1 vertex above azimuth ring
        telescope = telescope.withLocallyRotatedOptic("LSSTCamera", batoid.RotZ(np.deg2rad(rtp)))
        telescope = telescope.withLocalRotation(batoid.RotZ(np.deg2rad(90-az)))
        telescope = telescope.withLocalRotation(batoid.RotX(np.deg2rad(90-alt)), rotOrigin=[0, 0, 5.425], coordSys=batoid.globalCoordSys)
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
        fig, axes = plt.subplots(
            nrows=3, ncols=1,
            figsize=(2, 6.2), dpi=100,
            facecolor='k'
        )
        fig.canvas.header_visible=False

        # Spot diagram
        self.spot_ax = axes[0]
        self.spot_scatter = self.spot_ax.scatter(
            [], [], s=0.1, c=[], cmap='plasma',
            vmin=2.55, vmax=4.18
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
            cmap='seismic'
        )

        for ax in axes:
            ax.set_aspect('equal')
            ax.set_facecolor('k')

        out = ipywidgets.Output()
        with out:
            plt.show(fig)
        return out

    def handle_alt(self, change):
        self.alt = change['new']
        self.update_telescope()

    def handle_az(self, change):
        self.az = change['new']
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

    def handle_lst(self, change):
        self.lst = change['new']
        self.update_constellations()
        self.update_stars()

    def handle_thx(self, change):
        self.thx = change['new']
        self.update_rays()
        self.update_spot()
        self.update_wf()

    def handle_thy(self, change):
        self.thy = change['new']
        self.update_rays()
        self.update_spot()
        self.update_wf()

    def handle_z(self, change):
        self.z_offset = change['new']
        self.update_spot()

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

    def handle_DVCS(self, change):
        self.show_DVCS = not self.show_DVCS
        self.update_DVCS()

    def update_CCS(self):
        if self.show_CCS:
            coordSys = self.actual_telescope['LSSTCamera'].coordSys
            for axis, xyz in zip(self.CCS, self._cs_xyz(coordSys, length=2)):
                axis.x = xyz[0]
                axis.y = xyz[1]
                axis.z = xyz[2]
        for axis in self.CCS:
            axis.visible = self.show_CCS

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
        perturbed = self.actual_telescope.withLocallyShiftedOptic(
            "Detector",
            (0, 0, self.z_offset*1e-3)
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
        r = np.hypot(rv.x, rv.y)
        perturbed.trace(rays)
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
        x = rays.x * 1e6  # microns
        y = rays.y * 1e6  # microns

        xmax = 1.5*np.quantile(np.abs(np.array([x[visible], y[visible]])), 0.98)

        # Keep the axis limits fixed, rescale x,y instead.
        x /= xmax
        y /= xmax

        self.spot_scatter.set_array(r)
        self.spot_scatter.set_alpha(visible.astype(float)*0.5)
        self.spot_scatter.set_offsets(np.array([x, y]).T)

    def update_wf(self):
        nx = 256
        wf = batoid.wavefront(
            self.fiducial_telescope,
            np.deg2rad(self.thx),
            np.deg2rad(self.thy),
            620e-9,
            nx=nx
        )
        arr = sub_ptt(wf.array)
        self.wf_imshow.set_array(arr)

    def display(self):
        from IPython.display import display
        ipvfig = ipv.Figure(width=800, height=600)
        ipvfig.camera.far = 100000
        ipvfig.camera.near = 0.01
        ipvfig.style = ipv.styles.dark
        ipvfig.style['box'] = dict(visible=False)
        ipvfig.style['axes']['visible'] = False
        ipvfig.xlim = (-10, 10)
        ipvfig.ylim = (-10, 10)
        ipvfig.zlim = (-10, 10)
        ipvfig.animation = 100
        ipvfig.scatters = self.scatters

        display(ipywidgets.HBox([
            self.mpl,
            ipvfig,
            ipywidgets.VBox(self.controls)
        ]))