import batoid

import numpy as np


def parse_obscuration(config):
    typ = config.pop('type')
    if typ in [
        'ObscCircle', 'ObscAnnulus', 'ObscRay', 'ObscRectangle', 'ObscPolygon'
    ]:
        evalstr = "batoid.{}(**config)".format(typ)
        return eval(evalstr)
    elif typ == 'ObscNegation':
        original = parse_obscuration(config['original'])
        return batoid.ObscNegation(original)
    elif typ in ['ObscUnion', 'ObscIntersection']:
        items = [parse_obscuration(c) for c in config['items']]  # noqa
        evalstr = "batoid.{}(items)".format(typ)
        return eval(evalstr)
    elif typ.startswith('Clear'): # triggers negation
        # put type back into config, but with Clear->Obsc
        config['type'] = typ.replace("Clear", "Obsc")
        return batoid.ObscNegation(parse_obscuration(config))
    else:
        raise ValueError(f"Unknown obscuration type {typ}")


def parse_surface(config):
    typ = config.pop('type')
    evalstr = "batoid.{}(**config)".format(typ)
    return eval(evalstr)


def parse_coordSys(config, coordSys=batoid.CoordSys()):
    """
    @param config  configuration dictionary
    @param coordSys  sys to which transformations in config are added
    """
    shift = [0.0, 0.0, 0.0]
    if any(x in config for x in ['x', 'y', 'z']):
        if 'shift' in config:
            raise ValueError("Cannot specify both shift and x/y/z")
        x = config.pop('x', 0.0)
        y = config.pop('y', 0.0)
        z = config.pop('z', 0.0)
        shift = [x, y, z]
    elif 'shift' in config:
        shift = config.pop('shift')
    if shift != [0.0, 0.0, 0.0]:
        coordSys = coordSys.shiftLocal(shift)
    # At most one (nonzero) rotation can be included and is applied after the shift.
    rotXYZ = np.array([config.pop('rot' + axis, 0.0) for axis in 'XYZ'])
    axes = np.where(rotXYZ != 0)[0]
    if len(axes) > 1:
        raise ValueError('Cannot specify rotation about more than one axis.')
    elif len(axes) == 1:
        axis, angle = axes[0], rotXYZ[axes[0]]
        rotator = (batoid.RotX, batoid.RotY, batoid.RotZ)[axis](angle)
        coordSys = coordSys.rotateLocal(rotator)
    return coordSys


def parse_optic(config,
                coordSys=batoid.CoordSys(),
                inMedium=batoid.ConstMedium(1.0),
                outMedium=None):
    """
    @param config  configuration dictionary
    @param coordSys  sys to which transformations in config are added
    @param inMedium  default in Medium, often set by optic parent
    @param outMedium default out Medium, often set by optic parent
    """
    if 'obscuration' in config:
        obscuration = parse_obscuration(config.pop('obscuration'))
    else:
        obscuration = None
    name = config.pop('name', "")
    if 'coordSys' in config:
        coordSys = parse_coordSys(config.pop('coordSys'), coordSys)
    inMedium = parse_medium(config.pop('inMedium', inMedium))
    outMedium = parse_medium(config.pop('outMedium', outMedium))
    if outMedium is None:
        outMedium = inMedium

    typ = config.pop('type')
    if typ == 'Mirror':
        surface = parse_surface(config.pop('surface'))
        return batoid.optic.Mirror(
            surface, name=name,
            coordSys=coordSys, obscuration=obscuration,
            inMedium=inMedium, outMedium=outMedium)
    elif typ == 'RefractiveInterface':
        surface = parse_surface(config.pop('surface'))
        return batoid.optic.RefractiveInterface(
            surface, name=name,
            coordSys=coordSys, obscuration=obscuration,
            inMedium=inMedium, outMedium=outMedium)
    elif typ == 'Baffle':
        surface = parse_surface(config.pop('surface'))
        return batoid.optic.Baffle(
            surface, name=name,
            coordSys=coordSys, obscuration=obscuration,
            inMedium=inMedium, outMedium=outMedium)
    elif typ == 'Detector':
        surface = parse_surface(config.pop('surface'))
        return batoid.optic.Detector(
            surface, name=name,
            coordSys=coordSys, obscuration=obscuration,
            inMedium=inMedium, outMedium=outMedium)
    elif typ == 'Lens':
        medium = parse_medium(config.pop('medium'))
        itemsConfig = config.pop('items')
        items = [
            parse_optic(
                itemsConfig[0],
                coordSys=coordSys,
                inMedium=inMedium,
                outMedium=medium
            ),
            parse_optic(
                itemsConfig[1],
                coordSys=coordSys,
                inMedium=medium,
                outMedium=outMedium
            )
        ]
        return batoid.optic.Lens(
            items, name=name, coordSys=coordSys,
            inMedium=inMedium, outMedium=outMedium)
    elif typ == 'CompoundOptic':
        itemsConfig = config.pop('items')
        items = [
            parse_optic(
                iC,
                coordSys=coordSys,
                inMedium=inMedium,
                outMedium=outMedium
            )
            for iC in itemsConfig
        ]
        # Look for a few more possible attributes
        kwargs = {}
        for k in ['backDist', 'sphereRadius', 'pupilSize', 'pupilObscuration']:
            if k in config:
                kwargs[k] = config[k]
        if 'stopSurface' in config:
            kwargs['stopSurface'] = parse_optic(config['stopSurface'])
        return batoid.optic.CompoundOptic(
                items, inMedium=inMedium, outMedium=outMedium,
                name=name, coordSys=coordSys, **kwargs)
    elif typ == 'Interface':
        surface = parse_surface(config.pop('surface'))
        return batoid.optic.Interface(
            surface, name=name,
            coordSys=coordSys
        )
    else:
        raise ValueError("Unknown optic type")


def parse_medium(config):
    from numbers import Real
    if config is None:
        return None
    if isinstance(config, batoid.Medium):
        return config
    if isinstance(config, Real):
        return batoid.ConstMedium(config)
    # This dict may be referenced again in an ancestor config, so copy it
    # before parsing
    config = dict(**config)
    typ = config.pop('type')
    # TableMedium, Sellmeier, ConstMedium, SumitaMedium, Air end up here...
    evalstr = "batoid.{}(**config)".format(typ)
    return eval(evalstr)


def parse_table(config):
    return batoid.Table(config['args'], config['vals'], config['interp'])
