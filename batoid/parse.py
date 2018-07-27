import batoid


def parse_obscuration(config):
    typ = config.pop('type')
    if typ in ['ObscCircle', 'ObscAnnulus', 'ObscRay', 'ObscRectangle']:
        evalstr = "batoid.{}(**config)".format(typ)
        return eval(evalstr)
    if typ == 'ObscNegation':
        original = parse_obscuration(config['original'])
        return batoid.ObscNegation(original)
    if typ in ['ObscUnion', 'ObscIntersection']:
        items = [parse_obscuration(c) for c in config['items']]  # noqa
        evalstr = "batoid.{}(items)".format(typ)
        return eval(evalstr)
    if typ.startswith('Clear'): # triggers negation
        # put type back into config, but with Clear->Obsc
        config['type'] = typ.replace("Clear", "Obsc")
        return batoid.ObscNegation(parse_obscuration(config))


def parse_surface(config):
    typ = config.pop('type')
    evalstr = "batoid.{}(**config)".format(typ)
    return eval(evalstr)


def parse_coordSys(config, coordSys=batoid.CoordSys()):
    """
    @param config  configuration dictionary
    @param coordSys  sys to which transformations in config are added
    """
    if any(x in config for x in ['x', 'y', 'z']):
        if 'shift' in config:
            raise ValueError("Cannot specify both shift and x/y/z")
        x = config.pop('x', 0.0)
        y = config.pop('y', 0.0)
        z = config.pop('z', 0.0)
        shift = [x, y, z]
    elif 'shift' in config:
        shift = config.pop('shift')
    # Leaving rotation out for the moment...
    if shift != [0.0, 0.0, 0.0]:
        coordSys = coordSys.shiftLocal(shift)
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
            parse_optic(itemsConfig[0], coordSys=coordSys, inMedium=inMedium, outMedium=medium),
            parse_optic(itemsConfig[1], coordSys=coordSys, inMedium=medium, outMedium=outMedium)
        ]
        return batoid.optic.Lens(
            items, medium, name=name, coordSys=coordSys,
            inMedium=inMedium, outMedium=outMedium)
    elif typ == 'CompoundOptic':
        itemsConfig = config.pop('items')
        items = [
            parse_optic(iC, coordSys=coordSys, inMedium=inMedium, outMedium=outMedium)
            for iC in itemsConfig
        ]
        # Look for a few more possible attributes
        kwargs = {}
        for k in ['dist', 'sphereRadius', 'pupilSize', 'pupilObscuration']:
            if k in config:
                kwargs[k] = config[k]
        return batoid.optic.CompoundOptic(
                items, inMedium=inMedium, outMedium=outMedium,
                name=name, coordSys=coordSys, **kwargs)
    else:
        raise ValueError("Unknown optic type")


def parse_medium(value):
    from numbers import Real
    if isinstance(value, batoid.Medium):
        return value
    elif isinstance(value, Real):
        return batoid.ConstMedium(value)
    elif isinstance(value, str):
        if value == 'air':
            return batoid.Air()
        elif value == 'silica':
            return batoid.SellmeierMedium(
                0.6961663, 0.4079426, 0.8974794,
                0.0684043**2, 0.1162414**2, 9.896161**2)
        elif value == 'hsc_air':
            return batoid.ConstMedium(1.0)
        w = [0.4, 0.6, 0.75, 0.9, 1.1]
        w = [w_*1e-6 for w_ in w]
        if value == 'hsc_silica':
            return batoid.TableMedium(
                batoid.Table(
                    w,
                    [1.47009272, 1.45801158, 1.45421013, 1.45172729, 1.44917721],
                    batoid.Table.Interpolant.linear
                )
            )
        elif value == 'hsc_bsl7y':
            return batoid.TableMedium(
                batoid.Table(
                    w,
                    [1.53123287, 1.51671428, 1.51225242, 1.50939738, 1.50653251],
                    batoid.Table.Interpolant.linear
                )
            )
        elif value == 'hsc_pbl1y':
            return batoid.TableMedium(
                batoid.Table(
                    w,
                    [1.57046066, 1.54784671, 1.54157395, 1.53789058, 1.53457169],
                    batoid.Table.Interpolant.linear
                )
            )
        elif value == 'NLAK10':
            return batoid.SellmeierMedium(
                1.72878017, 0.169257825, 1.19386956,
                0.00886014635, 0.0363416509, 82.9009069
            )
        elif value == 'PFK85':
            return batoid.SumitaMedium(
                2.1858326, -0.0050155632, 0.0075107775,
                0.00017770562, -1.2164148e-05, 6.1341005e-07
            )
        elif value == 'BK7':
            return batoid.SellmeierMedium(
                1.03961212, 0.231792344, 1.01046945,
                0.00600069867, 0.0200179144, 103.560653
            )
        else:
            raise RuntimeError("Unknown medium {}".format(value))
