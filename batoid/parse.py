import batoid

def parse_obscuration(config):
    typ = config.pop('type')
    if typ in ['ObscCircle', 'ObscRay', 'ObscRectangle']:
        evalstr = "batoid._batoid.{}(**config)".format(typ)
        return eval(evalstr)
    if typ == 'ObscNegation':
        original = parse_obscuration(config['original'])
        return batoid._batoid.ObscNegation(original)
    if typ in ['ObscUnion', 'ObscIntersection']:
        items = [parse_obscuration(c) for c in config['items']]  # noqa
        evalstr = "batoid._batoid.{}(items)".format(typ)
        return eval(evalstr)
    if typ.startswith('Clear'): # triggers negation
        # put type back into config, but with Clear->Obsc
        config['type'] = typ.replace("Clear", "Obsc")
        return batoid._batoid.ObscNegation(parse_obscuration(config))


def parse_surface(config):
    typ = config.pop('type')
    evalstr = "batoid.{}(B=0.0, **config)".format(typ)
    return eval(evalstr)


def parse_coordSys(config, coordSys=batoid._batoid.CoordSys()):
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
        coordSys = coordSys.shiftLocal(batoid.Vec3(shift))
    return coordSys

def parse_optic(config,
                coordSys=batoid._batoid.CoordSys(),
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
    inMedium = config.pop('inMedium', inMedium)
    outMedium = config.pop('outMedium', outMedium)
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
    elif typ == 'Phantom':
        surface = parse_surface(config.pop('surface'))
        return batoid.optic.Phantom(
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
        medium = config.pop('medium')
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
        return batoid.optic.CompoundOptic(items, name=name, coordSys=coordSys)
    else:
        raise ValueError("Unknown optic type")
