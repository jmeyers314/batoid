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


def parse_coordSys(config):
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
    coordSys = batoid._batoid.CoordSys()
    if shift != [0.0, 0.0, 0.0]:
        coordSys = coordSys.shiftLocal(batoid.Vec3(shift))
    return coordSys

def parse_optic(config):
    if 'obscuration' in config:
        obscuration = parse_obscuration(config.pop('obscuration'))
    else:
        obscuration = None
    name = config.pop('name', "")
    if 'coordSys' in 'config':
        coordSys = parse_coordSys(config.pop('coordSys'))
    else:
        coordSys = batoid._batoid.CoordSys()

    # Need inMedium, outMedium.  inherited from parent most likely.

    typ = config.pop('type')
    if typ == 'Mirror':
        surface = parse_surface(config.pop('surface'))

        return batoid.optic.Mirror(surface, name=name, coordSys=coordSys, obscuration=obscuration)
