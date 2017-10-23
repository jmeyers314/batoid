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
