# https://stackoverflow.com/questions/5121931/in-python-how-can-you-load-yaml-mappings-as-ordereddicts

import yaml
from collections import OrderedDict
from past.builtins import basestring
from numbers import Integral

def ordered_load(stream, Loader=yaml.Loader, object_pairs_hook=OrderedDict):
    class OrderedLoader(Loader):
        pass
    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))
    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping)
    return yaml.load(stream, OrderedLoader)

# # usage example:
# ordered_load(stream, yaml.SafeLoader)


def ordered_dump(data, stream=None, Dumper=yaml.Dumper, **kwds):
    class OrderedDumper(Dumper):
        def represent_mapping(self, tag, mapping, flow_style=False):
            return yaml.Dumper.represent_mapping(self, tag, mapping, flow_style)
    def _dict_representer(dumper, data):
        return dumper.represent_mapping(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
            data.items())
    OrderedDumper.add_representer(OrderedDict, _dict_representer)
    return yaml.dump(data, stream, OrderedDumper, **kwds)

# # usage:
# ordered_dump(data, Dumper=yaml.SafeDumper)


class ListDict(OrderedDict):
    # Like an ordered dict, but you can access items by number instead of just by key.  So it's
    # a sequence and a mapping.
    def __init__(self, *args, **kwargs):
        OrderedDict.__init__(self, *args, **kwargs)
        if any(isinstance(k, Integral) for k in self):
            raise ValueError

    def _getKeyFromIndex(self, idx):
        return list(self.keys())[idx]

    def __setitem__(self, key, value):
        # If key is Integral, access that item in order.  Cannot create a new item using Integral
        # key.  If key is not Integral though, can modify or create as needed.
        if isinstance(key, Integral):
            if key >= len(self):
                raise KeyError
            key = self._getKeyFromIndex(key)
        OrderedDict.__setitem__(self, key, value)

    def __getitem__(self, key):
        if isinstance(key, Integral):
            key = self._getKeyFromIndex(key)
        return OrderedDict.__getitem__(self, key)

    def __delitem__(self, key):
        if isinstance(key, Integral):
            key = self._getKeyFromIndex(key)
        OrderedDict.__delitem__(self, key)
