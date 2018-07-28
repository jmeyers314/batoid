#https://stackoverflow.com/a/7071358/7530778

__version__ = "0.1.0rc2"
s = __version__.find("rc")
__version_info__ = tuple(map(int, __version__[:s].split('.')))
