import yaml
import traceback


class Map(dict):
    # https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary
    """
    Example:
    m = Map({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    """
    def __init__(self, *args, **kwargs):
        super(Map, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v
                    if isinstance(v, dict):
                        self[k] = Map(v)

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v
                if isinstance(v, dict):
                    self[k] = Map(v)

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Map, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Map, self).__delitem__(key)
        del self.__dict__[key]


"""
m = Map({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
# Add new key
m.new_key = 'Hello world!'
# Or
m['new_key'] = 'Hello world!'
print m.new_key
print m['new_key']
# Update values
m.new_key = 'Yay!'
# Or
m['new_key'] = 'Yay!'
# Delete key
del m.new_key
# Or
del m['new_key']
"""


class ConfigParser(object):
    def __init__(self, config_path):
        self._config_path = config_path

    def parse(self):
        parsed = None
        with open(self._config_path, 'r') as stream:
            try:
                # https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary
                parsed = Map(yaml.safe_load(stream))
            except yaml.YAMLError as exc:
                traceback.print_exc(exc)

        return parsed
