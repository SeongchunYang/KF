class MixedClassMeta(type):
    '''Source: https://stackoverflow.com/a/6100595/9166319'''
    def __new__(cls, name, bases, classdict):
        classinit = classdict.get('__init__')  # Possibly None.

        # Define an __init__ function for the new class.
        def __init__(self, *args, **kwargs):
            '''Note that each 'mixin's are initialized separately.
               Until the very last moment, variables and fns are not
               shared.'''
            # Call the __init__ functions of all the bases.
            for base in type(self).__bases__:
                base.__init__(self, *args, **kwargs)
            # Call any __init__ function that was in the new class.
            if classinit:
                classinit(self, *args, **kwargs)

        # Add the local function to the new class.
        classdict['__init__'] = __init__
        return type.__new__(cls, name, bases, classdict)