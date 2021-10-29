import logging
import os; pjoin = os.path.join

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

def init_log(log_level=logging.INFO,path=None):
    if path is None:
        path = pjoin(os.path.expanduser('~'),'.cache')
        if not os.path.exists(path):
            os.makedirs(path)
        path = pjoin(path,'logfile.log')
        
    logging.basicConfig(
        filename=path,
        filemode='w',
        format='LINE %(lineno)-4d: %(asctime)s [%(levelname)-8s] %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p',
        encoding='utf-8',
        level=logging.DEBUG
    )
    
    logger = logging.getLogger(__name__)
    # pass (>= log_level) info to sys.stderr
    console = logging.StreamHandler()
    console.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s %(message)s')
    console.setFormatter(formatter)
    # Finalize
    logger.handlers.clear()
    logger.addHandler(console)
    logger.debug(f'{__name__} logger initiated.')
    logger.info(f'Log is saved in {path}')
    return logger