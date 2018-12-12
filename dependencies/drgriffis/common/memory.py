'''
Code source: http://code.activestate.com/recipes/286222/
(Alternate: http://stackoverflow.com/questions/938733/total-memory-used-by-python-process)
'''
import os
_proc_status = '/proc/%d/status' % os.getpid()

_B=1.0
_KB=1024.0
_MB=1024.0**2
_GB=1024.0**3
_scale = {
    'B': _B, 'b': _B,
    'kB': _KB, 'KB': _KB, 'kb': _KB,
    'mB': _MB, 'MB': _MB, 'mb': _MB,
    'gB': _GB, 'GB': _GB, 'gb': _GB
}

def _VmB(VmKey):
    '''Private.
    '''
    global _proc_status, _scale
     # get pseudo file  /proc/<pid>/status
    try:
        t = open(_proc_status)
        v = t.read()
        t.close()
    except:
        return 0.0  # non-Linux?
     # get VmKey line e.g. 'VmRSS:  9999  kB\n ...'
    i = v.index(VmKey)
    v = v[i:].split(None, 3)  # whitespace
    if len(v) < 3:
        return 0.0  # invalid format?
     # convert Vm value to bytes
    return float(v[1]) * _scale[v[2]]


def memory(since=0.0, scale='B'):
    '''Return memory usage in bytes.
    '''
    global _scale
    return (_VmB('VmSize:') - since) / _scale[scale]


def resident(since=0.0, scale='B'):
    '''Return resident memory usage in bytes.
    '''
    global _scale
    return (_VmB('VmRSS:') - since) / _scale[scale]


def stacksize(since=0.0, scale='B'):
    '''Return stack size in bytes.
    '''
    global _scale
    return (_VmB('VmStk:') - since) / _scale[scale]
