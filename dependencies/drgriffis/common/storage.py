class BagOfHolding:
    '''Generic holder class; allows for .<anything> attribute usage
    '''
    pass

class FreqPair:
    '''Container for an arbitrary object and an observed frequency count
    '''
    item = None
    freq = 0

    def __init__(self, item, freq=0):
        self.item = item
        self.freq = freq

    def increment(self):
        '''Increment frequency counter'''
        self.freq += 1
