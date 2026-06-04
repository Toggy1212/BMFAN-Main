from data import srdata

class Set5(srdata.SRData):
    def __init__(self, args, name='Set5', train=False, benchmark=True):
        super(Set5, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )
