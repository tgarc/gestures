class Processor(object):
    def __init__(self,process,**kwargs):
        self._params.update(kwargs)

        for k,v in self._params.items():
            setattr(self,k,v)

        self.__process = process

    def __call__(self,*args,**kwargs):
        return self.__process(*args,**kwargs)

    @property
    def params(self):
        return {k:getattr(self,k) for k in self._params}
