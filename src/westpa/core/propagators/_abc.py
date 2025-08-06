import abc


class Propagator(abc.ABC):

    @abc.abstractmethod
    def __call__(self, segment):
        return segment
