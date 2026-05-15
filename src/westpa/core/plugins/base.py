class Plugin:
    """Base class for plugins. Plugins can be used to insert extra behavior at
    specific points in the simulation loop.

    Parameters
    ----------
    priority : int, default 0
        Relative priority of the plugin. If two plugins both implement the same
        method, the plugin with the lower value of `priority` (which is used as
        a sort key) will be run first.

    Attributes
    ----------
    priority : int

    Methods
    -------
    prepare_run
    finalize_run
    prepare_iteration
    pre_we

    """

    def __init__(self, priority=0):
        self.priority = priority

    @property
    def priority(self):
        """Relative priority of the plugin. Lower value means more priority."""
        return self._priority

    @priority.setter
    def priority(self, value):
        self._priority = int(value)

    def prepare_run(self, sim):
        """Hook called at the beginning of each run."""
        pass

    def finalize_run(self, sim):
        """Hook called at the end of each run."""
        pass

    def prepare_iteration(self, sim):
        """Hook called at the beginning of each iteration."""
        pass

    def pre_we(self):
        """Hook called just before binning and resampling."""
        pass
