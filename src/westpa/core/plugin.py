class Plugin:
    """Base class for plugins. Plugins can be used to insert extra behavior at
    specific points in the simulation loop.

    Parameters
    ----------
    priority : int, default 0
        Relative priority of the plugin. If two plugins both implement the same
        hook, the plugin with the lower value of `priority` (which is used as a
        sort key) will be run first.

    Attributes
    ----------
    priority : int

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

    def prepare_run(self, simulation):
        """Method to call at the beginning of each run."""
        pass

    def finalize_run(self, simulation):
        """Method to call at the end of each run."""
        pass

    def prepare_iteration(self, simulation):
        """Method to call at the beginning of each iteration."""
        pass

    def finalize_iteration(self, simulation):
        """Method to call at the end of each iteration."""
        pass

    def pre_propagation(self, simulation):
        """Method to call before running dynamics."""
        pass

    def post_propagation(self, simulation):
        """Method to call after running dynamics."""
        pass

    def pre_we(self, simulation):
        """Method to call before weighted ensemble resampling."""
        pass

    def post_we(self, simulation):
        """Method to call after weighted ensemble resampling."""
        pass

    def prepare_new_iteration(self, simulation):
        """Method to call after preparing the next iteration's segments."""
        pass
