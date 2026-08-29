import abc
import os
import signal
import subprocess

from .base import SerialPropagator
from ..state import State


class SubprocessPropagator(SerialPropagator):
    """Propagates a trajectory segment by running an external program.
    The program is executed in the directory created by the
    :meth:`make_segment_dir` method.

    Subclasses must implement the :meth:`get_command` method.

    Parameters
    ----------
    final_state_filename : str
        Name of the file used to store the final state of each segment (e.g.,
        ``'final_state.rst'``).
    stdout : str or io.TextIOBase, optional
        File or stream to redirect stdout to. If a string is passed, it must
        be a path template containing ``{n_iter}`` and ``{seg_id}`` replacement
        fields. If None, no redirection will occur.
    stderr : str or io.TextIOBase, optional
        File or stream to redirect stderr to. If a string is passed, it must
        be a path template containing ``{n_iter}`` and ``{seg_id}`` replacement
        fields. If None, no redirection will occur.
    shell : bool, default True
        If True, the string returned by the :meth:`get_command` method will be
        executed through the shell. If False, the string must be the name of a
        program to execute.
    **kwargs
        Parameters to pass to the :class:`Propagator` base class constructor.

    """

    def __init__(
        self,
        final_state_filename,
        stdout=None,
        stderr=None,
        shell=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.final_state_filename = final_state_filename
        self.stdout = os.path.abspath(stdout) if isinstance(stdout, str) else stdout
        self.stderr = os.path.abspath(stderr) if isinstance(stderr, str) else stderr
        self.shell = shell

    @abc.abstractmethod
    def get_command(self, segment, rng):
        """ "Return the command to execute for a given segment.

        Parameters
        ----------
        segment : :class:`Segment`
            Segment to propagate.
        rng : numpy.random.Generator
            Pseudo-random number generator for seeding the stochastic dynamics
            engine.

        Returns
        -------
        command : str
            Command to execute for the given segment. It must write the final
            state of the segment to ``self.final_state_filename``.

        """
        ...

    def propagate(self, segment, rng):
        segment_dir = self.make_segment_dir(segment)
        command = self.get_command(segment, rng)

        if isinstance(self.stdout, str):
            stdout = self.stdout.format(n_iter=segment.n_iter, seg_id=segment.seg_id)
        else:
            stdout = self.stdout
        if isinstance(self.stderr, str):
            stderr = self.stderr.format(n_iter=segment.n_iter, seg_id=segment.seg_id)
        else:
            stderr = self.stderr

        process = subprocess.Popen(
            command,
            shell=self.shell,
            stdout=stdout,
            stderr=stderr,
            cwd=segment_dir,
            close_fds=True,
        )

        options = 0
        pid, status, rusage = os.wait4(process.pid, options)

        if (rc := process.wait()) == 0:
            final_state_file = os.path.join(segment_dir, self.final_state_filename)
            if os.path.exists(final_state_file):
                segment.final_state = State(file=final_state_file)
            else:
                raise RuntimeError(f"final state must be written to {final_state_file}")
        elif rc < 0:
            signum = -rc
            description = signal.strsignal(signum)
            reason = 'child process for segment %d exited on signal %d (%s)' % (segment.seg_id, signum, description)
            segment.mark_as_failed(reason)
        else:
            reason = 'child process for segment %d exited with code %d' % (segment.seg_id, rc)
            segment.mark_as_failed(reason)

        segment.cputime = rusage.ru_utime

        return segment
