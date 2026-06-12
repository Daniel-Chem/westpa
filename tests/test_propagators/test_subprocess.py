import os

import pytest

import westpa
from westpa.core.propagators.subprocess import SubprocessPropagator


class TestPropagator(SubprocessPropagator):
    """Creates a symlink from the final state file to the initial state file."""

    def get_command(self, segment, rng):
        return f'ln -s {segment.initial_state.file} {self.final_state_filename}'


@pytest.fixture
def propagator(tmp_path):
    return TestPropagator(
        final_state_filename='final_state.npy',
        segment_dir_template=os.path.join(tmp_path, '{n_iter:06d}', '{seg_id:06d}'),
    )


@pytest.fixture
def initial_state(tmp_path):
    file = tmp_path / "initial_state.npy"
    return westpa.State(file=file)


@pytest.fixture
def segment(initial_state):
    return westpa.Segment(n_iter=1, seg_id=0, initial_state=initial_state)


def test_call(propagator, segment):
    segments = propagator([segment])
    segment = segments[0]
    assert os.readlink(segment.final_state.file) == segment.initial_state.file
    assert segment.cputime > 0
