import os
import re
import stat

import pytest

import westpa
from westpa.core.propagators._amber import AmberPropagator
from westpa.core.sim_manager import PropagationError


@pytest.fixture
def prmtop_file(tmp_path):
    return tmp_path / 'system.prmtop'


@pytest.fixture
def initial_state(tmp_path):
    return westpa.State(file=tmp_path / 'initial.rst')


def make_segment(initial_state, n_iter, seg_id, parent_id):
    return westpa.Segment(n_iter=n_iter, seg_id=seg_id, parent_id=parent_id, initial_state=initial_state)


@pytest.fixture
def newtraj_segment(initial_state):
    return make_segment(initial_state, n_iter=1, seg_id=0, parent_id=-1)


@pytest.fixture
def continues_segment(initial_state):
    return make_segment(initial_state, n_iter=2, seg_id=0, parent_id=0)


def get_ig(command):
    match = re.search(r'ig\s*=\s*(\d+)', command)
    assert match is not None, 'expected an ig= entry in the mdin block'
    return int(match.group(1))


@pytest.fixture
def propagator(prmtop_file, tmp_path):
    return AmberPropagator(
        md_parameters={'nstlim': 100, 'dt': 0.002},
        prmtop_file=str(prmtop_file),
        engine='sander',
        root_seed=12345,
        segment_dir_template=os.path.join(tmp_path, '{n_iter:06d}', '{seg_id:06d}'),
    )


# ---- defaults ----


def test_default_final_state_filename(propagator):
    assert propagator.final_state_filename == 'restrt'


# When enginge is not a valid option
def test_invalid_engine_raises(propagator, newtraj_segment):
    propagator.engine = 'not_a_real_engine'
    with pytest.raises(PropagationError, match='not a valid'):
        propagator.get_command(newtraj_segment, rng=None)


# When enginge is not found on Path
def test_engine_not_on_path_raises(propagator, newtraj_segment, monkeypatch):
    monkeypatch.setattr('shutil.which', lambda engine: None)
    with pytest.raises(PropagationError, match='not found in PATH'):
        propagator.get_command(newtraj_segment, rng=None)


# Ensures engine names are always converted to lower case
def test_engine_name_is_lowercased(prmtop_file):
    propagator = AmberPropagator(md_parameters={}, prmtop_file=str(prmtop_file), engine='SANDER')
    assert propagator.engine == 'sander'


# checks that given a correctly constructred path from shutil.which the command correctly includes the enginge name in the command
@pytest.mark.parametrize('engine', ['sander', 'pmemd', 'pmemd.cuda'])
def test_valid_engines_accepted(prmtop_file, initial_state, engine, monkeypatch):
    monkeypatch.setattr('shutil.which', lambda engine: f'/usr/bin/{engine}')
    propagator = AmberPropagator(md_parameters={}, prmtop_file=str(prmtop_file), engine=engine, root_seed=1)
    segment = make_segment(initial_state, n_iter=1, seg_id=0, parent_id=-1)
    rng = propagator._get_rng(segment)
    command = propagator.get_command(segment, rng)
    assert engine in command


# checks that given a correctly constructred path from shutil.which the command correctly includes the -p and -c flags
def test_command_contains_topology_and_initial_state(propagator, newtraj_segment, prmtop_file, monkeypatch):
    monkeypatch.setattr('shutil.which', lambda engine: '/usr/bin/sander')
    rng = propagator._get_rng(newtraj_segment)
    command = propagator.get_command(newtraj_segment, rng)
    assert f'-p {prmtop_file}' in command
    assert f'-c {newtraj_segment.initial_state.file}' in command


# check that the header of the mdin file contains the correct title
def test_title_reflects_segment(propagator, newtraj_segment, monkeypatch):
    monkeypatch.setattr('shutil.which', lambda engine: '/usr/bin/sander')
    rng = propagator._get_rng(newtraj_segment)
    command = propagator.get_command(newtraj_segment, rng)
    assert f'n_iter={newtraj_segment.n_iter}, seg_id={newtraj_segment.seg_id}' in command


# ---- mdin / restart logic ----


# ensures the command contains the correct irest and ntx flags for a new segment
def test_newtraj_segment_sets_fresh_start_flags(propagator, newtraj_segment, monkeypatch):
    monkeypatch.setattr('shutil.which', lambda engine: '/usr/bin/sander')
    rng = propagator._get_rng(newtraj_segment)
    command = propagator.get_command(newtraj_segment, rng)
    assert 'irest = 0' in command
    assert 'ntx = 1' in command


# ensures the command contains the correct irest and ntx flags for a continuing a segment
def test_continues_segment_sets_restart_flags(propagator, continues_segment, monkeypatch):
    monkeypatch.setattr('shutil.which', lambda engine: '/usr/bin/sander')
    rng = propagator._get_rng(continues_segment)
    command = propagator.get_command(continues_segment, rng)
    assert 'irest = 1' in command
    assert 'ntx = 5' in command


def test_seed_is_reproducible_for_same_segment(propagator, newtraj_segment, monkeypatch):
    monkeypatch.setattr('shutil.which', lambda engine: '/usr/bin/sander')
    ig_1 = get_ig(propagator.get_command(newtraj_segment, propagator._get_rng(newtraj_segment)))
    ig_2 = get_ig(propagator.get_command(newtraj_segment, propagator._get_rng(newtraj_segment)))
    assert ig_1 == ig_2


# ensures writing the command does not overwrite the input variables in memory, only a copy
def test_md_parameters_not_mutated(propagator, newtraj_segment, monkeypatch):
    monkeypatch.setattr('shutil.which', lambda engine: '/usr/bin/sander')
    original = propagator.md_parameters.copy()
    propagator.get_command(newtraj_segment, propagator._get_rng(newtraj_segment))
    assert propagator.md_parameters == original


@pytest.fixture
def fake_sander(tmp_path, monkeypatch):
    """Stubs out `sander` on PATH with a script that just writes a restrt file."""
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    script = bin_dir / 'sander'
    script.write_text('#!/bin/sh\necho fake_restart > restrt\n')
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv('PATH', f'{bin_dir}{os.pathsep}{os.environ["PATH"]}')


def test_call_produces_final_state(propagator, newtraj_segment, fake_sander):
    segments = propagator([newtraj_segment])
    segment = segments[0]
    assert segment.status == westpa.Segment.Status.COMPLETE
    assert os.path.exists(segment.final_state.file)
    assert segment.cputime > 0
