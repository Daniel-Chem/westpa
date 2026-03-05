"""Tests for the new Simulation API (westpa.core.simulation)."""

import pytest
import numpy as np

from westpa.core.simulation import Simulation
from westpa.core.state import State
from westpa.core.segment import Segment
from westpa.core.propagators import Propagator
from westpa.core.binning import NopMapper, RectilinearBinMapper
from westpa.core.resamplers import HuberKimResampler
from westpa.core.source_sink import Source, Sink
from westpa.core.plugin import Plugin
from westpa.work_managers import SerialWorkManager

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class TrivialPropagator(Propagator):
    """Moves a 1-D coordinate by a fixed delta each step."""

    def __init__(self, delta=0.1, **kwargs):
        super().__init__(**kwargs)
        self.delta = delta

    def propagate(self, segment):
        coord = segment.initial_state.coord
        segment.final_state = State(coord=coord + self.delta)
        return segment


class FailingPropagator(Propagator):
    """Always marks segments as failed."""

    def propagate(self, segment):
        segment.mark_as_failed("deliberate test failure")
        return segment


def trivial_pcoord_calculator(segment):
    """Sets pcoord to the final x-coordinate as a (1, 1)-shaped array."""
    x = segment.final_state.coord[0]
    segment.pcoord = np.array([[x]])
    return segment


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def datafile(tmp_path):
    return str(tmp_path / "west.h5")


@pytest.fixture
def propagator():
    return TrivialPropagator(seed=42)


@pytest.fixture
def sim(datafile, propagator):
    return Simulation(
        datafile=datafile,
        propagator=propagator,
        pcoord_calculator=trivial_pcoord_calculator,
    )


# ---------------------------------------------------------------------------
# Constructor tests
# ---------------------------------------------------------------------------


class TestSimulationConstructor:
    def test_basic_construction(self, datafile, propagator):
        sim = Simulation(
            datafile=datafile,
            propagator=propagator,
            pcoord_calculator=trivial_pcoord_calculator,
        )
        assert sim.datafile == datafile
        assert sim.propagator is propagator
        assert sim.pcoord_calculator is trivial_pcoord_calculator

    def test_default_bin_mapper_is_nop(self, sim):
        assert isinstance(sim.bin_mapper, NopMapper)

    def test_default_bin_target_counts(self, sim):
        np.testing.assert_array_equal(sim.bin_target_counts, [1])

    def test_default_resampler_is_huber_kim(self, sim):
        assert isinstance(sim.resampler, HuberKimResampler)

    def test_default_work_manager_is_serial(self, sim):
        assert isinstance(sim.work_manager, SerialWorkManager)

    def test_default_source_is_none(self, sim):
        assert sim.source is None

    def test_default_sinks_is_empty(self, sim):
        assert len(sim.sinks) == 0

    def test_default_istate_generator_is_none(self, sim):
        assert sim.istate_generator is None

    def test_default_propagator_block_size(self, sim):
        assert sim.propagator_block_size == 1

    def test_invalid_propagator_type(self, datafile):
        with pytest.raises(TypeError, match="'propagator' must be a Propagator"):
            Simulation(
                datafile=datafile,
                propagator="not_a_propagator",
                pcoord_calculator=trivial_pcoord_calculator,
            )

    def test_invalid_pcoord_calculator_not_callable(self, datafile, propagator):
        with pytest.raises(TypeError, match="'pcoord_calculator' must be callable"):
            Simulation(
                datafile=datafile,
                propagator=propagator,
                pcoord_calculator="not_callable",
            )

    def test_invalid_resampler_type(self, datafile, propagator):
        with pytest.raises(TypeError, match="'resampler' must be a Resampler"):
            Simulation(
                datafile=datafile,
                propagator=propagator,
                pcoord_calculator=trivial_pcoord_calculator,
                resampler="not_a_resampler",
            )

    def test_invalid_work_manager_type(self, datafile, propagator):
        with pytest.raises(TypeError, match="'work_manager' must be a WorkManager"):
            Simulation(
                datafile=datafile,
                propagator=propagator,
                pcoord_calculator=trivial_pcoord_calculator,
                work_manager="not_a_work_manager",
            )

    def test_invalid_propagator_block_size_type(self, datafile, propagator):
        with pytest.raises(TypeError, match="'propagator_block_size' must be an integer"):
            Simulation(
                datafile=datafile,
                propagator=propagator,
                pcoord_calculator=trivial_pcoord_calculator,
                propagator_block_size=1.5,
            )

    def test_invalid_propagator_block_size_value(self, datafile, propagator):
        with pytest.raises(ValueError, match="'propagator_block_size' must be at least 1"):
            Simulation(
                datafile=datafile,
                propagator=propagator,
                pcoord_calculator=trivial_pcoord_calculator,
                propagator_block_size=0,
            )

    def test_source_without_sink_raises(self, datafile, propagator):
        source = Source(State(coord=[0.0]))
        with pytest.raises(ValueError, match="'source' and 'sinks' must be provided together"):
            Simulation(
                datafile=datafile,
                propagator=propagator,
                pcoord_calculator=trivial_pcoord_calculator,
                source=source,
            )

    def test_sink_without_source_raises(self, datafile, propagator):
        sink = Sink(lambda seg: seg.pcoord[-1, 0] > 1.0)
        with pytest.raises(ValueError, match="'source' and 'sinks' must be provided together"):
            Simulation(
                datafile=datafile,
                propagator=propagator,
                pcoord_calculator=trivial_pcoord_calculator,
                sinks=[sink],
            )

    def test_source_and_sink_together(self, datafile, propagator):
        source = Source(State(coord=[0.0]))
        sink = Sink(lambda seg: seg.pcoord[-1, 0] > 1.0)
        sim = Simulation(
            datafile=datafile,
            propagator=propagator,
            pcoord_calculator=trivial_pcoord_calculator,
            source=source,
            sinks=[sink],
        )
        assert sim.source is source
        assert sim.sinks[0] is sink

    def test_invalid_istate_generator_type(self, datafile, propagator):
        with pytest.raises(TypeError, match="'istate_generator' must be callable"):
            Simulation(
                datafile=datafile,
                propagator=propagator,
                pcoord_calculator=trivial_pcoord_calculator,
                istate_generator="not_callable",
            )

    def test_plugins_added_in_priority_order(self, datafile, propagator):
        p_low = Plugin(priority=0)
        p_mid = Plugin(priority=5)
        p_high = Plugin(priority=10)
        sim = Simulation(
            datafile=datafile,
            propagator=propagator,
            pcoord_calculator=trivial_pcoord_calculator,
            plugins=[p_high, p_low, p_mid],
        )
        assert list(sim.plugins) == [p_low, p_mid, p_high]

    def test_custom_bin_mapper_and_target_counts(self, datafile, propagator):
        mapper = RectilinearBinMapper([[-np.inf, 0.5, np.inf]])  # 2 bins
        sim = Simulation(
            datafile=datafile,
            propagator=propagator,
            pcoord_calculator=trivial_pcoord_calculator,
            bin_mapper=mapper,
            bin_target_counts=3,
        )
        assert sim.bin_mapper is mapper
        np.testing.assert_array_equal(sim.bin_target_counts, [3, 3])


# ---------------------------------------------------------------------------
# update_bins tests
# ---------------------------------------------------------------------------


class TestUpdateBins:
    def test_integer_target_counts_broadcast(self, sim):
        mapper = RectilinearBinMapper([[-np.inf, 0.5, np.inf]])  # 2 bins
        sim.update_bins(mapper, target_counts=5)
        np.testing.assert_array_equal(sim.bin_target_counts, [5, 5])

    def test_sequence_target_counts(self, sim):
        mapper = RectilinearBinMapper([[-np.inf, 0.5, np.inf]])  # 2 bins
        sim.update_bins(mapper, target_counts=[3, 7])
        np.testing.assert_array_equal(sim.bin_target_counts, [3, 7])
        assert sim.bin_mapper is mapper

    def test_invalid_mapper_type(self, sim):
        with pytest.raises(TypeError, match="'mapper' must be a BinMapper"):
            sim.update_bins("not_a_mapper", target_counts=1)

    def test_target_counts_length_mismatch(self, sim):
        mapper = RectilinearBinMapper([[-np.inf, 0.5, np.inf]])  # 2 bins
        with pytest.raises(ValueError, match="length of 'target_counts' must equal"):
            sim.update_bins(mapper, target_counts=[1, 2, 3])

    def test_target_counts_zero_raises(self, sim):
        mapper = RectilinearBinMapper([[-np.inf, 0.5, np.inf]])  # 2 bins
        with pytest.raises(ValueError, match="'target_counts' must be positive"):
            sim.update_bins(mapper, target_counts=[0, 1])

    def test_target_counts_negative_raises(self, sim):
        mapper = RectilinearBinMapper([[-np.inf, 0.5, np.inf]])  # 2 bins
        with pytest.raises(ValueError, match="'target_counts' must be positive"):
            sim.update_bins(mapper, target_counts=[-1, 1])


# ---------------------------------------------------------------------------
# update_source_and_sinks tests
# ---------------------------------------------------------------------------


class TestUpdateSourceAndSinks:
    def test_valid_update(self, sim):
        source = Source(State(coord=[0.0]))
        sink = Sink(lambda seg: seg.pcoord[-1, 0] > 1.0)
        sim.update_source_and_sinks(source, [sink])
        assert sim.source is source
        assert sim.sinks[0] is sink

    def test_invalid_source_type(self, sim):
        sink = Sink(lambda seg: seg.pcoord[-1, 0] > 1.0)
        with pytest.raises(TypeError, match="'source' must be a Source object"):
            sim.update_source_and_sinks("not_a_source", [sink])

    def test_invalid_sink_type(self, sim):
        source = Source(State(coord=[0.0]))
        with pytest.raises(TypeError, match="items in 'sinks' must be Sink objects"):
            sim.update_source_and_sinks(source, ["not_a_sink"])


# ---------------------------------------------------------------------------
# add_plugin tests
# ---------------------------------------------------------------------------


class TestAddPlugin:
    def test_add_valid_plugin(self, sim):
        plugin = Plugin(priority=0)
        sim.add_plugin(plugin)
        assert plugin in sim.plugins

    def test_add_invalid_plugin_type(self, sim):
        with pytest.raises(TypeError, match="'plugin' must be a Plugin"):
            sim.add_plugin("not_a_plugin")

    def test_plugins_sorted_by_priority(self, sim):
        p_high = Plugin(priority=10)
        p_low = Plugin(priority=0)
        sim.add_plugin(p_high)
        sim.add_plugin(p_low)
        ordered = list(sim.plugins)
        assert ordered[0] is p_low
        assert ordered[1] is p_high


# ---------------------------------------------------------------------------
# initialize tests
# ---------------------------------------------------------------------------


class TestInitialize:
    def test_initialize_single_state(self, sim, tmp_path):
        sim.initialize(State(coord=[0.5]))
        assert (tmp_path / "west.h5").exists()
        assert len(sim.segments) == 1

    def test_initialize_multiple_states(self, sim):
        states = [State(coord=[float(i)]) for i in range(4)]
        sim.initialize(states)
        assert len(sim.segments) == 4

    def test_initialize_uniform_weights(self, sim):
        states = [State(coord=[float(i)]) for i in range(4)]
        sim.initialize(states)
        weights = [seg.weight for seg in sim.segments]
        assert all(pytest.approx(w) == 0.25 for w in weights)

    def test_initialize_custom_weights_normalized(self, sim):
        states = [State(coord=[float(i)]) for i in range(3)]
        sim.initialize(states, weights=[1, 2, 1])
        weights = [seg.weight for seg in sim.segments]
        assert pytest.approx(sum(weights)) == 1.0
        # The segment with weight=2 gets 2/4 = 0.5
        assert pytest.approx(max(weights)) == 0.5

    def test_initialize_file_exists_raises(self, sim):
        state = State(coord=[0.5])
        sim.initialize(state)
        with pytest.raises(FileExistsError, match="can't initialize the simulation"):
            sim.initialize(state)

    def test_initialize_weights_length_mismatch(self, sim):
        states = [State(coord=[float(i)]) for i in range(3)]
        with pytest.raises(ValueError, match="length of 'weights' must match"):
            sim.initialize(states, weights=[0.5, 0.5])

    def test_initialize_segments_are_prepared(self, sim):
        states = [State(coord=[float(i)]) for i in range(2)]
        sim.initialize(states)
        for seg in sim.segments:
            assert seg.status == Segment.Status.PREPARED

    def test_initialize_segment_initial_states_match(self, sim):
        states = [State(coord=[0.0]), State(coord=[1.0])]
        sim.initialize(states)
        init_coords = {tuple(seg.initial_state.coord) for seg in sim.segments}
        assert (0.0,) in init_coords
        assert (1.0,) in init_coords

    def test_initialize_sets_iteration_to_one(self, sim):
        sim.initialize(State(coord=[0.5]))
        # current_iteration is HDF5-backed; file is still open right after initialize
        # (close_backing is called, but we can reopen)
        sim.data_manager.open_backing(mode='r')
        assert sim.current_iteration == 1
        sim.data_manager.close_backing()


# ---------------------------------------------------------------------------
# run integration tests
# ---------------------------------------------------------------------------


class TestRun:
    def test_run_one_iteration(self, sim):
        sim.initialize(State(coord=[0.0]))
        sim.run(n_iters=1)
        assert len(sim.segments) == 1  # NopMapper with target_count=1

    def test_run_multiple_iterations(self, sim, tmp_path):
        sim.initialize(State(coord=[0.0]))
        sim.run(n_iters=3)
        assert sim.current_iteration == 4  # started at 1, ran 3 iterations

    def test_run_preserves_total_probability(self, sim):
        states = [State(coord=[float(i) * 0.1]) for i in range(4)]
        sim.initialize(states)
        sim.run(n_iters=2)
        total_weight = sum(seg.weight for seg in sim.segments)
        assert pytest.approx(total_weight) == 1.0

    def test_run_with_rectilinear_bin_mapper(self, tmp_path, propagator):
        sim = Simulation(
            datafile=str(tmp_path / "west.h5"),
            propagator=propagator,
            pcoord_calculator=trivial_pcoord_calculator,
            bin_mapper=RectilinearBinMapper([[-np.inf, 0.5, np.inf]]),
            bin_target_counts=2,
        )
        states = [State(coord=[0.1]), State(coord=[0.2])]
        sim.initialize(states)
        sim.run(n_iters=2)
        assert sim.current_iteration == 3

    def test_run_calls_plugin_hooks(self, sim):
        hook_calls = []

        class TrackingPlugin(Plugin):
            def prepare_iteration(self, simulation):
                hook_calls.append('prepare_iteration')

            def finalize_iteration(self, simulation):
                hook_calls.append('finalize_iteration')

        sim.add_plugin(TrackingPlugin())
        sim.initialize(State(coord=[0.0]))
        sim.run(n_iters=2)

        assert hook_calls.count('prepare_iteration') == 2
        assert hook_calls.count('finalize_iteration') == 2

    def test_propagation_error_is_raised(self, datafile):
        sim = Simulation(
            datafile=datafile,
            propagator=FailingPropagator(seed=0),
            pcoord_calculator=trivial_pcoord_calculator,
        )
        sim.initialize(State(coord=[0.0]))
        from westpa.core.sim_manager import PropagationError

        with pytest.raises(PropagationError):
            sim.run(n_iters=1)

    def test_run_with_source_and_sink(self, tmp_path, propagator):
        """Walkers that reach the sink should be recycled to the source."""
        source = Source(State(coord=[0.0]))
        # Sink: any segment whose final pcoord > 0.5 is recycled
        sink = Sink(lambda seg: seg.pcoord[-1, 0] > 0.5)

        sim = Simulation(
            datafile=str(tmp_path / "west.h5"),
            propagator=TrivialPropagator(delta=1.0, seed=0),  # large step → always sinks
            pcoord_calculator=trivial_pcoord_calculator,
            source=source,
            sinks=sink,
        )
        states = [State(coord=[0.0])]
        sim.initialize(states)
        # Should not raise; recycled walkers are re-initiated from source
        sim.run(n_iters=2)
        total_weight = sum(seg.weight for seg in sim.segments)
        assert pytest.approx(total_weight) == 1.0
