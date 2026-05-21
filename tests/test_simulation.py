"""Tests for the new Simulation API (westpa.core.simulation)."""

import pytest
import numpy as np
import westpa

from westpa.core.sim_manager import PropagationError
from westpa.core.binning import NopMapper
from westpa.work_managers import SerialWorkManager

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class TrivialPropagator(westpa.Propagator):
    """Moves a 1-D coordinate by a fixed delta each step."""

    def __init__(self, delta=0.1, **kwargs):
        super().__init__(**kwargs)
        self.delta = delta

    def propagate(self, segment, rng):
        coord = segment.initial_state.coord
        segment.final_state = westpa.State(coord=coord + self.delta)
        return segment


class FailingPropagator(westpa.Propagator):
    """Always marks segments as failed."""

    def propagate(self, segment):
        segment.mark_as_failed("deliberate test failure")
        return segment


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def datafile(tmp_path):
    return str(tmp_path / "west.h5")


@pytest.fixture
def propagator():
    return TrivialPropagator()


@pytest.fixture
def sim(datafile, propagator):
    return westpa.Simulation(
        datafile=datafile,
        propagator=propagator,
    )


# ---------------------------------------------------------------------------
# Constructor tests
# ---------------------------------------------------------------------------


class TestSimulationConstructor:
    def test_basic_construction(self, datafile, propagator):
        sim = westpa.Simulation(
            datafile=datafile,
            propagator=propagator,
        )
        assert sim.datafile == datafile
        assert sim.propagator is propagator

    def test_default_bin_mapper_is_nop(self, sim):
        assert isinstance(sim.bin_mapper, NopMapper)

    def test_default_bin_target_counts(self, sim):
        np.testing.assert_array_equal(sim.bin_target_counts, [1])

    def test_default_resampler_is_huber_kim(self, sim):
        assert isinstance(sim.resampler, westpa.HuberKimResampler)

    def test_default_work_manager_is_serial(self, sim):
        assert isinstance(sim.work_manager, SerialWorkManager)

    def test_default_source_is_none(self, sim):
        assert sim.source is None

    def test_default_sinks_is_empty(self, sim):
        assert len(sim.sinks) == 0

    def test_default_istate_generator_is_none(self, sim):
        assert sim.istate_generator is None

    def test_invalid_propagator_type(self, datafile):
        with pytest.raises(TypeError, match="'propagator' must be a Propagator object or None"):
            westpa.Simulation(
                datafile=datafile,
                propagator="not_a_propagator",
            )

    def test_invalid_pcoord_calculator_not_callable(self, datafile, propagator):
        with pytest.raises(TypeError, match="'pcoord_calculator' must be callable or None"):
            westpa.Simulation(
                datafile=datafile,
                propagator=propagator,
                pcoord_calculator="not_callable",
            )

    def test_invalid_resampler_type(self, datafile, propagator):
        with pytest.raises(TypeError, match="'resampler' must be a Resampler object"):
            westpa.Simulation(
                datafile=datafile,
                propagator=propagator,
                resampler="not_a_resampler",
            )

    def test_invalid_work_manager_type(self, datafile, propagator):
        with pytest.raises(TypeError, match="'work_manager' must be a WorkManager object"):
            westpa.Simulation(
                datafile=datafile,
                propagator=propagator,
                work_manager="not_a_work_manager",
            )

    def test_source_without_sink_raises(self, datafile, propagator):
        source = westpa.Source(westpa.State(coord=[0.0]))
        with pytest.raises(ValueError, match="'source' and 'sinks' must be provided together"):
            westpa.Simulation(
                datafile=datafile,
                propagator=propagator,
                source=source,
            )

    def test_sink_without_source_raises(self, datafile, propagator):
        sink = westpa.Sink(lambda seg: seg.pcoord[-1, 0] > 1.0)
        with pytest.raises(ValueError, match="'source' and 'sinks' must be provided together"):
            westpa.Simulation(
                datafile=datafile,
                propagator=propagator,
                sinks=[sink],
            )

    def test_source_and_sink_together(self, datafile, propagator):
        source = westpa.Source(westpa.State(coord=[0.0]))
        sink = westpa.Sink(lambda seg: seg.pcoord[-1, 0] > 1.0)
        sim = westpa.Simulation(
            datafile=datafile,
            propagator=propagator,
            source=source,
            sinks=[sink],
        )
        assert sim.source is source
        assert sim.sinks[0] is sink

    def test_invalid_istate_generator_type(self, datafile, propagator):
        with pytest.raises(TypeError, match="'istate_generator' must be callable"):
            westpa.Simulation(
                datafile=datafile,
                propagator=propagator,
                istate_generator="not_callable",
            )

    def test_plugins_added_in_priority_order(self, datafile, propagator):
        p_low = westpa.Plugin(priority=0)
        p_mid = westpa.Plugin(priority=5)
        p_high = westpa.Plugin(priority=10)
        sim = westpa.Simulation(
            datafile=datafile,
            propagator=propagator,
            plugins=[p_high, p_low, p_mid],
        )
        assert list(sim.plugins) == [p_low, p_mid, p_high]

    def test_custom_bin_mapper_and_target_counts(self, datafile, propagator):
        mapper = westpa.RectilinearBinMapper([[-np.inf, 0.5, np.inf]])  # 2 bins
        sim = westpa.Simulation(
            datafile=datafile,
            propagator=propagator,
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
        mapper = westpa.RectilinearBinMapper([[-np.inf, 0.5, np.inf]])  # 2 bins
        sim.update_bins(mapper, target_counts=5)
        np.testing.assert_array_equal(sim.bin_target_counts, [5, 5])

    def test_sequence_target_counts(self, sim):
        mapper = westpa.RectilinearBinMapper([[-np.inf, 0.5, np.inf]])  # 2 bins
        sim.update_bins(mapper, target_counts=[3, 7])
        np.testing.assert_array_equal(sim.bin_target_counts, [3, 7])
        assert sim.bin_mapper is mapper

    def test_invalid_mapper_type(self, sim):
        with pytest.raises(TypeError, match="'mapper' must be a BinMapper"):
            sim.update_bins("not_a_mapper", target_counts=1)

    def test_target_counts_length_mismatch(self, sim):
        mapper = westpa.RectilinearBinMapper([[-np.inf, 0.5, np.inf]])  # 2 bins
        with pytest.raises(ValueError, match="length of 'target_counts' must equal"):
            sim.update_bins(mapper, target_counts=[1, 2, 3])

    def test_target_counts_zero_raises(self, sim):
        mapper = westpa.RectilinearBinMapper([[-np.inf, 0.5, np.inf]])  # 2 bins
        with pytest.raises(ValueError, match="'target_counts' must be positive"):
            sim.update_bins(mapper, target_counts=[0, 1])

    def test_target_counts_negative_raises(self, sim):
        mapper = westpa.RectilinearBinMapper([[-np.inf, 0.5, np.inf]])  # 2 bins
        with pytest.raises(ValueError, match="'target_counts' must be positive"):
            sim.update_bins(mapper, target_counts=[-1, 1])


# ---------------------------------------------------------------------------
# update_source_and_sinks tests
# ---------------------------------------------------------------------------


class TestUpdateSourceAndSinks:
    def test_valid_update(self, sim):
        source = westpa.Source(westpa.State(coord=[0.0]))
        sink = westpa.Sink(lambda seg: seg.pcoord[-1, 0] > 1.0)
        sim.update_source_and_sinks(source, [sink])
        assert sim.source is source
        assert sim.sinks[0] is sink

    def test_invalid_source_type(self, sim):
        sink = westpa.Sink(lambda seg: seg.pcoord[-1, 0] > 1.0)
        with pytest.raises(TypeError, match="'source' must be a Source object"):
            sim.update_source_and_sinks("not_a_source", [sink])

    def test_invalid_sink_type(self, sim):
        source = westpa.Source(westpa.State(coord=[0.0]))
        with pytest.raises(TypeError, match="'sinks' must be a Sink object or an iterable of Sink objects"):
            sim.update_source_and_sinks(source, ["not_a_sink"])


# ---------------------------------------------------------------------------
# add_plugin tests
# ---------------------------------------------------------------------------


class TestAddPlugin:
    def test_add_valid_plugin(self, sim):
        plugin = westpa.Plugin(priority=0)
        sim.add_plugin(plugin)
        assert plugin in sim.plugins

    def test_add_invalid_plugin_type(self, sim):
        with pytest.raises(TypeError, match="'plugin' must be a Plugin"):
            sim.add_plugin("not_a_plugin")

    def test_plugins_sorted_by_priority(self, sim):
        p_high = westpa.Plugin(priority=10)
        p_low = westpa.Plugin(priority=0)
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
        sim.initialize(westpa.State(coord=[0.5]))
        assert (tmp_path / "west.h5").exists()
        assert len(sim.get_segments()) == 1

    def test_initialize_multiple_states(self, sim):
        states = [westpa.State(coord=[float(i)]) for i in range(4)]
        sim.initialize(states)
        assert len(sim.get_segments()) == 4

    def test_initialize_uniform_weights(self, sim):
        states = [westpa.State(coord=[float(i)]) for i in range(4)]
        sim.initialize(states)
        weights = [seg.weight for seg in sim.get_segments()]
        assert all(pytest.approx(w) == 0.25 for w in weights)

    def test_initialize_custom_weights_normalized(self, sim):
        states = [westpa.State(coord=[float(i)]) for i in range(3)]
        sim.initialize(states, weights=[1, 2, 1])
        weights = [seg.weight for seg in sim.get_segments()]
        assert pytest.approx(sum(weights)) == 1.0
        # The segment with weight=2 gets 2/4 = 0.5
        assert pytest.approx(max(weights)) == 0.5

    def test_initialize_runtime_raises(self, sim):
        state = westpa.State(coord=[0.5])
        sim.initialize(state)
        with pytest.raises(RuntimeError, match="can't initialize the simulation"):
            sim.initialize(state)

    def test_initialize_weights_length_mismatch(self, sim):
        states = [westpa.State(coord=[float(i)]) for i in range(3)]
        with pytest.raises(ValueError, match="length of 'weights' must match"):
            sim.initialize(states, weights=[0.5, 0.5])

    def test_initialize_segments_are_prepared(self, sim):
        states = [westpa.State(coord=[float(i)]) for i in range(2)]
        sim.initialize(states)
        for seg in sim.get_segments():
            assert seg.status == westpa.Segment.Status.PREPARED

    def test_initialize_segment_initial_states_match(self, sim):
        states = [westpa.State(coord=[0.0]), westpa.State(coord=[1.0])]
        sim.initialize(states)
        init_coords = {tuple(seg.initial_state.coord) for seg in sim.get_segments()}
        assert (0.0,) in init_coords
        assert (1.0,) in init_coords

    def test_initialize_sets_iteration_to_one(self, sim):
        sim.initialize(westpa.State(coord=[0.5]))
        assert sim.current_iteration == 1


# ---------------------------------------------------------------------------
# run integration tests
# ---------------------------------------------------------------------------


class TestRun:
    def test_run_one_iteration(self, sim):
        sim.initialize(westpa.State(coord=[0.0]))
        sim.run(n_iters=1)
        assert len(sim.get_segments()) == 1  # NopMapper with target_count=1

    def test_run_multiple_iterations(self, sim):
        sim.initialize(westpa.State(coord=[0.0]))
        sim.run(n_iters=3)
        assert sim.current_iteration == 4  # started at 1, ran 3 iterations

    def test_run_preserves_total_probability(self, sim):
        states = [westpa.State(coord=[float(i) * 0.1]) for i in range(4)]
        sim.initialize(states)
        sim.run(n_iters=2)
        total_weight = sum(seg.weight for seg in sim.get_segments())
        assert pytest.approx(total_weight) == 1.0

    def test_run_with_rectilinear_bin_mapper(self, datafile, propagator):
        sim = westpa.Simulation(
            datafile=datafile,
            propagator=propagator,
            bin_mapper=westpa.RectilinearBinMapper([[-np.inf, 0.5, np.inf]]),
            bin_target_counts=2,
        )
        states = [westpa.State(coord=[0.1]), westpa.State(coord=[0.2])]
        sim.initialize(states)
        sim.run(n_iters=2)
        assert sim.current_iteration == 3

    def test_run_calls_plugin_hooks(self, sim):
        hook_calls = []

        class TrackingPlugin(westpa.Plugin):
            def prepare_run(self, sim):
                hook_calls.append('prepare_run')

            def finalize_run(self, sim):
                hook_calls.append('finalize_run')

            def prepare_iteration(self, sim):
                hook_calls.append('prepare_iteration')

            def pre_we(self, sim):
                hook_calls.append('pre_we')

        sim.add_plugin(TrackingPlugin())
        sim.initialize(westpa.State(coord=[0.0]))
        sim.run(n_iters=2)

        assert hook_calls.count('prepare_run') == 1
        assert hook_calls.count('prepare_iteration') == 2
        assert hook_calls.count('pre_we') == 2
        assert hook_calls.count('finalize_run') == 1

    def test_propagation_error_is_raised(self, datafile):
        sim = westpa.Simulation(
            datafile=datafile,
            propagator=FailingPropagator(),
        )
        sim.initialize(westpa.State(coord=[0.0]))
        with pytest.raises(PropagationError):
            sim.run(n_iters=1)

    def test_run_with_source_and_sink(self, datafile, propagator):
        """Walkers that reach the sink should be recycled to the source."""
        source = westpa.Source(westpa.State(coord=[0.0]))
        # Sink: any segment whose final pcoord > 0.5 is recycled
        sink = westpa.Sink(lambda seg: seg.pcoord[-1, 0] > 0.5)

        sim = westpa.Simulation(
            datafile=datafile,
            propagator=TrivialPropagator(delta=1.0),  # large step → always sinks
            source=source,
            sinks=sink,
        )
        states = [westpa.State(coord=[0.0])]
        sim.initialize(states)

        sim.run(n_iters=2)
        total_weight = sum(seg.weight for seg in sim.get_segments())
        assert pytest.approx(total_weight) == 1.0

        for segment in sim.get_segments():
            assert segment.initial_state is None
            assert segment.status == segment.Status.UNSET

    def test_continue_run(self, datafile, propagator):
        sim = westpa.Simulation(
            datafile=datafile,
            propagator=propagator,
        )
        sim.initialize(westpa.State(coord=[0.0]))
        sim.run(n_iters=2)
        del sim

        # continue run from last checkpoint in 'datafile'
        sim2 = westpa.Simulation(
            datafile=datafile,
            propagator=propagator,
        )
        with pytest.raises(RuntimeError, match="already initialized"):
            sim2.initialize(westpa.State(coord=[0.0]))
        sim2.run(n_iters=2)
        assert sim2.current_iteration == 5
