"""
Test Suite for Recording Catalog System

Run with: pytest test_recording_catalog.py -v
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil

from braindance.utils.data_manager import (
    Recording, RecordingCatalog, DataContext,
    load_catalog, load_recording
)
# BatchResults is an internal class, accessed via catalog.apply_batch()


# ==================== Fixtures ====================

@pytest.fixture
def sample_catalog_df():
    """Create a sample catalog DataFrame for testing."""
    return pd.DataFrame({
        'proj': ['proj1', 'proj1', 'proj1', 'proj2', 'proj2'],
        'chip': ['chip_a', 'chip_a', 'chip_b', 'chip_c', 'chip_c'],
        'experiment': ['exp1', 'exp2', 'exp1', 'exp1', 'exp2'],
        'freq': [0, 10, 5, 0, 20],
        'baseline': [True, False, False, True, False],
        'drug': [None, 'bicuculline', None, None, 'apv'],
        'type': ['spontaneous', 'stimulated', 'stimulated', 'spontaneous', 'stimulated'],
        'data_path': [f'/data/rec{i}' for i in range(5)],
    })


@pytest.fixture
def sample_catalog(sample_catalog_df):
    """Create a sample RecordingCatalog."""
    return RecordingCatalog(sample_catalog_df)


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory with mock data files."""
    tmpdir = tempfile.mkdtemp()
    
    # Create mock data files
    # STUB: Create mock spike data, stim logs, etc.
    
    yield Path(tmpdir)
    
    # Cleanup
    shutil.rmtree(tmpdir)


@pytest.fixture
def sample_row(sample_catalog_df):
    """Get a single row for Recording tests."""
    return sample_catalog_df.iloc[0]


# ==================== DataContext Tests ====================

class TestDataContext:
    """Tests for DataContext lazy loading and persistence."""
    
    def test_init_empty(self):
        """DataContext initializes with no data."""
        ctx = DataContext()
        assert len(ctx.keys()) == 0
    
    def test_set_and_get_attribute(self):
        """Can set and get data via attributes."""
        ctx = DataContext()
        ctx.my_data = [1, 2, 3]
        assert ctx.my_data == [1, 2, 3]
    
    def test_contains(self):
        """__contains__ works for checking keys."""
        ctx = DataContext()
        ctx.my_data = [1, 2, 3]
        assert 'my_data' in ctx
        assert 'other' not in ctx
    
    def test_get_with_default(self):
        """get() returns default for missing keys."""
        ctx = DataContext()
        assert ctx.get('missing', 'default') == 'default'
    
    def test_overwrite_false_prevents_overwrite(self):
        """overwrite_existing=False prevents overwriting."""
        ctx = DataContext(overwrite_existing=False)
        ctx.my_data = [1, 2, 3]
        ctx.my_data = [4, 5, 6]  # Should be ignored
        assert ctx.my_data == [1, 2, 3]
    
    def test_overwrite_true_allows_overwrite(self):
        """overwrite_existing=True allows overwriting."""
        ctx = DataContext(overwrite_existing=True)
        ctx.my_data = [1, 2, 3]
        ctx.my_data = [4, 5, 6]
        assert ctx.my_data == [4, 5, 6]
    
    def test_save_and_load_numpy(self, temp_data_dir):
        """Save and load numpy arrays."""
        ctx = DataContext()
        ctx.array_data = np.array([1, 2, 3])
        ctx.save(temp_data_dir)
        
        ctx2 = DataContext(path=temp_data_dir)
        assert np.array_equal(ctx2.array_data, np.array([1, 2, 3]))
    
    def test_save_and_load_pickle(self, temp_data_dir):
        """Save and load pickle data."""
        ctx = DataContext()
        ctx.dict_data = {'a': 1, 'b': 2}
        ctx.save(temp_data_dir)
        
        ctx2 = DataContext(path=temp_data_dir)
        assert ctx2.dict_data == {'a': 1, 'b': 2}
    
    def test_lazy_load_on_access(self, temp_data_dir):
        """Data is loaded lazily when accessed."""
        # Save some data
        ctx = DataContext()
        ctx.lazy_data = np.array([1, 2, 3])
        ctx.save(temp_data_dir)
        
        # Create new context pointing to same path
        ctx2 = DataContext(path=temp_data_dir)
        assert 'lazy_data' not in ctx2._data  # Not loaded yet
        _ = ctx2.lazy_data  # Access triggers load
        assert 'lazy_data' in ctx2._data  # Now loaded
    
    def test_keys_includes_disk_files(self, temp_data_dir):
        """keys() includes files on disk not yet loaded."""
        ctx = DataContext()
        ctx.saved_data = [1, 2, 3]
        ctx.save(temp_data_dir)
        
        ctx2 = DataContext(path=temp_data_dir)
        assert 'saved_data' in ctx2.keys()


# ==================== Recording Tests ====================

class TestRecording:
    """Tests for Recording class."""
    
    def test_init_from_row(self, sample_row):
        """Recording initializes from catalog row."""
        rec = Recording(sample_row)
        assert rec is not None
    
    def test_repr(self, sample_row):
        """Recording has useful repr."""
        rec = Recording(sample_row)
        repr_str = repr(rec)
        assert 'Recording' in repr_str
    
    def test_metadata_access_via_attribute(self, sample_row):
        """Can access catalog columns via attributes."""
        rec = Recording(sample_row)
        assert rec.chip == 'chip_a'
        assert rec.freq == 0
        assert rec.baseline == True
    
    def test_metadata_access_via_getitem(self, sample_row):
        """Can access catalog columns via [] notation."""
        rec = Recording(sample_row)
        assert rec['chip'] == 'chip_a'
        assert rec['freq'] == 0
    
    def test_contains_metadata(self, sample_row):
        """__contains__ works for metadata."""
        rec = Recording(sample_row)
        assert 'chip' in rec
        assert 'nonexistent' not in rec
    
    def test_metadata_property(self, sample_row):
        """metadata property returns dict of all metadata."""
        rec = Recording(sample_row)
        meta = rec.metadata
        assert isinstance(meta, dict)
        assert 'chip' in meta
    
    def test_identifier_property(self, sample_row):
        """identifier property builds path-like string."""
        rec = Recording(sample_row)
        ident = rec.identifier
        assert 'proj1' in ident
        assert 'chip_a' in ident
    
    def test_results_is_datacontext(self, sample_row):
        """results property returns DataContext."""
        rec = Recording(sample_row)
        assert isinstance(rec.results, DataContext)
    
    def test_results_persist(self, sample_row, temp_data_dir):
        """Results can be saved and loaded."""
        rec = Recording(sample_row, base_path=temp_data_dir)
        rec.results.connectivity = np.eye(3)
        rec.save_results()
        
        # New recording pointing to same location
        rec2 = Recording(sample_row, base_path=temp_data_dir)
        assert 'connectivity' in rec2.results
    
    def test_spikes_lazy_load(self, sample_row, temp_data_dir):
        """spikes property triggers lazy load."""
        # STUB: Needs mock spike data to be created
        pass
    
    def test_stim_log_lazy_load(self, sample_row, temp_data_dir):
        """stim_log property triggers lazy load."""
        # STUB: Needs mock stim log to be created
        pass
    
    def test_clear_cache(self, sample_row):
        """clear_cache clears loaded data."""
        rec = Recording(sample_row)
        rec._data._data['test'] = 'value'
        rec.clear_cache()
        assert 'test' not in rec._data._data
    
    def test_info(self, sample_row):
        """info() returns summary dict."""
        rec = Recording(sample_row)
        info = rec.info()
        assert 'identifier' in info
        assert 'metadata' in info
    
    def test_load_from_path(self, temp_data_dir):
        """Recording.load() creates recording from path."""
        # STUB: Needs mock directory structure
        pass


# ==================== RecordingCatalog Tests ====================

class TestRecordingCatalogBasics:
    """Basic RecordingCatalog functionality."""
    
    def test_init_from_dataframe(self, sample_catalog_df):
        """RecordingCatalog initializes from DataFrame."""
        catalog = RecordingCatalog(sample_catalog_df)
        assert len(catalog) == 5
    
    def test_repr(self, sample_catalog):
        """RecordingCatalog has useful repr."""
        repr_str = repr(sample_catalog)
        assert 'RecordingCatalog' in repr_str
        assert '5' in repr_str
    
    def test_len(self, sample_catalog):
        """len() returns number of recordings."""
        assert len(sample_catalog) == 5
    
    def test_columns_property(self, sample_catalog):
        """columns property lists available columns."""
        cols = sample_catalog.columns
        assert 'chip' in cols
        assert 'freq' in cols


class TestRecordingCatalogIndexing:
    """Tests for catalog indexing and slicing."""
    
    def test_integer_index(self, sample_catalog):
        """Integer index returns Recording."""
        rec = sample_catalog[0]
        assert isinstance(rec, Recording)
    
    def test_negative_index(self, sample_catalog):
        """Negative index works."""
        rec = sample_catalog[-1]
        assert isinstance(rec, Recording)
        assert rec.chip == 'chip_c'
    
    def test_index_out_of_range(self, sample_catalog):
        """Out of range index raises IndexError."""
        with pytest.raises(IndexError):
            _ = sample_catalog[100]
    
    def test_slice_returns_catalog(self, sample_catalog):
        """Slice returns new RecordingCatalog."""
        sliced = sample_catalog[1:3]
        assert isinstance(sliced, RecordingCatalog)
        assert len(sliced) == 2
    
    def test_boolean_mask_pandas_series(self, sample_catalog):
        """Boolean pd.Series filters catalog."""
        mask = sample_catalog.freq > 0
        filtered = sample_catalog[mask]
        assert isinstance(filtered, RecordingCatalog)
        assert len(filtered) == 3
    
    def test_boolean_mask_numpy_array(self, sample_catalog):
        """Boolean np.array filters catalog."""
        mask = np.array([True, False, True, False, True])
        filtered = sample_catalog[mask]
        assert len(filtered) == 3
    
    def test_boolean_mask_list(self, sample_catalog):
        """Boolean list filters catalog."""
        mask = [True, False, True, False, True]
        filtered = sample_catalog[mask]
        assert len(filtered) == 3


class TestRecordingCatalogIteration:
    """Tests for catalog iteration."""
    
    def test_iteration(self, sample_catalog):
        """Can iterate over catalog."""
        recordings = list(sample_catalog)
        assert len(recordings) == 5
        assert all(isinstance(r, Recording) for r in recordings)
    
    def test_iteration_preserves_order(self, sample_catalog):
        """Iteration order matches index order."""
        for i, rec in enumerate(sample_catalog):
            assert rec.chip == sample_catalog[i].chip


class TestRecordingCatalogFiltering:
    """Tests for filter() method."""
    
    def test_filter_exact_match(self, sample_catalog):
        """Exact match filtering."""
        filtered = sample_catalog.filter(chip='chip_a')
        assert len(filtered) == 2
    
    def test_filter_gt(self, sample_catalog):
        """Greater than filtering."""
        filtered = sample_catalog.filter(freq__gt=0)
        assert len(filtered) == 3
        assert all(rec.freq > 0 for rec in filtered)
    
    def test_filter_gte(self, sample_catalog):
        """Greater than or equal filtering."""
        filtered = sample_catalog.filter(freq__gte=10)
        assert len(filtered) == 2
    
    def test_filter_lt(self, sample_catalog):
        """Less than filtering."""
        filtered = sample_catalog.filter(freq__lt=10)
        assert len(filtered) == 3
    
    def test_filter_lte(self, sample_catalog):
        """Less than or equal filtering."""
        filtered = sample_catalog.filter(freq__lte=5)
        assert len(filtered) == 3
    
    def test_filter_in(self, sample_catalog):
        """In list filtering."""
        filtered = sample_catalog.filter(freq__in=[0, 10])
        assert len(filtered) == 3
    
    def test_filter_contains(self, sample_catalog):
        """String contains filtering."""
        filtered = sample_catalog.filter(chip__contains='_a')
        assert len(filtered) == 2
    
    def test_filter_isnull_true(self, sample_catalog):
        """Is null filtering."""
        filtered = sample_catalog.filter(drug__isnull=True)
        assert len(filtered) == 3
    
    def test_filter_isnull_false(self, sample_catalog):
        """Is not null filtering."""
        filtered = sample_catalog.filter(drug__isnull=False)
        assert len(filtered) == 2
    
    def test_filter_multiple_conditions(self, sample_catalog):
        """Multiple conditions in one filter call."""
        filtered = sample_catalog.filter(freq__gt=0, chip='chip_a')
        assert len(filtered) == 1
    
    def test_filter_chainable(self, sample_catalog):
        """Filter calls are chainable."""
        filtered = sample_catalog.filter(freq__gt=0).filter(chip='chip_a')
        assert len(filtered) == 1
    
    def test_filter_unknown_column_raises(self, sample_catalog):
        """Unknown column raises ValueError."""
        with pytest.raises(ValueError):
            sample_catalog.filter(nonexistent='value')
    
    def test_filter_unknown_lookup_raises(self, sample_catalog):
        """Unknown lookup type raises ValueError."""
        with pytest.raises(ValueError):
            sample_catalog.filter(freq__badlookup=5)


class TestRecordingCatalogColumnAccess:
    """Tests for column attribute access."""
    
    def test_column_as_attribute(self, sample_catalog):
        """Column accessible as attribute returns Series."""
        freqs = sample_catalog.freq
        assert isinstance(freqs, pd.Series)
        assert len(freqs) == 5
    
    def test_column_for_filtering(self, sample_catalog):
        """Column attribute usable in filter expression."""
        filtered = sample_catalog[sample_catalog.freq > 0]
        assert len(filtered) == 3
    
    def test_combined_column_filter(self, sample_catalog):
        """Multiple column conditions combinable."""
        mask = (sample_catalog.freq > 0) & (sample_catalog.baseline == False)
        filtered = sample_catalog[mask]
        assert len(filtered) == 3


class TestRecordingCatalogBatchAccess:
    """Tests for batch data access."""
    
    def test_batch_spikes_returns_list(self, sample_catalog):
        """catalog.spikes returns list."""
        # STUB: Needs mock data
        pass
    
    def test_batch_results(self, sample_catalog):
        """catalog.results.X returns list."""
        # First add some results
        for rec in sample_catalog:
            rec.results._data['test_result'] = rec.freq * 2
        
        results = sample_catalog.results.test_result
        assert isinstance(results, list)
        assert len(results) == 5
    
    def test_batch_results_available(self, sample_catalog):
        """results.available() returns counts."""
        for i, rec in enumerate(sample_catalog):
            if i < 3:
                rec.results._data['partial_result'] = i
        
        available = sample_catalog.results.available()
        assert available.get('partial_result', 0) == 3


class TestRecordingCatalogCollect:
    """Tests for collect() method."""
    
    def test_collect_simple_attribute(self, sample_catalog):
        """Collect simple attribute."""
        chips = sample_catalog.collect('chip')
        assert chips == ['chip_a', 'chip_a', 'chip_b', 'chip_c', 'chip_c']
    
    def test_collect_nested_attribute(self, sample_catalog):
        """Collect nested attribute like results.X."""
        for rec in sample_catalog:
            rec.results._data['nested'] = rec.freq
        
        values = sample_catalog.collect('results.nested')
        assert values == [0, 10, 5, 0, 20]


class TestRecordingCatalogGroupBy:
    """Tests for group_by() method."""
    
    def test_group_by_returns_dict(self, sample_catalog):
        """group_by returns dict of catalogs."""
        groups = sample_catalog.group_by('chip')
        assert isinstance(groups, dict)
        assert len(groups) == 3  # chip_a, chip_b, chip_c
    
    def test_group_by_values_are_catalogs(self, sample_catalog):
        """Grouped values are RecordingCatalogs."""
        groups = sample_catalog.group_by('chip')
        for key, catalog in groups.items():
            assert isinstance(catalog, RecordingCatalog)
    
    def test_group_by_correct_counts(self, sample_catalog):
        """Group sizes are correct."""
        groups = sample_catalog.group_by('chip')
        assert len(groups['chip_a']) == 2
        assert len(groups['chip_b']) == 1
        assert len(groups['chip_c']) == 2


class TestRecordingCatalogClassMethods:
    """Tests for class construction methods."""
    
    def test_from_csv(self, temp_data_dir, sample_catalog_df):
        """from_csv creates catalog from CSV file."""
        csv_path = temp_data_dir / 'catalog.csv'
        sample_catalog_df.to_csv(csv_path, index=False)
        
        catalog = RecordingCatalog.from_csv(csv_path)
        assert len(catalog) == 5
    
    def test_from_dataframe(self, sample_catalog_df):
        """from_dataframe creates catalog from DataFrame."""
        catalog = RecordingCatalog.from_dataframe(sample_catalog_df)
        assert len(catalog) == 5
    
    def test_from_directory(self, temp_data_dir):
        """from_directory scans directory for recordings."""
        # STUB: Needs mock directory structure
        pass


class TestRecordingCatalogCaching:
    """Tests for Recording caching in catalog."""
    
    def test_same_index_returns_cached(self, sample_catalog):
        """Accessing same index returns cached Recording."""
        rec1 = sample_catalog[0]
        rec2 = sample_catalog[0]
        assert rec1 is rec2
    
    def test_different_index_returns_different(self, sample_catalog):
        """Different indices return different Recordings."""
        rec1 = sample_catalog[0]
        rec2 = sample_catalog[1]
        assert rec1 is not rec2


# ==================== Integration Tests ====================

class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_full_workflow(self, sample_catalog_df, temp_data_dir):
        """Test typical analysis workflow."""
        # Create catalog
        catalog = RecordingCatalog(sample_catalog_df, base_path=temp_data_dir)
        
        # Filter
        stimulated = catalog.filter(baseline=False, freq__gt=0)
        assert len(stimulated) == 3
        
        # Process each
        for rec in stimulated:
            rec.results._data['processed'] = True
        
        # Batch check
        processed = stimulated.results.processed
        assert all(processed)
    
    def test_pandas_and_list_style_combined(self, sample_catalog):
        """Can use pandas-style filtering then list-style access."""
        filtered = sample_catalog[sample_catalog.freq > 0]
        rec = filtered[0]
        assert rec.freq > 0
    
    def test_chained_operations(self, sample_catalog):
        """Complex chained operations work."""
        result = (
            sample_catalog
            .filter(freq__gt=0)
            .filter(chip__contains='chip')
            [:2]
        )
        assert len(result) == 2


# ==================== Edge Cases ====================

class TestEdgeCases:
    """Edge cases and error handling."""
    
    def test_empty_catalog(self):
        """Empty catalog works."""
        catalog = RecordingCatalog(pd.DataFrame())
        assert len(catalog) == 0
        assert list(catalog) == []
    
    def test_filter_to_empty(self, sample_catalog):
        """Filtering to empty result works."""
        empty = sample_catalog.filter(freq__gt=1000)
        assert len(empty) == 0
    
    def test_missing_column_attribute(self, sample_catalog):
        """Accessing non-existent column raises AttributeError."""
        with pytest.raises(AttributeError):
            _ = sample_catalog.nonexistent_column
    
    def test_recording_missing_attribute(self, sample_row):
        """Recording raises AttributeError for missing attribute."""
        rec = Recording(sample_row)
        with pytest.raises(AttributeError):
            _ = rec.nonexistent_attribute
