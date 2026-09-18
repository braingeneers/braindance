import pytest
import pandas as pd
from braindance.utils.data_manager import RecordingCatalog, Recording

def test_catalog_filtering_django_style(mock_catalog_df):
    """Verify Django-style filtering (e.g., freq__gt)."""
    # Create a larger catalog for filtering
    df = pd.concat([mock_catalog_df] * 5, ignore_index=True)
    df['freq'] = [0, 10, 20, 30, 40]
    catalog = RecordingCatalog(df)
    
    # Filter freq > 10
    filtered = catalog.filter(freq__gt=10)
    assert len(filtered) == 3
    assert all(filtered.freq > 10)
    
    # Filter chip exact match
    filtered = catalog.filter(chip='test_chip')
    assert len(filtered) == 5

def test_catalog_indexing(mock_catalog_df):
    """Verify integer and slice indexing."""
    df = pd.concat([mock_catalog_df] * 3, ignore_index=True)
    catalog = RecordingCatalog(df)
    
    # Integer index
    rec = catalog[0]
    assert isinstance(rec, Recording)
    
    # Slice
    sub_catalog = catalog[0:2]
    assert isinstance(sub_catalog, RecordingCatalog)
    assert len(sub_catalog) == 2

def test_catalog_attribute_access(mock_catalog_df):
    """Verify that columns are accessible as attributes."""
    catalog = RecordingCatalog(mock_catalog_df)
    assert isinstance(catalog.chip, pd.Series)
    assert catalog.chip[0] == 'test_chip'

def test_catalog_iteration(mock_catalog_df):
    """Verify that we can iterate over recordings."""
    df = pd.concat([mock_catalog_df] * 2, ignore_index=True)
    catalog = RecordingCatalog(df)
    
    recs = list(catalog)
    assert len(recs) == 2
    assert all(isinstance(r, Recording) for r in recs)

def test_catalog_batch_results(mock_recording, mock_catalog_df):
    """Verify batch results access (catalog.results.X)."""
    # Create catalog and manually inject a result into the recording
    catalog = RecordingCatalog(mock_catalog_df)
    # We need to make sure the catalog[0] returns our mock_recording or similar
    # In practice, RecordingCatalog creates new Recording objects, 
    # so we might need to mock the Recording creation or just use the real one and mock its results.
    
    # Let's mock a result for the first recording
    rec = catalog[0]
    rec.results.test_val = 123
    
    assert 123 in catalog.results.test_val
    assert catalog.results.available()['test_val'] == 1

def test_catalog_empty_handling():
    """Verify handling of empty catalog."""
    catalog = RecordingCatalog(pd.DataFrame(columns=['chip', 'experiment']))
    assert len(catalog) == 0
    assert catalog.filter(chip='none') is not None
    assert len(catalog.filter(chip='none')) == 0
