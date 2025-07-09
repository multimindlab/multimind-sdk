import warnings
import pytest

def test_adapterconfig_deprecated():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        from multimind.fine_tuning.adapter_fusion import AdapterConfig
        cfg = AdapterConfig()
        assert any(issubclass(warn.category, DeprecationWarning) for warn in w)
        # Should proxy to LoraConfig
        assert hasattr(cfg, '_config')

def test_tasktype_deprecated():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        from multimind.fine_tuning.adapter_fusion import TaskType
        t = TaskType()
        assert any(issubclass(warn.category, DeprecationWarning) for warn in w)
        # Should proxy to PeftType
        assert hasattr(t, '_type')

def test_advanced_features_importable():
    from multimind.fine_tuning.adapter_fusion import AdapterFusionLayer, AdapterFusionTuner
    # Just check instantiation (not full functionality)
    layer = AdapterFusionLayer(in_features=8, out_features=8, num_adapters=2)
    tuner = AdapterFusionTuner(
        base_model_name='gpt2',
        output_dir='.',
        adapter_configs=[{"r": 8}],
    )
    assert layer is not None
    assert tuner is not None 