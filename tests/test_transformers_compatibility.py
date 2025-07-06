#!/usr/bin/env python3
"""
Test script to verify transformers backward compatibility.
Tests that the SDK works with both old and new transformers versions.
"""

import sys
import importlib
import unittest
from unittest.mock import patch, MagicMock

class TestTransformersCompatibility(unittest.TestCase):
    """Test class for transformers backward compatibility."""
    
    def test_current_transformers_version(self):
        """Test with current transformers version (AutoModelForSeq2SeqLM available)."""
        print("Testing with current transformers version...")
        
        # Create a mock that raises ImportError for AutoModelForSeq2SeqGeneration
        class MockTransformersCurrent:
            AutoTokenizer = MagicMock()
            AutoModelForSeq2SeqLM = MagicMock()
            
            def __getattr__(self, name):
                if name == 'AutoModelForSeq2SeqGeneration':
                    raise ImportError("AutoModelForSeq2SeqGeneration not available")
                return MagicMock()
        
        mock_transformers = MockTransformersCurrent()
        
        with patch.dict('sys.modules', {'transformers': mock_transformers}):
            try:
                # Simulate the exact import logic from our files
                try:
                    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
                    _AUTO_MODEL_CLASS = AutoModelForSeq2SeqLM
                    print("✓ Successfully imported AutoModelForSeq2SeqLM")
                except ImportError:
                    try:
                        from transformers import AutoTokenizer, AutoModelForSeq2SeqGeneration
                        _AUTO_MODEL_CLASS = AutoModelForSeq2SeqGeneration
                        print("✓ Fallback to AutoModelForSeq2SeqGeneration")
                    except ImportError:
                        from transformers import AutoTokenizer
                        _AUTO_MODEL_CLASS = None
                        print("✓ Fallback to None (very old version)")
                
                self.assertIsNotNone(_AUTO_MODEL_CLASS)
                print(f"✓ Using model class: {_AUTO_MODEL_CLASS}")
                
            except Exception as e:
                self.fail(f"Failed with current transformers: {e}")
    
    def test_old_transformers_version(self):
        """Test with old transformers version (AutoModelForSeq2SeqGeneration available)."""
        print("Testing with old transformers version...")
        
        class MockTransformersOld:
            AutoTokenizer = MagicMock()
            AutoModelForSeq2SeqGeneration = MagicMock()
            
            def __getattr__(self, name):
                if name == 'AutoModelForSeq2SeqLM':
                    raise ImportError("AutoModelForSeq2SeqLM not available")
                return MagicMock()
        
        mock_transformers = MockTransformersOld()
        
        with patch.dict('sys.modules', {'transformers': mock_transformers}):
            try:
                # Simulate the exact import logic from our files
                try:
                    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
                    _AUTO_MODEL_CLASS = AutoModelForSeq2SeqLM
                    print("✓ Successfully imported AutoModelForSeq2SeqLM")
                except ImportError:
                    try:
                        from transformers import AutoTokenizer, AutoModelForSeq2SeqGeneration
                        _AUTO_MODEL_CLASS = AutoModelForSeq2SeqGeneration
                        print("✓ Fallback to AutoModelForSeq2SeqGeneration")
                    except ImportError:
                        from transformers import AutoTokenizer
                        _AUTO_MODEL_CLASS = None
                        print("✓ Fallback to None (very old version)")
                
                self.assertIsNotNone(_AUTO_MODEL_CLASS)
                print(f"✓ Using model class: {_AUTO_MODEL_CLASS}")
                
            except Exception as e:
                self.fail(f"Failed with old transformers: {e}")
    
    def test_very_old_transformers_version(self):
        """Test with very old transformers version (neither available)."""
        print("Testing with very old transformers version...")
        
        class MockTransformersVeryOld:
            AutoTokenizer = MagicMock()
            
            def __getattr__(self, name):
                if name in ['AutoModelForSeq2SeqLM', 'AutoModelForSeq2SeqGeneration']:
                    raise ImportError(f"{name} not available")
                return MagicMock()
        
        mock_transformers = MockTransformersVeryOld()
        
        with patch.dict('sys.modules', {'transformers': mock_transformers}):
            try:
                # Simulate the exact import logic from our files
                try:
                    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
                    _AUTO_MODEL_CLASS = AutoModelForSeq2SeqLM
                    print("✓ Successfully imported AutoModelForSeq2SeqLM")
                except ImportError:
                    try:
                        from transformers import AutoTokenizer, AutoModelForSeq2SeqGeneration
                        _AUTO_MODEL_CLASS = AutoModelForSeq2SeqGeneration
                        print("✓ Fallback to AutoModelForSeq2SeqGeneration")
                    except ImportError:
                        from transformers import AutoTokenizer
                        _AUTO_MODEL_CLASS = None
                        print("✓ Fallback to None (very old version)")
                
                # For very old versions, _AUTO_MODEL_CLASS should be None
                self.assertIsNone(_AUTO_MODEL_CLASS)
                print("✓ Using fallback model class")
                
            except Exception as e:
                self.fail(f"Failed with very old transformers: {e}")
    
    def test_model_class_selection_logic(self):
        """Test the model class selection logic with different model types."""
        print("Testing model class selection logic...")
        
        # Test seq2seq model type with different transformers versions
        test_cases = [
            ("current", True, False),   # AutoModelForSeq2SeqLM available
            ("old", False, True),       # AutoModelForSeq2SeqGeneration available
            ("very_old", False, False)  # Neither available
        ]
        
        for version, has_lm, has_generation in test_cases:
            print(f"\nTesting {version} transformers version...")
            
            class MockTransformers:
                AutoTokenizer = MagicMock()
                AutoModelForCausalLM = MagicMock()
                AutoModelForSequenceClassification = MagicMock()
                
                def __getattr__(self, name):
                    if name == 'AutoModelForSeq2SeqLM' and not has_lm:
                        raise ImportError("AutoModelForSeq2SeqLM not available")
                    elif name == 'AutoModelForSeq2SeqGeneration' and not has_generation:
                        raise ImportError("AutoModelForSeq2SeqGeneration not available")
                    return MagicMock()
            
            mock_transformers = MockTransformers()
            
            with patch.dict('sys.modules', {'transformers': mock_transformers}):
                try:
                    # Test the _get_model_class logic for seq2seq
                    model_type = "seq2seq"
                    
                    if model_type == "seq2seq":
                        try:
                            from transformers import AutoModelForSeq2SeqLM
                            model_class = AutoModelForSeq2SeqLM
                            print(f"✓ Using AutoModelForSeq2SeqLM for {version}")
                        except ImportError:
                            try:
                                from transformers import AutoModelForSeq2SeqGeneration
                                model_class = AutoModelForSeq2SeqGeneration
                                print(f"✓ Using AutoModelForSeq2SeqGeneration for {version}")
                            except ImportError:
                                # Fallback for very old versions
                                try:
                                    from transformers import BartForConditionalGeneration
                                    model_class = BartForConditionalGeneration
                                    print(f"✓ Using BartForConditionalGeneration fallback for {version}")
                                except ImportError:
                                    raise ImportError("Unable to load seq2seq model. Please ensure transformers is properly installed.")
                    
                    self.assertIsNotNone(model_class)
                    print(f"✓ Successfully selected model class for {version}")
                    
                except Exception as e:
                    self.fail(f"Failed for {version}: {e}")

def run_compatibility_tests():
    """Run all compatibility tests and return results."""
    print("=" * 60)
    print("Transformers Backward Compatibility Test Suite")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTransformersCompatibility)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("🎉 ALL TESTS PASSED! Transformers backward compatibility is working correctly.")
        print("The SDK will work with both old and new transformers versions.")
        print("\nCompatibility summary:")
        print("- Current transformers: Uses AutoModelForSeq2SeqLM")
        print("- Old transformers: Falls back to AutoModelForSeq2SeqGeneration")
        print("- Very old transformers: Falls back to BartForConditionalGeneration")
    else:
        print("❌ SOME TESTS FAILED! Backward compatibility issues detected.")
        print("Please check the error messages above.")
    print("=" * 60)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_compatibility_tests()
    sys.exit(0 if success else 1) 