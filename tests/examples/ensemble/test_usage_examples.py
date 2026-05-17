"""
Tests for ensemble usage examples.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import os
import sys
import subprocess
from pathlib import Path
import json
import requests

# Add project root to path (works from any directory)
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import the usage examples
try:
    from examples.ensemble.usage_examples import (
        run_cli_examples,
        run_api_examples,
        main
    )
except ImportError as e:
    print("IMPORT ERROR:", e)
    run_cli_examples = None
    run_api_examples = None
    main = None


@pytest.mark.skipif(
    run_cli_examples is None,
    reason="usage_examples module not available"
)
class TestCLIExamples:
    """Test CLI examples functionality."""
    
    @patch('examples.ensemble.usage_examples.subprocess.run')
    @patch('examples.ensemble.usage_examples.Path')
    def test_run_cli_examples_text_generation(self, mock_path, mock_subprocess):
        """Test CLI text generation example."""
        # Mock subprocess to return success
        mock_subprocess.return_value = Mock(returncode=0)
        
        # Mock Path for temp file
        mock_code_file = Mock()
        mock_code_file.write_text = Mock()
        mock_code_file.unlink = Mock()
        mock_path.return_value = mock_code_file
        
        # Run CLI examples
        run_cli_examples()
        
        # Verify subprocess was called for text generation
        assert mock_subprocess.call_count >= 1
        calls = [str(call) for call in mock_subprocess.call_args_list]
        assert any("generate" in str(call) for call in calls)
    
    @patch('examples.ensemble.usage_examples.subprocess.run')
    @patch('examples.ensemble.usage_examples.Path')
    def test_run_cli_examples_code_review(self, mock_path, mock_subprocess):
        """Test CLI code review example."""
        mock_subprocess.return_value = Mock(returncode=0)
        
        mock_code_file = Mock()
        mock_code_file.write_text = Mock()
        mock_code_file.unlink = Mock()
        mock_path.return_value = mock_code_file
        
        run_cli_examples()
        
        # Verify code file was created and deleted
        assert mock_code_file.write_text.called
        assert mock_code_file.unlink.called
    
    @patch('examples.ensemble.usage_examples.subprocess.run')
    @patch('examples.ensemble.usage_examples.Path')
    def test_run_cli_examples_embedding(self, mock_path, mock_subprocess):
        """Test CLI embedding example."""
        mock_subprocess.return_value = Mock(returncode=0)
        
        mock_code_file = Mock()
        mock_code_file.write_text = Mock()
        mock_code_file.unlink = Mock()
        mock_path.return_value = mock_code_file
        
        run_cli_examples()
        
        # Verify subprocess was called for embedding
        calls = [str(call) for call in mock_subprocess.call_args_list]
        assert any("embed" in str(call) for call in calls)


@pytest.mark.skipif(
    run_api_examples is None,
    reason="usage_examples module not available"
)
class TestAPIExamples:
    """Test API examples functionality."""
    
    @pytest.mark.asyncio
    @patch('examples.ensemble.usage_examples.requests.post')
    @patch('examples.ensemble.usage_examples.requests.get')
    @patch('examples.ensemble.usage_examples.subprocess.Popen')
    @patch('examples.ensemble.usage_examples.asyncio.sleep')
    async def test_run_api_examples_server_startup(self, mock_sleep, mock_popen, mock_get, mock_post):
        """Test API server startup and readiness check."""
        # Mock server process
        mock_process = Mock()
        mock_process.poll.return_value = None  # Process is running
        mock_process.stderr = None
        mock_popen.return_value = mock_process
        
        # Mock server readiness check
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {}
        
        # Mock API responses
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "result": "Test response",
            "confidence": 0.9,
            "explanation": "Test explanation",
            "provider_votes": {"openai": 1.0}
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        # Run API examples
        await run_api_examples()
        
        # Verify server process was started
        assert mock_popen.called
        # Verify readiness check was performed
        assert mock_get.called
    
    @pytest.mark.asyncio
    @patch('examples.ensemble.usage_examples.requests.post')
    @patch('examples.ensemble.usage_examples.requests.get')
    @patch('examples.ensemble.usage_examples.subprocess.Popen')
    @patch('examples.ensemble.usage_examples.asyncio.sleep')
    async def test_run_api_examples_text_generation(self, mock_sleep, mock_popen, mock_get, mock_post):
        """Test API text generation example."""
        # Mock server process
        mock_process = Mock()
        mock_process.poll.return_value = None
        mock_process.stderr = None
        mock_popen.return_value = mock_process
        
        # Mock server ready
        mock_get.return_value.status_code = 200
        
        # Mock successful API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "result": "Ensemble learning combines multiple models...",
            "confidence": 0.95,
            "explanation": "Selected result from openai",
            "provider_votes": {"openai": 0.6, "ollama": 0.4}
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        await run_api_examples()
        
        # Verify POST request was made to /generate
        post_calls = [call[0][0] for call in mock_post.call_args_list]
        assert any("/generate" in url for url in post_calls)
    
    @pytest.mark.asyncio
    @patch('examples.ensemble.usage_examples.requests.post')
    @patch('examples.ensemble.usage_examples.requests.get')
    @patch('examples.ensemble.usage_examples.subprocess.Popen')
    @patch('examples.ensemble.usage_examples.asyncio.sleep')
    async def test_run_api_examples_code_review(self, mock_sleep, mock_popen, mock_get, mock_post):
        """Test API code review example."""
        mock_process = Mock()
        mock_process.poll.return_value = None
        mock_process.stderr = None
        mock_popen.return_value = mock_process
        
        mock_get.return_value.status_code = 200
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "review": "Code quality is good. Consider adding error handling.",
            "confidence": 0.85,
            "explanation": "Selected result using confidence cascade",
            "provider_votes": {"openai": 1.0}
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        await run_api_examples()
        
        # Verify POST request was made to /review
        post_calls = [call[0][0] for call in mock_post.call_args_list]
        assert any("/review" in url for url in post_calls)
    
    @pytest.mark.asyncio
    @patch('examples.ensemble.usage_examples.requests.post')
    @patch('examples.ensemble.usage_examples.requests.get')
    @patch('examples.ensemble.usage_examples.subprocess.Popen')
    @patch('examples.ensemble.usage_examples.asyncio.sleep')
    async def test_run_api_examples_embedding(self, mock_sleep, mock_popen, mock_get, mock_post):
        """Test API embedding example."""
        mock_process = Mock()
        mock_process.poll.return_value = None
        mock_process.stderr = None
        mock_popen.return_value = mock_process
        
        mock_get.return_value.status_code = 200
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "embedding": [0.1] * 384,
            "confidence": 0.9,
            "explanation": "Selected result from openai",
            "provider_votes": {"openai": 1.0}
        }
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        await run_api_examples()
        
        # Verify POST request was made to /embed
        post_calls = [call[0][0] for call in mock_post.call_args_list]
        assert any("/embed" in url for url in post_calls)
    
    @pytest.mark.asyncio
    @patch('examples.ensemble.usage_examples.requests.post')
    @patch('examples.ensemble.usage_examples.requests.get')
    @patch('examples.ensemble.usage_examples.subprocess.Popen')
    @patch('examples.ensemble.usage_examples.asyncio.sleep')
    async def test_run_api_examples_error_handling(self, mock_sleep, mock_popen, mock_get, mock_post):
        """Test API error handling."""
        mock_process = Mock()
        mock_process.poll.return_value = None
        mock_process.stderr = None
        mock_popen.return_value = mock_process
        
        mock_get.return_value.status_code = 200
        
        # Mock responses: first succeeds, second fails with HTTPError
        mock_success = Mock()
        mock_success.status_code = 200
        mock_success.json.return_value = {"result": "Success"}
        mock_success.raise_for_status = Mock()
        
        mock_error = Mock()
        mock_error.status_code = 500
        mock_error.json.return_value = {"detail": "Internal server error"}
        mock_error.raise_for_status.side_effect = requests.exceptions.HTTPError("500 Server Error")
        
        # First POST (text generation) succeeds, second (code review) fails
        mock_post.side_effect = [mock_success, mock_error, mock_success]
        
        # Should not raise exception, should handle gracefully
        await run_api_examples()
        
        # Verify error was handled
        assert mock_post.called
    
    @pytest.mark.asyncio
    @patch('examples.ensemble.usage_examples.requests.get')
    @patch('examples.ensemble.usage_examples.subprocess.Popen')
    @patch('examples.ensemble.usage_examples.asyncio.sleep')
    async def test_run_api_examples_server_fails_to_start(self, mock_sleep, mock_popen, mock_get):
        """Test handling when server fails to start."""
        # Mock server process that fails immediately
        mock_process = Mock()
        mock_process.poll.return_value = 1  # Process exited
        mock_process.stderr = Mock()
        mock_process.stderr.read.return_value = b"Server startup error"
        mock_popen.return_value = mock_process
        
        # Should return early without making requests
        await run_api_examples()
        
        # Should not make GET requests if server failed
        assert not mock_get.called or mock_get.call_count == 0
    
    @pytest.mark.asyncio
    @patch('examples.ensemble.usage_examples.requests.post')
    @patch('examples.ensemble.usage_examples.requests.get')
    @patch('examples.ensemble.usage_examples.subprocess.Popen')
    @patch('examples.ensemble.usage_examples.asyncio.sleep')
    async def test_run_api_examples_server_cleanup(self, mock_sleep, mock_popen, mock_get, mock_post):
        """Test server cleanup in finally block."""
        mock_process = Mock()
        mock_process.poll.return_value = None
        mock_process.stderr = None
        mock_process.wait.return_value = None
        mock_popen.return_value = mock_process
        
        mock_get.return_value.status_code = 200
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {}
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        await run_api_examples()
        
        # Verify server was terminated
        assert mock_process.terminate.called or mock_process.kill.called
        assert mock_process.wait.called


@pytest.mark.skipif(
    main is None,
    reason="usage_examples module not available"
)
class TestMainFunction:
    """Test main function integration."""
    
    @pytest.mark.asyncio
    @patch('examples.ensemble.usage_examples.run_api_examples')
    @patch('examples.ensemble.usage_examples.run_cli_examples')
    async def test_main_runs_both_examples(self, mock_cli, mock_api):
        """Test that main runs both CLI and API examples."""
        # Mock API function to return a coroutine
        mock_api.return_value = None
        
        await main()
        
        # Verify both functions were called
        assert mock_cli.called
        assert mock_api.called


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("SKIP_INTEGRATION_TESTS") == "1",
    reason="Integration tests skipped"
)
class TestIntegration:
    """Integration tests (require actual API keys and services)."""
    
    @pytest.mark.requires_api_key
    @pytest.mark.slow
    @pytest.mark.skip(reason="Not yet implemented — placeholder for live CLI integration")
    def test_cli_examples_integration(self):
        """Integration test for CLI examples (requires API keys + implementation)."""
        # Auto-skipped by the ``requires_api_key`` marker when keys aren't
        # set; additionally hard-skipped until the test body actually runs
        # the CLI end-to-end. Remove the hard ``skip`` once implemented.

    @pytest.mark.requires_api_key
    @pytest.mark.slow
    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Not yet implemented — placeholder for live API integration")
    async def test_api_examples_integration(self):
        """Integration test for API examples (requires API keys + implementation)."""
        # See test_cli_examples_integration. Hard-skipped until the test body
        # actually starts a server and makes real requests.
