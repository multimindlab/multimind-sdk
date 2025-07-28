#!/usr/bin/env python3
"""
Comprehensive test runner for all MultiMind SDK examples.

This script systematically tests all examples in the examples/ directory
and generates a detailed report of their status.
"""

import asyncio
import importlib
import inspect
import os
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import json

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

@dataclass
class ExampleTestResult:
    """Result of testing an example."""
    name: str
    path: str
    status: str  # "passed", "failed", "skipped", "error"
    error_message: Optional[str] = None
    execution_time: Optional[float] = None
    has_main_function: bool = False
    is_async: bool = False
    dependencies: List[str] = None
    test_notes: str = ""

class ExampleTester:
    """Systematic tester for all examples."""
    
    def __init__(self, examples_dir: Path):
        self.examples_dir = examples_dir
        self.results: List[ExampleTestResult] = []
        self.start_time = datetime.now()
    
    def find_python_files(self, directory: Path) -> List[Path]:
        """Find all Python files in a directory recursively."""
        python_files = []
        for item in directory.iterdir():
            if item.is_file() and item.suffix == '.py':
                python_files.append(item)
            elif item.is_dir() and not item.name.startswith('__'):
                python_files.extend(self.find_python_files(item))
        return python_files
    
    def analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a Python file to understand its structure."""
        analysis = {
            "has_main": False,
            "is_async": False,
            "imports": [],
            "functions": [],
            "classes": [],
            "dependencies": []
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for main function
            if "def main(" in content or "async def main(" in content:
                analysis["has_main"] = True
                if "async def main(" in content:
                    analysis["is_async"] = True
            
            # Check for imports
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('import ') or line.startswith('from '):
                    analysis["imports"].append(line)
                
                # Check for common dependencies
                if 'openai' in line:
                    analysis["dependencies"].append("openai")
                if 'anthropic' in line:
                    analysis["dependencies"].append("anthropic")
                if 'transformers' in line:
                    analysis["dependencies"].append("transformers")
                if 'torch' in line:
                    analysis["dependencies"].append("torch")
                if 'numpy' in line:
                    analysis["dependencies"].append("numpy")
                if 'pandas' in line:
                    analysis["dependencies"].append("pandas")
                if 'requests' in line:
                    analysis["dependencies"].append("requests")
                if 'aiohttp' in line:
                    analysis["dependencies"].append("aiohttp")
                if 'fastapi' in line:
                    analysis["dependencies"].append("fastapi")
                if 'uvicorn' in line:
                    analysis["dependencies"].append("uvicorn")
            
            # Remove duplicates
            analysis["dependencies"] = list(set(analysis["dependencies"]))
            
        except Exception as e:
            analysis["error"] = str(e)
        
        return analysis
    
    async def test_example_file(self, file_path: Path) -> ExampleTestResult:
        """Test a single example file."""
        relative_path = file_path.relative_to(self.examples_dir)
        name = str(relative_path).replace('/', '.').replace('\\', '.')
        
        print(f"Testing: {name}")
        
        # Analyze the file
        analysis = self.analyze_file(file_path)
        
        # Skip files that don't have a main function
        if not analysis.get("has_main", False):
            return ExampleTestResult(
                name=name,
                path=str(file_path),
                status="skipped",
                test_notes="No main function found",
                has_main_function=False,
                is_async=analysis.get("is_async", False),
                dependencies=analysis.get("dependencies", [])
            )
        
        # Try to import and test the module
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Create a mock environment for testing
            with self.create_mock_environment():
                # Try to import the module
                clean_name = name.replace('.py', '').replace('/', '.').replace('\\', '.')
                module_name = f"examples.{clean_name}"
                
                # Import the module
                module = importlib.import_module(module_name)
                
                # Check if it has a main function
                if hasattr(module, 'main'):
                    main_func = getattr(module, 'main')
                    
                    # Test if it's async
                    if inspect.iscoroutinefunction(main_func):
                        # Run async main function
                        await main_func()
                    else:
                        # Run sync main function
                        main_func()
                    
                    execution_time = asyncio.get_event_loop().time() - start_time
                    
                    return ExampleTestResult(
                        name=name,
                        path=str(file_path),
                        status="passed",
                        execution_time=execution_time,
                        has_main_function=True,
                        is_async=inspect.iscoroutinefunction(main_func),
                        dependencies=analysis.get("dependencies", []),
                        test_notes="Successfully executed main function"
                    )
                else:
                    return ExampleTestResult(
                        name=name,
                        path=str(file_path),
                        status="skipped",
                        test_notes="Module imported but no main function found",
                        has_main_function=False,
                        is_async=analysis.get("is_async", False),
                        dependencies=analysis.get("dependencies", [])
                    )
        
        except ImportError as e:
            return ExampleTestResult(
                name=name,
                path=str(file_path),
                status="error",
                error_message=f"Import error: {str(e)}",
                has_main_function=analysis.get("has_main", False),
                is_async=analysis.get("is_async", False),
                dependencies=analysis.get("dependencies", []),
                test_notes="Failed to import module"
            )
        
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            return ExampleTestResult(
                name=name,
                path=str(file_path),
                status="failed",
                error_message=str(e),
                execution_time=execution_time,
                has_main_function=analysis.get("has_main", False),
                is_async=analysis.get("is_async", False),
                dependencies=analysis.get("dependencies", []),
                test_notes="Runtime error during execution"
            )
    
    def create_mock_environment(self):
        """Create a context manager that mocks external dependencies."""
        import unittest.mock
        
        # Mock common external services
        mocks = {
            'openai.AsyncOpenAI': unittest.mock.MagicMock(),
            'anthropic.Anthropic': unittest.mock.MagicMock(),
            'requests.get': unittest.mock.MagicMock(return_value=unittest.mock.MagicMock(status_code=200)),
            'requests.post': unittest.mock.MagicMock(return_value=unittest.mock.MagicMock(status_code=200)),
        }
        
        # Create patches
        patches = []
        for target, mock_obj in mocks.items():
            patches.append(unittest.mock.patch(target, mock_obj))
        
        # Return a context manager
        class MockContext:
            def __enter__(self):
                for patch in patches:
                    patch.start()
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                for patch in patches:
                    patch.stop()
        
        return MockContext()
    
    async def test_all_examples(self) -> List[ExampleTestResult]:
        """Test all examples in the examples directory."""
        print(f"Starting comprehensive example testing at {self.start_time}")
        print(f"Examples directory: {self.examples_dir}")
        
        # Find all Python files
        python_files = self.find_python_files(self.examples_dir)
        print(f"Found {len(python_files)} Python files to test")
        
        # Test each file
        for file_path in python_files:
            result = await self.test_example_file(file_path)
            self.results.append(result)
        
        return self.results
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive test report."""
        end_time = datetime.now()
        total_time = (end_time - self.start_time).total_seconds()
        
        # Count results by status
        status_counts = {}
        for result in self.results:
            status_counts[result.status] = status_counts.get(result.status, 0) + 1
        
        # Calculate success rate
        total_tested = len([r for r in self.results if r.status != "skipped"])
        success_rate = (status_counts.get("passed", 0) / total_tested * 100) if total_tested > 0 else 0
        
        # Group by directory
        by_directory = {}
        for result in self.results:
            dir_name = Path(result.path).parent.name
            if dir_name not in by_directory:
                by_directory[dir_name] = []
            by_directory[dir_name].append(result)
        
        # Find common issues
        common_errors = {}
        for result in self.results:
            if result.error_message:
                error_type = type(result.error_message).__name__
                common_errors[error_type] = common_errors.get(error_type, 0) + 1
        
        report = {
            "summary": {
                "total_files": len(self.results),
                "total_tested": total_tested,
                "passed": status_counts.get("passed", 0),
                "failed": status_counts.get("failed", 0),
                "error": status_counts.get("error", 0),
                "skipped": status_counts.get("skipped", 0),
                "success_rate": round(success_rate, 2),
                "total_time_seconds": round(total_time, 2),
                "start_time": self.start_time.isoformat(),
                "end_time": end_time.isoformat()
            },
            "status_counts": status_counts,
            "by_directory": {
                dir_name: {
                    "total": len(results),
                    "passed": len([r for r in results if r.status == "passed"]),
                    "failed": len([r for r in results if r.status == "failed"]),
                    "error": len([r for r in results if r.status == "error"]),
                    "skipped": len([r for r in results if r.status == "skipped"])
                }
                for dir_name, results in by_directory.items()
            },
            "common_errors": common_errors,
            "detailed_results": [
                {
                    "name": r.name,
                    "path": r.path,
                    "status": r.status,
                    "error_message": r.error_message,
                    "execution_time": r.execution_time,
                    "has_main_function": r.has_main_function,
                    "is_async": r.is_async,
                    "dependencies": r.dependencies,
                    "test_notes": r.test_notes
                }
                for r in self.results
            ]
        }
        
        return report
    
    def save_report(self, report: Dict[str, Any], output_file: str = "example_test_report.json"):
        """Save the test report to a JSON file."""
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to {output_file}")
    
    def print_summary(self, report: Dict[str, Any]):
        """Print a summary of the test results."""
        summary = report["summary"]
        
        print("\n" + "="*60)
        print("MULTIMIND SDK EXAMPLES TEST SUMMARY")
        print("="*60)
        print(f"Total files found: {summary['total_files']}")
        print(f"Files tested: {summary['total_tested']}")
        print(f"Passed: {summary['passed']}")
        print(f"Failed: {summary['failed']}")
        print(f"Errors: {summary['error']}")
        print(f"Skipped: {summary['skipped']}")
        print(f"Success rate: {summary['success_rate']}%")
        print(f"Total time: {summary['total_time_seconds']} seconds")
        print("="*60)
        
        # Print by directory
        print("\nResults by directory:")
        for dir_name, stats in report["by_directory"].items():
            print(f"  {dir_name}: {stats['passed']}/{stats['total']} passed")
        
        # Print common errors
        if report["common_errors"]:
            print("\nCommon error types:")
            for error_type, count in report["common_errors"].items():
                print(f"  {error_type}: {count} occurrences")


async def main():
    """Main function to run the comprehensive example testing."""
    examples_dir = project_root / "examples"
    
    if not examples_dir.exists():
        print(f"Error: Examples directory not found at {examples_dir}")
        return 1
    
    # Create tester and run tests
    tester = ExampleTester(examples_dir)
    results = await tester.test_all_examples()
    
    # Generate and save report
    report = tester.generate_report()
    tester.save_report(report)
    tester.print_summary(report)
    
    # Return exit code based on success rate
    success_rate = report["summary"]["success_rate"]
    if success_rate >= 80:
        print("\n✅ Overall test result: GOOD")
        return 0
    elif success_rate >= 60:
        print("\n⚠️  Overall test result: NEEDS IMPROVEMENT")
        return 1
    else:
        print("\n❌ Overall test result: POOR")
        return 2


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 