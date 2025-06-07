#!/usr/bin/env python3
import os
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any
import requests
from qwen_to_ollama import download_qwen_model, convert_to_ollama

class QwenConversionTester:
    """Test suite for Qwen model conversion."""
    
    def __init__(self, model_path: str, ollama_host: str = "http://localhost:11434"):
        self.model_path = model_path
        self.ollama_host = ollama_host
        self.test_results = []
    
    def run_performance_test(self, prompt: str, num_tokens: int = 100) -> Dict[str, Any]:
        """Test model performance with a given prompt."""
        try:
            start_time = time.time()
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": os.path.basename(self.model_path),
                    "prompt": prompt,
                    "max_tokens": num_tokens,
                    "stream": False
                }
            )
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                tokens_per_second = num_tokens / (end_time - start_time)
                return {
                    "status": "success",
                    "prompt": prompt,
                    "response": result["response"],
                    "generation_time": end_time - start_time,
                    "tokens_per_second": tokens_per_second,
                    "total_tokens": result.get("total_tokens", 0)
                }
            else:
                return {
                    "status": "error",
                    "prompt": prompt,
                    "error": response.text
                }
        except Exception as e:
            return {
                "status": "error",
                "prompt": prompt,
                "error": str(e)
            }
    
    def run_accuracy_test(self, test_cases: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Test model accuracy with a set of test cases."""
        results = []
        for test_case in test_cases:
            try:
                response = requests.post(
                    f"{self.ollama_host}/api/generate",
                    json={
                        "model": os.path.basename(self.model_path),
                        "prompt": test_case["prompt"],
                        "stream": False
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    results.append({
                        "test_case": test_case["name"],
                        "prompt": test_case["prompt"],
                        "expected_keywords": test_case["expected_keywords"],
                        "response": result["response"],
                        "contains_keywords": all(
                            keyword.lower() in result["response"].lower()
                            for keyword in test_case["expected_keywords"]
                        )
                    })
                else:
                    results.append({
                        "test_case": test_case["name"],
                        "status": "error",
                        "error": response.text
                    })
            except Exception as e:
                results.append({
                    "test_case": test_case["name"],
                    "status": "error",
                    "error": str(e)
                })
        return results
    
    def run_memory_test(self) -> Dict[str, Any]:
        """Test model memory usage."""
        try:
            response = requests.get(f"{self.ollama_host}/api/tags")
            if response.status_code == 200:
                models = response.json()
                model_info = next(
                    (m for m in models["models"] if m["name"] == os.path.basename(self.model_path)),
                    None
                )
                if model_info:
                    return {
                        "status": "success",
                        "model_size": model_info.get("size", 0),
                        "format": model_info.get("format", "unknown"),
                        "quantization": model_info.get("quantization", "unknown")
                    }
            return {
                "status": "error",
                "error": "Failed to get model information"
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all test cases and collect results."""
        # Performance test cases
        performance_prompts = [
            "Explain quantum computing in simple terms.",
            "Write a short poem about artificial intelligence.",
            "Summarize the key points of climate change.",
            "Explain the concept of blockchain technology.",
            "Describe the process of photosynthesis."
        ]
        
        # Accuracy test cases
        accuracy_test_cases = [
            {
                "name": "mathematical_reasoning",
                "prompt": "Solve the following math problem: If a train travels at 60 mph for 2.5 hours, how far does it go?",
                "expected_keywords": ["150", "miles", "distance", "calculation"]
            },
            {
                "name": "code_generation",
                "prompt": "Write a Python function to calculate the Fibonacci sequence.",
                "expected_keywords": ["def", "fibonacci", "return", "recursive"]
            },
            {
                "name": "language_translation",
                "prompt": "Translate 'Hello, how are you?' to Spanish.",
                "expected_keywords": ["hola", "como", "estas"]
            },
            {
                "name": "factual_knowledge",
                "prompt": "What is the capital of France?",
                "expected_keywords": ["paris", "france", "capital"]
            },
            {
                "name": "creative_writing",
                "prompt": "Write a short story about a robot learning to paint.",
                "expected_keywords": ["robot", "paint", "art", "learn"]
            }
        ]
        
        # Run tests
        results = {
            "performance_tests": [],
            "accuracy_tests": [],
            "memory_test": None,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Run performance tests
        print("Running performance tests...")
        for prompt in performance_prompts:
            result = self.run_performance_test(prompt)
            results["performance_tests"].append(result)
            print(f"Completed performance test: {prompt[:50]}...")
        
        # Run accuracy tests
        print("\nRunning accuracy tests...")
        accuracy_results = self.run_accuracy_test(accuracy_test_cases)
        results["accuracy_tests"] = accuracy_results
        print("Completed accuracy tests")
        
        # Run memory test
        print("\nRunning memory test...")
        results["memory_test"] = self.run_memory_test()
        print("Completed memory test")
        
        return results
    
    def save_results(self, results: Dict[str, Any], output_file: str):
        """Save test results to a JSON file."""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nTest results saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Test Qwen model conversion")
    parser.add_argument("--model-path", type=str, required=True,
                      help="Path to the converted model")
    parser.add_argument("--output-file", type=str, default="test_results.json",
                      help="Output file for test results")
    parser.add_argument("--ollama-host", type=str, default="http://localhost:11434",
                      help="Ollama API host")
    
    args = parser.parse_args()
    
    try:
        # Initialize tester
        tester = QwenConversionTester(args.model_path, args.ollama_host)
        
        # Run all tests
        print("Starting comprehensive test suite...")
        results = tester.run_all_tests()
        
        # Save results
        tester.save_results(results, args.output_file)
        
        # Print summary
        print("\nTest Summary:")
        print(f"Total performance tests: {len(results['performance_tests'])}")
        print(f"Total accuracy tests: {len(results['accuracy_tests'])}")
        print(f"Average tokens per second: {sum(t['tokens_per_second'] for t in results['performance_tests'] if t['status'] == 'success') / len(results['performance_tests']):.2f}")
        print(f"Accuracy test pass rate: {sum(1 for t in results['accuracy_tests'] if t.get('contains_keywords', False)) / len(results['accuracy_tests']) * 100:.2f}%")
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 