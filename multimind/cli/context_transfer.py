"""
Advanced Context Transfer CLI Command

Provides comprehensive command-line interface for transferring conversation context
between different LLM providers across the entire AI ecosystem.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from multimind.context_transfer import ContextTransferManager, AdapterFactory


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def validate_file_path(file_path: str, must_exist: bool = True) -> str:
    """
    Validate file path and return absolute path.
    
    Args:
        file_path: File path to validate
        must_exist: Whether the file must exist
        
    Returns:
        Absolute file path
        
    Raises:
        FileNotFoundError: If file doesn't exist and must_exist is True
        ValueError: If path is invalid
    """
    path = Path(file_path).resolve()
    
    if must_exist and not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    return str(path)


def validate_model_name(model_name: str, supported_models: list) -> str:
    """
    Validate model name against supported models.
    
    Args:
        model_name: Model name to validate
        supported_models: List of supported model names
        
    Returns:
        Validated model name
        
    Raises:
        ValueError: If model is not supported
    """
    model_lower = model_name.lower()
    
    if model_lower not in [m.lower() for m in supported_models]:
        supported = ", ".join(supported_models)
        raise ValueError(f"Model '{model_name}' not supported. Supported models: {supported}")
    
    return model_lower


def list_supported_models() -> None:
    """List all supported models with their capabilities."""
    print("🤖 Supported Models and Capabilities:")
    print("=" * 60)
    
    capabilities = AdapterFactory.list_all_capabilities()
    
    for model_name, caps in capabilities.items():
        print(f"\n📋 {model_name.upper()}")
        print(f"   Context Length: {caps.get('max_context_length', 'Unknown'):,} tokens")
        print(f"   Code Support: {'✅' if caps.get('supports_code') else '❌'}")
        print(f"   Image Support: {'✅' if caps.get('supports_images') else '❌'}")
        print(f"   Tools Support: {'✅' if caps.get('supports_tools') else '❌'}")
        print(f"   Formats: {', '.join(caps.get('supported_formats', []))}")


def show_model_info(model_name: str) -> None:
    """Show detailed information about a specific model."""
    try:
        capabilities = AdapterFactory.get_model_capabilities(model_name)
        print(f"\n📊 Model Information: {model_name.upper()}")
        print("=" * 50)
        
        for key, value in capabilities.items():
            if key == "name":
                continue
            print(f"   {key.replace('_', ' ').title()}: {value}")
            
    except ValueError as e:
        print(f"❌ Error: {e}")


def main(args: Optional[list] = None) -> int:
    """
    Main CLI function for advanced context transfer.
    
    Args:
        args: Command line arguments (if None, uses sys.argv)
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    parser = argparse.ArgumentParser(
        description="Advanced context transfer between different LLM providers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic transfer from ChatGPT to DeepSeek
  multimind context-transfer --from_model chatgpt --to_model deepseek \\
    --input_file conversation.json --output_file deepseek_prompt.txt

  # Advanced transfer with smart extraction and detailed summary
  multimind context-transfer --from_model claude --to_model gemini \\
    --input_file chat.json --output_file gemini_prompt.txt \\
    --last_n 10 --summary_type detailed --smart_extraction \\
    --output_format json --include_code_context

  # Transfer with custom formatting options
  multimind context-transfer --from_model chatgpt --to_model mistral \\
    --input_file conv.json --output_file mistral_prompt.txt \\
    --include_reasoning --include_step_by_step

  # List all supported models
  multimind context-transfer --list_models

  # Show model capabilities
  multimind context-transfer --model_info deepseek
        """
    )
    
    # Main command group
    transfer_group = parser.add_argument_group('Transfer Options')
    
    # Required arguments
    transfer_group.add_argument(
        "--from_model",
        help="Source model name (e.g., chatgpt, claude, deepseek)"
    )
    
    transfer_group.add_argument(
        "--to_model", 
        help="Target model name (e.g., deepseek, claude, gemini)"
    )
    
    transfer_group.add_argument(
        "--input_file",
        help="Path to input file containing conversation history"
    )
    
    transfer_group.add_argument(
        "--output_file",
        help="Path to output file for formatted prompt"
    )
    
    # Optional arguments
    transfer_group.add_argument(
        "--last_n",
        type=int,
        default=5,
        help="Number of recent conversation turns to extract (default: 5)"
    )
    
    transfer_group.add_argument(
        "--no_summary",
        action="store_true",
        help="Skip conversation summarization (use only last message)"
    )
    
    transfer_group.add_argument(
        "--summary_type",
        choices=["concise", "detailed", "structured"],
        default="concise",
        help="Type of summary to generate (default: concise)"
    )
    
    transfer_group.add_argument(
        "--smart_extraction",
        action="store_true",
        help="Use intelligent context extraction based on importance"
    )
    
    transfer_group.add_argument(
        "--output_format",
        choices=["txt", "json", "markdown"],
        default="txt",
        help="Output format for the formatted prompt (default: txt)"
    )
    
    # Advanced formatting options
    advanced_group = parser.add_argument_group('Advanced Formatting Options')
    
    advanced_group.add_argument(
        "--include_code_context",
        action="store_true",
        help="Include code-specific formatting instructions"
    )
    
    advanced_group.add_argument(
        "--include_reasoning",
        action="store_true",
        help="Include reasoning instructions in the prompt"
    )
    
    advanced_group.add_argument(
        "--include_safety",
        action="store_true",
        help="Include safety and ethical considerations"
    )
    
    advanced_group.add_argument(
        "--include_creativity",
        action="store_true",
        help="Include creativity instructions"
    )
    
    advanced_group.add_argument(
        "--include_examples",
        action="store_true",
        help="Include instruction to provide examples"
    )
    
    advanced_group.add_argument(
        "--include_step_by_step",
        action="store_true",
        help="Include step-by-step explanation instructions"
    )
    
    advanced_group.add_argument(
        "--include_multimodal",
        action="store_true",
        help="Include multimodal content handling instructions"
    )
    
    advanced_group.add_argument(
        "--include_web_search",
        action="store_true",
        help="Include web search capabilities instructions"
    )
    
    # Information commands
    info_group = parser.add_argument_group('Information Commands')
    
    info_group.add_argument(
        "--list_models",
        action="store_true",
        help="List all supported models and their capabilities"
    )
    
    info_group.add_argument(
        "--model_info",
        help="Show detailed information about a specific model"
    )
    
    # General options
    general_group = parser.add_argument_group('General Options')
    
    general_group.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    # Parse arguments
    parsed_args = parser.parse_args(args)
    
    # Setup logging
    setup_logging(parsed_args.verbose)
    logger = logging.getLogger(__name__)
    
    # Handle information commands
    if parsed_args.list_models:
        list_supported_models()
        return 0
    
    if parsed_args.model_info:
        show_model_info(parsed_args.model_info)
        return 0
    
    # Validate required arguments for transfer
    if not all([parsed_args.from_model, parsed_args.to_model, parsed_args.input_file, parsed_args.output_file]):
        parser.error("Transfer requires --from_model, --to_model, --input_file, and --output_file")
    
    try:
        # Validate input file
        input_path = validate_file_path(parsed_args.input_file, must_exist=True)
        logger.info(f"Input file: {input_path}")
        
        # Validate output directory exists
        output_path = Path(parsed_args.output_file).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output file: {output_path}")
        
        # Validate model names
        manager = ContextTransferManager()
        supported_models = manager.get_supported_models()
        
        from_model = validate_model_name(parsed_args.from_model, supported_models)
        to_model = validate_model_name(parsed_args.to_model, supported_models)
        
        logger.info(f"Transferring context from {from_model} to {to_model}")
        
        # Build formatting options
        formatting_options = {}
        if parsed_args.include_code_context:
            formatting_options['include_code_context'] = True
        if parsed_args.include_reasoning:
            formatting_options['include_reasoning'] = True
        if parsed_args.include_safety:
            formatting_options['include_safety'] = True
        if parsed_args.include_creativity:
            formatting_options['include_creativity'] = True
        if parsed_args.include_examples:
            formatting_options['include_examples'] = True
        if parsed_args.include_step_by_step:
            formatting_options['include_step_by_step'] = True
        if parsed_args.include_multimodal:
            formatting_options['include_multimodal'] = True
        if parsed_args.include_web_search:
            formatting_options['include_web_search'] = True
        
        # Perform context transfer
        formatted_prompt = manager.transfer_context(
            from_model=from_model,
            to_model=to_model,
            input_file=input_path,
            output_file=str(output_path),
            last_n=parsed_args.last_n,
            include_summary=not parsed_args.no_summary,
            summary_type=parsed_args.summary_type,
            smart_extraction=parsed_args.smart_extraction,
            output_format=parsed_args.output_format,
            **formatting_options
        )
        
        logger.info("Context transfer completed successfully!")
        logger.info(f"Formatted prompt saved to: {output_path}")
        
        # Print preview of the formatted prompt
        preview_lines = formatted_prompt.split('\n')[:10]
        preview = '\n'.join(preview_lines)
        if len(formatted_prompt.split('\n')) > 10:
            preview += '\n...'
        
        print(f"\n📝 Formatted Prompt Preview:\n{'-' * 50}")
        print(preview)
        print(f"{'-' * 50}")
        
        # Show model capabilities
        try:
            from_caps = manager.get_model_info(from_model)
            to_caps = manager.get_model_info(to_model)
            
            print(f"\n📊 Transfer Summary:")
            print(f"   From: {from_model} ({from_caps.get('max_context_length', 'Unknown'):,} tokens)")
            print(f"   To: {to_model} ({to_caps.get('max_context_length', 'Unknown'):,} tokens)")
            print(f"   Format: {parsed_args.output_format.upper()}")
            print(f"   Summary: {parsed_args.summary_type}")
            print(f"   Smart Extraction: {'✅' if parsed_args.smart_extraction else '❌'}")
            
        except Exception as e:
            logger.warning(f"Could not display model capabilities: {e}")
        
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"File error: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if parsed_args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 