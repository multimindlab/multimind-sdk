import argparse
import sys
import importlib
import os
from pathlib import Path

EXAMPLES_DIR = Path(__file__).parent.parent.parent / 'examples' / 'evolutionary'

EXAMPLES = {
    'reflexive': 'reflexive_agent_pipeline_demo',
    'hybrid': 'hybrid_evolutionary_reflexive_demo',
}

DESCRIPTION = """
MultiMindSDK Agentic CLI
------------------------
Run agentic, evolutionary, and reflexive workflow demos from the command line.
"""

def list_examples():
    print("Available examples:")
    for key, val in EXAMPLES.items():
        print(f"  {key}: {val}.py")

def run_demo(example: str):
    if example not in EXAMPLES:
        print(f"Unknown example: {example}")
        list_examples()
        sys.exit(1)
    module_name = f"examples.evolutionary.{EXAMPLES[example]}"
    try:
        module = importlib.import_module(module_name)
        if hasattr(module, 'main'):
            module.main()
        elif hasattr(module, 'asyncio') and hasattr(module, 'main'):
            import asyncio
            asyncio.run(module.main())
        else:
            print(f"No main() function found in {module_name}")
    except Exception as e:
        print(f"Error running {module_name}: {e}")
        sys.exit(1)

def info():
    print(DESCRIPTION)
    list_examples()

def main():
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    subparsers = parser.add_subparsers(dest='command')

    parser_list = subparsers.add_parser('list-examples', help='List available agentic workflow examples')
    parser_run = subparsers.add_parser('run-demo', help='Run a specific agentic workflow demo')
    parser_run.add_argument('example', choices=EXAMPLES.keys(), help='Example to run')
    parser_info = subparsers.add_parser('info', help='Show CLI info and available examples')

    args = parser.parse_args()

    if args.command == 'list-examples':
        list_examples()
    elif args.command == 'run-demo':
        run_demo(args.example)
    elif args.command == 'info' or args.command is None:
        info()
    else:
        parser.print_help()

if __name__ == '__main__':
    main() 