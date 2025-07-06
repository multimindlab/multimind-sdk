#!/usr/bin/env python3
"""
Script to prepare the MultiMind SDK pip package for manual publishing.
This script automates the process of building and validating the package before manual upload to PyPI.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(command, check=True, capture_output=False):
    """Run a shell command and return the result."""
    print(f"Running: {command}")
    result = subprocess.run(
        command, 
        shell=True, 
        check=check, 
        capture_output=capture_output,
        text=True
    )
    return result

def clean_build_files():
    """Clean up previous build files."""
    print("🧹 Cleaning up previous build files...")
    
    # Remove build directories
    build_dirs = ["build", "dist", "multimind_sdk.egg-info"]
    for dir_name in build_dirs:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            print(f"  Removed {dir_name}")

def check_dependencies():
    """Check if required tools are installed."""
    print("🔍 Checking dependencies...")
    
    required_tools = ["build"]
    missing_tools = []
    
    for tool in required_tools:
        try:
            if tool == "build":
                # Check build tool using python -m build
                subprocess.run(["python", "-m", "build", "--version"], capture_output=True, check=True)
            else:
                subprocess.run([tool, "--version"], capture_output=True, check=True)
            print(f"  ✅ {tool} is installed")
        except (subprocess.CalledProcessError, FileNotFoundError):
            missing_tools.append(tool)
            print(f"  ❌ {tool} is missing")
    
    if missing_tools:
        print(f"\n📦 Installing missing tools: {', '.join(missing_tools)}")
        run_command(f"pip install {' '.join(missing_tools)}")
    
    return len(missing_tools) == 0

def build_package():
    """Build the package distribution files."""
    print("🔨 Building package...")
    
    # Build the package
    run_command("python -m build")
    
    # Check if build was successful
    if not os.path.exists("dist"):
        raise Exception("Build failed - dist directory not created")
    
    print("  ✅ Package built successfully")

def check_package():
    """Check the built package for issues."""
    print("🔍 Checking package...")
    
    # Check if twine is available for validation
    try:
        subprocess.run(["twine", "--version"], capture_output=True, check=True)
        run_command("twine check dist/*")
        print("  ✅ Package check passed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("  ⚠️  twine not found - skipping package validation")
        print("  💡 Install twine with: pip install twine")

def show_next_steps():
    """Show the next steps for manual publishing."""
    print("\n📋 Next Steps for Manual Publishing:")
    print("=" * 50)
    print("1. 📦 Your package is ready in the 'dist/' directory")
    print("2. 🔍 Review the built files:")
    print("   - dist/multimind_sdk-0.2.1.tar.gz (source distribution)")
    print("   - dist/multimind_sdk-0.2.1-py3-none-any.whl (wheel distribution)")
    print("3. 🧪 Test the package locally:")
    print("   pip install dist/multimind_sdk-0.2.1-py3-none-any.whl")
    print("4. 🚀 Upload to PyPI manually:")
    print("   twine upload dist/*")
    print("5. 🏷️  Create git tag:")
    print("   git tag v0.2.1")
    print("   git push origin v0.2.1")
    print("\n💡 For testing on TestPyPI first:")
    print("   twine upload --repository testpypi dist/*")

def main():
    """Main function to prepare the pip package."""
    print("🚀 MultiMind SDK Package Preparation Script")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("setup.py"):
        print("❌ Error: setup.py not found. Please run this script from the project root.")
        sys.exit(1)
    
    # Get user confirmation
    print("\nThis script will:")
    print("1. Clean previous build files")
    print("2. Check dependencies")
    print("3. Build the package")
    print("4. Validate the package")
    print("5. Show next steps for manual publishing")
    
    response = input("\nDo you want to continue? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("❌ Aborted by user")
        sys.exit(0)
    
    try:
        # Step 1: Clean build files
        clean_build_files()
        
        # Step 2: Check dependencies
        if not check_dependencies():
            print("❌ Failed to install required dependencies")
            sys.exit(1)
        
        # Step 3: Build package
        build_package()
        
        # Step 4: Check package
        check_package()
        
        # Step 5: Show next steps
        show_next_steps()
        
        print("\n🎉 Package preparation completed successfully!")
        print("📦 Ready for manual publishing to PyPI")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 