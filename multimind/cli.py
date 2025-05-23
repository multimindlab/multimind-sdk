import click
import sys
import os
import json

# Add YAML suppor
try:
    import yaml
except ImportError:
    yaml = None

# Example: Import your actual training API
from multimind.fine_tuning.unified_peft import UniPELTTuner, UniPELTMethod
from transformers import AutoTokenizer


def load_config(config_path):
    """Load YAML or JSON config file."""
    ext = os.path.splitext(config_path)[-1].lower()
    with open(config_path, 'r') as f:
        if ext in ['.yaml', '.yml']:
            if not yaml:
                raise RuntimeError("PyYAML is required for YAML config files. Please install with 'pip install pyyaml'.")
            return yaml.safe_load(f)
        elif ext == '.json':
            return json.load(f)
        else:
            raise ValueError("Config file must be .yaml, .yml, or .json")

@click.group()
def cli():
    """MultiMind SDK CLI - Fine-tune, evaluate, and manage models with ease."""
    pass

@cli.command(name="train")
@click.option('--config', '-c', type=click.Path(exists=True), help='Path to training config file (YAML or JSON).')
def train(config):
    """Fine-tune a model using a config file (YAML or JSON)."""
    if not config:
        config = click.prompt('Please provide the path to your training config file', type=click.Path(exists=True))
    try:
        cfg = load_config(config)
    except Exception as e:
        click.echo(f"[train] Error loading config: {e}", err=True)
        sys.exit(1)

    # Example config keys: base_model_name, output_dir, methods, method_configs, training_args
    try:
        methods = [UniPELTMethod[m.upper()] if isinstance(m, str) else m for m in cfg['methods']]
        tuner = UniPELTTuner(
            base_model_name=cfg['base_model_name'],
            output_dir=cfg['output_dir'],
            methods=methods,
            method_configs=cfg.get('method_configs'),
            training_args=cfg.get('training_args'),
            model_type=cfg.get('model_type', 'causal_lm')
        )
        train_dataset = cfg['train_dataset']
        eval_dataset = cfg.get('eval_dataset')
        click.echo(f"[train] Starting training for {cfg['base_model_name']} with methods: {methods}")
        tuner.train(train_dataset=train_dataset, eval_dataset=eval_dataset)
        click.echo(f"[train] Training complete. Model saved to {cfg['output_dir']}")
    except Exception as e:
        click.echo(f"[train] Error during training: {e}", err=True)
        sys.exit(1)

# Add 'finetune' as an alias for 'train'
cli.add_command(train, name="finetune")

@cli.command()
@click.option('--model', '-m', type=click.Path(exists=True), help='Path to fine-tuned model.')
@click.option('--dataset', '-d', type=click.Path(exists=True), help='Path to evaluation dataset.')
def evaluate(model, dataset):
    """Evaluate a model on a dataset."""
    if not model:
        model = click.prompt('Model path', type=click.Path(exists=True))
    if not dataset:
        dataset = click.prompt('Evaluation dataset path', type=click.Path(exists=True))
    try:
        tuner = UniPELTTuner(base_model_name=model, output_dir=os.path.dirname(model), methods=[UniPELTMethod.LORA])
        tuner.load_model(model)
        # Assume dataset is a path to a JSON file with 'text' and 'label' fields
        with open(dataset) as f:
            data = json.load(f)
        eval_dataset = [d['text'] for d in data]
        metrics = tuner.trainer.evaluate(eval_dataset=eval_dataset)
        click.echo(f"[evaluate] Metrics: {metrics}")
    except Exception as e:
        click.echo(f"[evaluate] Error: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.option('--model', '-m', type=click.Path(exists=True), help='Path to fine-tuned model.')
@click.option('--input', '-i', type=str, help='Input text for inference.')
def infer(model, input):
    """Run inference with a fine-tuned model."""
    if not model:
        model = click.prompt('Model path', type=click.Path(exists=True))
    if not input:
        input = click.prompt('Input text')
    try:
        tuner = UniPELTTuner(base_model_name=model, output_dir=os.path.dirname(model), methods=[UniPELTMethod.LORA])
        tuner.load_model(model)
        tokenizer = AutoTokenizer.from_pretrained(model)
        inputs = tokenizer(input, return_tensors="pt").to('cuda' if tuner.model.device.type == 'cuda' else 'cpu')
        outputs = tuner.model.generate(**inputs, max_length=128)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        click.echo(f"[infer] Output: {result}")
    except Exception as e:
        click.echo(f"[infer] Error: {e}", err=True)
        sys.exit(1)

@cli.command('list-models')
@click.option('--output-dir', type=click.Path(), default='./output', help='Directory where models are saved.')
def list_models(output_dir):
    """List available or fine-tuned models."""
    try:
        if not os.path.exists(output_dir):
            click.echo(f"[list-models] No models found in {output_dir}")
            return
        models = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
        if not models:
            click.echo(f"[list-models] No models found in {output_dir}")
        else:
            click.echo("[list-models] Available models:")
            for m in models:
                click.echo(f"- {m}")
    except Exception as e:
        click.echo(f"[list-models] Error: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.option('--model', '-m', type=str, help='Model name to download (e.g., bert-base-uncased).')
def download(model):
    """Download a pretrained or fine-tuned model."""
    if not model:
        model = click.prompt('Model name to download')
    try:
        from transformers import AutoModelForCausalLM
        AutoModelForCausalLM.from_pretrained(model)
        click.echo(f"[download] Downloaded model: {model}")
    except Exception as e:
        click.echo(f"[download] Error: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.option('--model', '-m', type=click.Path(exists=True), help='Path to model to export.')
@click.option('--format', '-f', type=click.Choice(['onnx', 'torchscript'], case_sensitive=False), help='Export format.')
@click.option('--output', '-o', type=click.Path(), help='Output path for exported model.')
def export(model, format, output):
    """Export a model to ONNX or TorchScript format."""
    if not model:
        model = click.prompt('Model path', type=click.Path(exists=True))
    if not format:
        format = click.prompt('Export format (onnx/torchscript)', type=click.Choice(['onnx', 'torchscript']))
    if not output:
        output = click.prompt('Output path', type=click.Path())
    try:
        from transformers import AutoModelForCausalLM
        model_obj = AutoModelForCausalLM.from_pretrained(model)
        if format == 'onnx':
            import torch
            dummy_input = torch.randint(0, 100, (1, 16))
            torch.onnx.export(model_obj, dummy_input, output)
        elif format == 'torchscript':
            import torch
            scripted = torch.jit.script(model_obj)
            scripted.save(output)
        click.echo(f"[export] Exported {model} to {format} at {output}")
    except Exception as e:
        click.echo(f"[export] Error: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.option('--model', '-m', type=click.Path(), help='Path to model to delete.')
def delete(model):
    """Delete a local fine-tuned model."""
    if not model:
        model = click.prompt('Model path to delete', type=click.Path())
    if click.confirm(f'Are you sure you want to delete {model}?'):
        try:
            if os.path.isdir(model):
                import shutil
                shutil.rmtree(model)
            else:
                os.remove(model)
            click.echo(f"[delete] Deleted model: {model}")
        except Exception as e:
            click.echo(f"[delete] Error deleting model: {e}", err=True)
    else:
        click.echo("[delete] Aborted.")

@cli.command()
@click.option('--set', 'set_', nargs=2, type=str, help='Set a config key and value.')
@click.option('--get', 'get_', type=str, help='Get a config value by key.')
def config(set_, get_):
    """View or set global CLI configuration."""
    config_path = os.path.expanduser('~/.multimind_cli_config')
    import json
    if not os.path.exists(config_path):
        with open(config_path, 'w') as f:
            json.dump({}, f)
    with open(config_path, 'r') as f:
        cfg = json.load(f)
    if set_:
        key, value = set_
        cfg[key] = value
        with open(config_path, 'w') as f:
            json.dump(cfg, f)
        click.echo(f"[config] Set {key} = {value}")
    elif get_:
        value = cfg.get(get_)
        click.echo(f"[config] {get_} = {value}")
    else:
        click.echo(f"[config] Current config: {cfg}")

@cli.command()
def info():
    """Show model or environment info."""
    click.echo("[info] MultiMind SDK environment info:")
    try:
        import torch
        click.echo(f"PyTorch version: {torch.__version__}")
        import transformers
        click.echo(f"Transformers version: {transformers.__version__}")
    except Exception:
        pass
    click.echo(f"Python version: {sys.version}")
    click.echo(f"Platform: {sys.platform}")

@cli.command()
@click.argument('shell', required=False, type=click.Choice(['bash', 'zsh', 'fish', 'powershell'], case_sensitive=False))
def completion(shell):
    """Generate shell completion script for your shell."""
    if not shell:
        shell = click.prompt('Shell type (bash/zsh/fish/powershell)', type=click.Choice(['bash', 'zsh', 'fish', 'powershell']))
    click.echo(click.get_current_context().command.get_help(click.get_current_context()))
    click.echo(f"[completion] To enable completion, run: eval \"$(multimind completion {shell})\"")
    # TODO: Actually output the completion script for the shell

if __name__ == '__main__':
    cli()