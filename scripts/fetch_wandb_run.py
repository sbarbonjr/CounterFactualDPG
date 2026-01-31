#!/usr/bin/env python3
"""
Fetch WandB run information by run ID.

This script retrieves comprehensive information from a WandB run including:
- Run configuration
- Summary metrics
- History (time-series metrics)
- System metrics
- Artifacts
- Files

Usage:
    python scripts/fetch_wandb_run.py --run-id ptsvoh0c
    python scripts/fetch_wandb_run.py --url https://wandb.ai/entity/project/runs/run_id
    
Example:
    python scripts/fetch_wandb_run.py --run-id ptsvoh0c --entity mllab-ts-universit-di-trieste --project CounterFactualDPG
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

try:
    import wandb
    from wandb.apis.public import Run
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Error: wandb not available. Install with: pip install wandb")
    sys.exit(1)


def parse_wandb_url(url: str) -> Tuple[str, str, str]:
    """
    Parse a WandB URL to extract entity, project, and run_id.
    
    Args:
        url: WandB run URL like https://wandb.ai/entity/project/runs/run_id
        
    Returns:
        Tuple of (entity, project, run_id)
    """
    # Pattern: https://wandb.ai/<entity>/<project>/runs/<run_id>
    pattern = r'https?://wandb\.ai/([^/]+)/([^/]+)/runs/([^/?]+)'
    match = re.match(pattern, url)
    if match:
        return match.group(1), match.group(2), match.group(3)
    raise ValueError(f"Could not parse WandB URL: {url}")


def fetch_run(
    run_id: str,
    entity: Optional[str] = None,
    project: Optional[str] = None
) -> Dict[str, Any]:
    """
    Fetch comprehensive information from a WandB run.
    
    Args:
        run_id: The WandB run ID
        entity: WandB entity (username or team name)
        project: WandB project name
        
    Returns:
        Dictionary containing all run information
    """
    api = wandb.Api()
    
    # Construct the run path
    if entity and project:
        run_path = f"{entity}/{project}/{run_id}"
    else:
        run_path = run_id
    
    print(f"Fetching run: {run_path}")
    run: Run = api.run(run_path)
    
    # Collect all run information
    run_info = {
        'meta': {
            'id': run.id,
            'name': run.name,
            'display_name': run.display_name,
            'state': run.state,
            'url': run.url,
            'path': run.path,
            'entity': run.entity,
            'project': run.project,
            'created_at': run.created_at,
            'updated_at': getattr(run, 'updated_at', None),
            'notes': run.notes,
            'tags': list(run.tags) if run.tags else [],
            'group': run.group,
            'job_type': run.job_type,
        },
        'config': dict(run.config),
        'summary': dict(run.summary),
        'history_keys': [],
        'history': [],
        'system_metrics': {},
        'files': [],
        'artifacts': [],
    }
    
    # Get history (time-series data)
    try:
        history = run.history(pandas=False)
        if history:
            run_info['history'] = list(history)
            # Extract unique keys from history
            all_keys = set()
            for row in history:
                all_keys.update(row.keys())
            run_info['history_keys'] = sorted(list(all_keys))
    except Exception as e:
        print(f"Warning: Could not fetch history: {e}")
    
    # Get system metrics
    try:
        system_metrics = run.history(stream='events', pandas=False)
        if system_metrics:
            run_info['system_metrics'] = list(system_metrics)
    except Exception as e:
        print(f"Warning: Could not fetch system metrics: {e}")
    
    # Get files
    try:
        files = run.files()
        run_info['files'] = [
            {
                'name': f.name,
                'size': f.size,
                'mimetype': getattr(f, 'mimetype', None),
                'url': f.url,
            }
            for f in files
        ]
    except Exception as e:
        print(f"Warning: Could not fetch files: {e}")
    
    # Get artifacts
    try:
        artifacts = run.logged_artifacts()
        run_info['artifacts'] = [
            {
                'name': a.name,
                'type': a.type,
                'version': a.version,
                'size': a.size,
                'created_at': str(a.created_at) if a.created_at else None,
            }
            for a in artifacts
        ]
    except Exception as e:
        print(f"Warning: Could not fetch artifacts: {e}")
    
    return run_info


def print_run_summary(run_info: Dict[str, Any]) -> None:
    """Print a human-readable summary of the run."""
    meta = run_info['meta']
    
    print("\n" + "=" * 80)
    print("WANDB RUN SUMMARY")
    print("=" * 80)
    
    print(f"\n📋 Run Metadata:")
    print(f"   ID:           {meta['id']}")
    print(f"   Name:         {meta['name']}")
    print(f"   Display Name: {meta['display_name']}")
    print(f"   State:        {meta['state']}")
    print(f"   URL:          {meta['url']}")
    print(f"   Entity:       {meta['entity']}")
    print(f"   Project:      {meta['project']}")
    print(f"   Created:      {meta['created_at']}")
    print(f"   Tags:         {', '.join(meta['tags']) if meta['tags'] else 'None'}")
    print(f"   Group:        {meta['group'] or 'None'}")
    
    print(f"\n⚙️  Configuration ({len(run_info['config'])} keys):")
    for key, value in sorted(run_info['config'].items()):
        # Truncate long values
        value_str = str(value)
        if len(value_str) > 60:
            value_str = value_str[:57] + "..."
        print(f"   {key}: {value_str}")
    
    print(f"\n📊 Summary Metrics ({len(run_info['summary'])} keys):")
    for key, value in sorted(run_info['summary'].items()):
        if not key.startswith('_'):  # Skip internal keys
            value_str = str(value)
            if len(value_str) > 60:
                value_str = value_str[:57] + "..."
            print(f"   {key}: {value_str}")
    
    print(f"\n📈 History:")
    print(f"   Total steps:  {len(run_info['history'])}")
    print(f"   Logged keys:  {len(run_info['history_keys'])}")
    if run_info['history_keys']:
        print(f"   Keys:         {', '.join(run_info['history_keys'][:10])}")
        if len(run_info['history_keys']) > 10:
            print(f"                 ... and {len(run_info['history_keys']) - 10} more")
    
    print(f"\n📁 Files ({len(run_info['files'])}):")
    for f in run_info['files'][:5]:
        size_str = f"{f['size'] / 1024:.1f} KB" if f['size'] else "Unknown size"
        print(f"   - {f['name']} ({size_str})")
    if len(run_info['files']) > 5:
        print(f"   ... and {len(run_info['files']) - 5} more files")
    
    print(f"\n📦 Artifacts ({len(run_info['artifacts'])}):")
    for a in run_info['artifacts'][:5]:
        size_str = f"{a['size'] / 1024:.1f} KB" if a['size'] else "Unknown size"
        print(f"   - {a['name']} (v{a['version']}, {a['type']}, {size_str})")
    if len(run_info['artifacts']) > 5:
        print(f"   ... and {len(run_info['artifacts']) - 5} more artifacts")
    
    print("\n" + "=" * 80)


def save_run_info(run_info: Dict[str, Any], output_dir: Path) -> str:
    """
    Save run information to a JSON file.
    
    Args:
        run_info: Dictionary containing run information
        output_dir: Directory to save the file
        
    Returns:
        Path to the saved file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    run_id = run_info['meta']['id']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"run_{run_id}_{timestamp}.json"
    filepath = output_dir / filename
    
    # Custom JSON encoder for datetime and wandb objects
    def json_serializer(obj):
        # Check for datetime first
        if isinstance(obj, datetime):
            return obj.isoformat()
        # Handle wandb Summary objects specifically (they have problematic __getattr__)
        obj_type = type(obj).__name__
        if 'Summary' in obj_type or 'wandb' in type(obj).__module__:
            # Try to get underlying dict
            try:
                if hasattr(type(obj), '_json_dict'):
                    return obj._json_dict
                if hasattr(type(obj), '__dict__'):
                    return dict(obj)
                return str(obj)
            except:
                return str(obj)
        # Handle numpy types
        try:
            import numpy as np
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        except ImportError:
            pass
        # Fallback: convert to string
        return str(obj)
    
    with open(filepath, 'w') as f:
        json.dump(run_info, f, indent=2, default=json_serializer)
    
    print(f"\n💾 Run info saved to: {filepath}")
    return str(filepath)


def main():
    parser = argparse.ArgumentParser(
        description="Fetch WandB run information by run ID",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--run-id',
        type=str,
        help='WandB run ID (e.g., ptsvoh0c)'
    )
    parser.add_argument(
        '--url',
        type=str,
        help='Full WandB run URL (alternative to --run-id)'
    )
    parser.add_argument(
        '--entity',
        type=str,
        default='mllab-ts-universit-di-trieste',
        help='WandB entity (username or team name)'
    )
    parser.add_argument(
        '--project',
        type=str,
        default='CounterFactualDPG',
        help='WandB project name'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save run info JSON (default: temp/wandb_fetcher/runs)'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save run info to file'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress detailed output'
    )
    
    args = parser.parse_args()
    
    # Determine run_id, entity, and project
    if args.url:
        entity, project, run_id = parse_wandb_url(args.url)
    elif args.run_id:
        run_id = args.run_id
        entity = args.entity
        project = args.project
    else:
        parser.error("Either --run-id or --url must be provided")
        return
    
    # Fetch run information
    try:
        run_info = fetch_run(run_id, entity, project)
    except Exception as e:
        print(f"Error fetching run: {e}")
        sys.exit(1)
    
    # Print summary
    if not args.quiet:
        print_run_summary(run_info)
    
    # Save to file
    if not args.no_save:
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            # Default to temp/wandb_runs relative to repo root
            repo_root = Path(__file__).resolve().parent.parent
            output_dir = repo_root / 'temp' / 'wandb_runs'
        
        save_run_info(run_info, output_dir)
    
    return run_info


if __name__ == '__main__':
    main()
