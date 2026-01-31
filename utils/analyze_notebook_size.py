#!/usr/bin/env python3
"""
Analyze Jupyter Notebook file size and identify what's taking up space.
"""
import json
import sys
from pathlib import Path

def analyze_notebook_size(notebook_path):
    """Analyze what's taking up space in a Jupyter notebook."""
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    total_size = Path(notebook_path).stat().st_size
    
    # Initialize counters
    cell_stats = []
    total_code_size = 0
    total_markdown_size = 0
    total_output_size = 0
    output_by_type = {}
    
    for i, cell in enumerate(notebook.get('cells', [])):
        cell_type = cell.get('cell_type', 'unknown')
        
        # Calculate source size
        source = cell.get('source', [])
        if isinstance(source, list):
            source_text = ''.join(source)
        else:
            source_text = source
        source_size = len(source_text.encode('utf-8'))
        
        # Calculate output size
        output_size = 0
        outputs = cell.get('outputs', [])
        output_info = []
        
        for output in outputs:
            output_json = json.dumps(output)
            output_size += len(output_json.encode('utf-8'))
            
            # Track output types
            output_type = output.get('output_type', 'unknown')
            
            # Check for different data types
            if 'data' in output:
                for mime_type, data in output['data'].items():
                    if isinstance(data, list):
                        data_size = sum(len(str(item).encode('utf-8')) for item in data)
                    else:
                        data_size = len(str(data).encode('utf-8'))
                    
                    output_info.append({
                        'type': mime_type,
                        'size': data_size
                    })
                    
                    if mime_type not in output_by_type:
                        output_by_type[mime_type] = 0
                    output_by_type[mime_type] += data_size
        
        # Track totals
        if cell_type == 'code':
            total_code_size += source_size
        elif cell_type == 'markdown':
            total_markdown_size += source_size
        
        total_output_size += output_size
        
        # Store cell stats
        cell_stats.append({
            'cell_num': i + 1,
            'type': cell_type,
            'source_size': source_size,
            'output_size': output_size,
            'total_size': source_size + output_size,
            'output_info': output_info
        })
    
    # Sort cells by total size
    cell_stats.sort(key=lambda x: x['total_size'], reverse=True)
    
    # Print summary
    print("=" * 80)
    print(f"NOTEBOOK SIZE ANALYSIS: {notebook_path}")
    print("=" * 80)
    print(f"\nTotal file size: {total_size:,} bytes ({total_size / 1024 / 1024:.2f} MB)")
    print(f"\nBreakdown by component:")
    print(f"  Code cells (source):     {total_code_size:,} bytes ({total_code_size / total_size * 100:.1f}%)")
    print(f"  Markdown cells (source): {total_markdown_size:,} bytes ({total_markdown_size / total_size * 100:.1f}%)")
    print(f"  Cell outputs:            {total_output_size:,} bytes ({total_output_size / total_size * 100:.1f}%)")
    print(f"  Metadata/other:          {total_size - total_code_size - total_markdown_size - total_output_size:,} bytes")
    
    print(f"\n{'=' * 80}")
    print("OUTPUT BREAKDOWN BY MIME TYPE:")
    print("=" * 80)
    sorted_output_types = sorted(output_by_type.items(), key=lambda x: x[1], reverse=True)
    for mime_type, size in sorted_output_types:
        print(f"  {mime_type:50} {size:,} bytes ({size / total_output_size * 100:.1f}% of outputs)")
    
    print(f"\n{'=' * 80}")
    print("TOP 10 LARGEST CELLS:")
    print("=" * 80)
    print(f"{'Cell':>6} {'Type':10} {'Source':>15} {'Outputs':>15} {'Total':>15} {'% of File':>10}")
    print("-" * 80)
    
    for cell in cell_stats[:10]:
        print(f"{cell['cell_num']:>6} {cell['type']:10} "
              f"{cell['source_size']:>15,} {cell['output_size']:>15,} "
              f"{cell['total_size']:>15,} {cell['total_size'] / total_size * 100:>9.1f}%")
        
        # Show output details for cells with large outputs
        if cell['output_size'] > 100000:  # > 100KB
            for out in cell['output_info']:
                print(f"         └─ {out['type']:40} {out['size']:>10,} bytes")
    
    print(f"\n{'=' * 80}")
    print("RECOMMENDATIONS:")
    print("=" * 80)
    
    # Generate recommendations
    recommendations = []
    
    if total_output_size / total_size > 0.7:
        recommendations.append("⚠️  Cell outputs account for >70% of file size - consider clearing outputs")
    
    if 'image/png' in output_by_type and output_by_type['image/png'] > 1000000:
        recommendations.append(f"📊 PNG images: {output_by_type['image/png'] / 1024 / 1024:.2f} MB - consider reducing plot DPI or saving externally")
    
    if 'text/html' in output_by_type and output_by_type['text/html'] > 1000000:
        recommendations.append(f"📄 HTML output: {output_by_type['text/html'] / 1024 / 1024:.2f} MB - consider limiting DataFrame display rows")
    
    if 'application/vnd.jupyter.widget-view+json' in output_by_type:
        recommendations.append(f"🎛️  Interactive widgets stored - these add size even when not displayed")
    
    large_cells = [c for c in cell_stats if c['output_size'] > 1000000]
    if large_cells:
        recommendations.append(f"📦 {len(large_cells)} cell(s) with outputs >1MB - review cells: {', '.join(str(c['cell_num']) for c in large_cells[:5])}")
    
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
    else:
        print("✅ No major issues detected")
    
    print("\n" + "=" * 80)
    print("SUGGESTED ACTIONS:")
    print("=" * 80)
    print("1. Clear all outputs: Kernel → Restart & Clear Output")
    print("2. Clear specific cell outputs: Right-click cell → Clear Cell Outputs")
    print("3. Reduce plot sizes: Use plt.figure(figsize=(smaller_size), dpi=lower_value)")
    print("4. Limit DataFrame displays: Use df.head() instead of displaying full DataFrames")
    print("5. Save large outputs externally: Save plots/data to files instead of in notebook")
    print("6. Use output suppression: Add ';' at end of lines or use display(HTML('...'))")
    print("=" * 80)

if __name__ == '__main__':
    notebook_path = sys.argv[1] if len(sys.argv) > 1 else 'experiment.ipynb'
    analyze_notebook_size(notebook_path)
