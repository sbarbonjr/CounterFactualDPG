"""Interactive sample selector widget for Jupyter notebooks."""
import numpy as np
import pandas as pd
from ipywidgets import RadioButtons, VBox, Output, Button, Layout
from IPython.display import display, clear_output


def create_sample_selector_widget(
    features,
    labels,
    feature_names,
    target_names,
    cf_model,
    constraints,
    class_colors_list,
    plot_constraints_fn,
    num_samples=30,
    random_seed=42
):
    """
    Create an interactive widget for selecting samples from a dataset.
    
    Args:
        features: Array of feature values for all samples
        labels: Array of labels for all samples
        feature_names: List of feature names
        target_names: List of target class names
        cf_model: CounterFactualModel instance for constraint validation
        constraints: Constraints dictionary
        class_colors_list: List of colors for different classes
        plot_constraints_fn: Function to plot constraints (from CounterFactualVisualizer)
        num_samples: Number of random samples to display (default: 30)
        random_seed: Random seed for reproducibility (default: 42)
        
    Returns:
        tuple: (widget, state_dict) where state_dict contains:
            - 'ORIGINAL_SAMPLE': Selected sample as dict
            - 'SAMPLE_DATAFRAME': Selected sample as DataFrame
            - 'random_samples_df': DataFrame with all random samples
            - 'sample_selector': The radio button widget
            - 'selection_confirmed': Boolean indicating if selection was confirmed
    """
    np.random.seed(random_seed)
    total_samples = len(features)
    
    # Generate random indices
    random_indices = np.random.choice(
        total_samples, 
        size=min(num_samples, total_samples), 
        replace=False
    )
    
    # Create DataFrame with random samples
    random_samples_df = pd.DataFrame(features[random_indices], columns=feature_names)
    random_samples_df['target'] = labels[random_indices]
    random_samples_df['target_name'] = [target_names[label] for label in labels[random_indices]]
    random_samples_df['index'] = random_indices
    
    print(f"Displaying {len(random_samples_df)} random samples from the dataset:")
    print(f"Total dataset size: {total_samples}\n")
    
    # Create radio button options with constraint validation
    radio_options = []
    for idx, row in random_samples_df.iterrows():
        # Convert row to sample dict
        sample_dict = {name: row[name] for name in feature_names}
        
        # Get the target class for this sample
        sample_target_class = int(row['target'])
        
        # Validate constraints for this sample against its own class constraints
        is_valid, penalty = cf_model.validate_constraints(
            sample_dict, sample_dict, sample_target_class
        )
        
        # Create validation indicator
        validation_status = "✓" if is_valid else f"✗ (penalty: {penalty:.2f})"
        
        # Build label with constraint validation
        feature_str = " | ".join([f"{col}: {row[col]:.2f}" for col in feature_names])
        label = (
            f"#{idx} | idx:{int(row['index'])} | {validation_status} | "
            f"{feature_str} | class: {int(row['target'])} ({row['target_name']})"
        )
        radio_options.append((label, idx))
    
    # Create radio buttons widget
    sample_selector = RadioButtons(
        options=radio_options,
        value=random_samples_df.index[0],  # Select first sample by default
        description='Select:',
        layout=Layout(width='100%', height='400px'),
        style={'description_width': 'initial'}
    )
    
    # Output area for displaying selected sample details
    output_area = Output()
    
    # Output area for constraint visualizations
    constraints_viz_area = Output()
    
    # Button to confirm selection
    confirm_button = Button(
        description='Use Selected Sample',
        button_style='success',
        tooltip='Click to update ORIGINAL_SAMPLE with the selected sample',
        layout=Layout(width='200px')
    )
    
    # State dictionary to store selection state
    state = {
        'ORIGINAL_SAMPLE': None,
        'SAMPLE_DATAFRAME': None,
        'random_samples_df': random_samples_df,
        'sample_selector': sample_selector,
        'selection_confirmed': False
    }
    
    def update_sample_and_plots(selected_idx):
        """Update ORIGINAL_SAMPLE and re-render constraint plots"""
        selected_row = random_samples_df.loc[selected_idx]
        
        # Update ORIGINAL_SAMPLE
        state['ORIGINAL_SAMPLE'] = {name: selected_row[name] for name in feature_names}
        state['SAMPLE_DATAFRAME'] = pd.DataFrame([state['ORIGINAL_SAMPLE']])
        
        # Get the sample class
        sample_class = int(selected_row['target'])
        
        # Update output area with sample info
        with output_area:
            clear_output()
            print(f"Selected sample (original index: {int(selected_row['index'])}):")
            print(f"Target class: {sample_class} ({selected_row['target_name']})")
            
            # Validate and show constraint status
            is_valid, penalty = cf_model.validate_constraints(
                state['ORIGINAL_SAMPLE'], state['ORIGINAL_SAMPLE'], sample_class
            )
            print(f"Constraint validation: {'✓ Valid' if is_valid else f'✗ Invalid (penalty: {penalty:.2f})'}")
            
            print("\nCurrent ORIGINAL_SAMPLE values:")
            for key, value in state['ORIGINAL_SAMPLE'].items():
                print(f"  {key}: {value}")
        
        # Update constraint visualizations with sample_class
        with constraints_viz_area:
            clear_output(wait=True)
            display(plot_constraints_fn(
                constraints, overlapping=False, class_colors_list=class_colors_list, 
                sample=state['ORIGINAL_SAMPLE'], sample_class=sample_class
            ))
            display(plot_constraints_fn(
                constraints, overlapping=True, class_colors_list=class_colors_list, 
                sample=state['ORIGINAL_SAMPLE'], sample_class=sample_class
            ))
    
    def on_sample_change(change):
        """Handle radio button selection change"""
        selected_idx = change['new']
        update_sample_and_plots(selected_idx)
    
    def on_confirm_click(b):
        """Handle confirm button click"""
        state['selection_confirmed'] = True
        
        with output_area:
            clear_output()
            print("✓ ORIGINAL_SAMPLE confirmed!")
            selected_idx = sample_selector.value
            selected_row = random_samples_df.loc[selected_idx]
            print(f"\nConfirmed sample (original index: {int(selected_row['index'])}):")
            print(f"Target class: {int(selected_row['target'])} ({selected_row['target_name']})")
            
            # Validate and show constraint status
            sample_target_class = int(selected_row['target'])
            is_valid, penalty = cf_model.validate_constraints(
                state['ORIGINAL_SAMPLE'], state['ORIGINAL_SAMPLE'], sample_target_class
            )
            print(f"Constraint validation: {'✓ Valid' if is_valid else f'✗ Invalid (penalty: {penalty:.2f})'}")
            
            print("\nConfirmed ORIGINAL_SAMPLE values:")
            for key, value in state['ORIGINAL_SAMPLE'].items():
                print(f"  {key}: {value}")
    
    # Attach event handlers
    sample_selector.observe(on_sample_change, names='value')
    confirm_button.on_click(on_confirm_click)
    
    # Create widget container
    widget = VBox([sample_selector, confirm_button, output_area, constraints_viz_area])
    
    # Auto-select first sample for "Run All" functionality
    if not state['selection_confirmed']:
        selected_idx = random_samples_df.index[0]
        update_sample_and_plots(selected_idx)
        print(f"\n→ First sample auto-selected (index: {int(random_samples_df.loc[selected_idx]['index'])}, class: {random_samples_df.loc[selected_idx]['target_name']})")
    
    return widget, state
