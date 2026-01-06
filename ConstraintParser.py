import ast
import re
import json
import os
import sys
import numpy as np

class ConstraintParser:
    def __init__(self, filename=None):
        self.filename = filename
        self.constraints_dict = {}

    @staticmethod
    def parse_condition(condition):
        """Parse a single condition string into a list of dictionaries with feature, operator, and value."""
        parts = re.split(r" (<=|>=|<|>|==) ", condition.strip())
        if len(parts) == 3:
            feature, operator, value = parts
            return [{"feature": feature.strip(), "operator": operator, "value": float(value.strip())}]
        elif len(parts) == 5:
            value1, operator1, feature, operator2, value2 = parts
            return [
                {"feature": feature.strip(), "operator": operator1, "value": float(value1.strip())},
                {"feature": feature.strip(), "operator": operator2, "value": float(value2.strip())}
            ]
        else:
            return None

    @staticmethod
    def constraints_v1_to_dict(raw_string):
        stripped_string = raw_string.replace("Class Bounds: ", "").strip()
        parsed_dict = ast.literal_eval(stripped_string)
        nested_dict = {}
        for class_name, conditions in parsed_dict.items():
            nested_conditions = []
            for condition in conditions:
                parsed_conditions = ConstraintParser.parse_condition(condition)
                if parsed_conditions:
                    nested_conditions.extend(parsed_conditions)
            nested_dict[class_name] = nested_conditions
        return nested_dict

    @staticmethod
    def transform_by_feature(nested_dict):
        feature_dict = {}
        for class_name, conditions in nested_dict.items():
            for condition in conditions:
                feature = condition["feature"]
                if feature not in feature_dict:
                    feature_dict[feature] = []
                feature_dict[feature].append({"class": class_name, "operator": condition["operator"], "value": condition["value"]})
        return feature_dict

    @staticmethod
    def get_intervals_by_feature(feature_based_dict):
        feature_intervals = {}
        for feature, conditions in feature_based_dict.items():
            lower_bound = float('-inf')
            upper_bound = float('inf')
            for condition in conditions:
                operator = condition["operator"]
                value = condition["value"]
                if operator == "<":
                    upper_bound = min(upper_bound, value)
                elif operator == "<=":
                    upper_bound = min(upper_bound, value)
                elif operator == ">":
                    lower_bound = max(lower_bound, value)
                elif operator == ">=":
                    lower_bound = max(lower_bound, value)
            feature_intervals[feature] = (lower_bound, upper_bound)
        return feature_intervals

    @staticmethod
    def is_value_valid_for_class(class_name, feature, value, nested_dict):
        conditions = nested_dict.get(class_name, [])
        for condition in conditions:
            if condition["feature"] == feature:
                operator = condition["operator"]
                comparison_value = condition["value"]
                if operator == "<" and not (value < comparison_value):
                    return False
                elif operator == "<=" and not (value <= comparison_value):
                    return False
                elif operator == ">" and not (value > comparison_value):
                    return False
                elif operator == ">=" and not (value >= comparison_value):
                    return False
        return True

    def read_constraints_from_file(self):
        def _convert_operator_list_to_minmax(operator_list):
            # operator_list: [{'feature':'Age','operator':'<=','value':3.4}, ...]
            features_map = {}
            for cond in operator_list:
                f = cond.get('feature')
                op = cond.get('operator')
                val = cond.get('value')
                if f not in features_map:
                    features_map[f] = {'min': None, 'max': None}
                if op in ('>', '>='):
                    if features_map[f]['min'] is None or val > features_map[f]['min']:
                        features_map[f]['min'] = val
                elif op in ('<', '<='):
                    if features_map[f]['max'] is None or val < features_map[f]['max']:
                        features_map[f]['max'] = val
            return [{'feature': f, 'min': mm['min'], 'max': mm['max']} for f, mm in features_map.items()]

        with open(self.filename, 'r') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                class_label, json_string = line.split(":", 1)
                class_label = class_label.strip()
                json_string = json_string.strip().replace("'", '"').replace("None", "null")
                try:
                    parsed = json.loads(json_string)
                    if class_label.lower() == 'class bounds' and isinstance(parsed, dict):
                        # Merge inner class bounds entries into top-level
                        for inner_class_label, conditions in parsed.items():
                            if isinstance(conditions, list) and len(conditions) and isinstance(conditions[0], str):
                                # Old style: list of constraint strings; parse and convert
                                nested = []
                                for cond_str in conditions:
                                    parsed_conditions = ConstraintParser.parse_condition(cond_str)
                                    if parsed_conditions:
                                        nested.extend(parsed_conditions)
                                converted = _convert_operator_list_to_minmax(nested)
                                self.constraints_dict[inner_class_label.strip()] = converted
                            elif isinstance(conditions, list) and len(conditions) and isinstance(conditions[0], dict) and ('operator' in conditions[0]):
                                # List of operator dicts; convert to min/max
                                converted = _convert_operator_list_to_minmax(conditions)
                                self.constraints_dict[inner_class_label.strip()] = converted
                            else:
                                # Already in expected min/max dict format
                                self.constraints_dict[inner_class_label.strip()] = conditions
                    else:
                        # Regular class label line: could contain list of min/max dicts or other
                        if isinstance(parsed, list) and len(parsed) and isinstance(parsed[0], str):
                            # If it's a list of strings, convert using parse_condition
                            nested = []
                            for cond_str in parsed:
                                parsed_conditions = ConstraintParser.parse_condition(cond_str)
                                if parsed_conditions:
                                    nested.extend(parsed_conditions)
                            converted = _convert_operator_list_to_minmax(nested)
                            self.constraints_dict[class_label] = converted
                        elif isinstance(parsed, list) and len(parsed) and isinstance(parsed[0], dict) and ('operator' in parsed[0]):
                            converted = _convert_operator_list_to_minmax(parsed)
                            self.constraints_dict[class_label] = converted
                        else:
                            self.constraints_dict[class_label] = parsed
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON for {class_label}: {e}")
        return self.constraints_dict

    @staticmethod
    def extract_constraints_from_dataset(model, train_features, train_labels, feature_names):
        """
        Extract constraints from the dataset using Decision Predicate Graph (DPG).
        
        Args:
            model: Trained sklearn model
            train_features: Training features array
            train_labels: Training labels array
            feature_names: List of feature names
            
        Returns:
            Dictionary mapping class labels to feature constraints (min/max format)
        """
        # Import DPG modules
        sys.path.insert(0, os.path.abspath('DPG'))
        from dpg.core import DecisionPredicateGraph
        
        # Build DPG from the trained model
        config_path = os.path.join('DPG', 'config.yaml')
        dpg = DecisionPredicateGraph(
            model=model,
            feature_names=feature_names,
            target_names=np.unique(train_labels).astype(str).tolist(),
            config_file=config_path
        )
        dot = dpg.fit(train_features)
        
        # Convert the Digraph object to string
        dot_string = dot.source if hasattr(dot, 'source') else str(dot)
        
        # Extract decision rules from the graph
        # Parse the graph to extract feature bounds per class
        constraints_dict = {}
        
        # Parse DOT graph to extract constraints
        for line in dot_string.split('\n'):
            line = line.strip()
            if '->' in line and '[label=' in line:
                # Extract feature conditions from edges
                try:
                    label_start = line.find('[label="') + 8
                    label_end = line.find('"]', label_start)
                    if label_start > 7 and label_end > label_start:
                        label = line[label_start:label_end]
                        
                        # Parse conditions like "feature <= value" or "feature > value"
                        for condition in label.split('\\n'):
                            condition = condition.strip()
                            if '<=' in condition or '>' in condition or '<' in condition or '>=' in condition:
                                for feat_name in feature_names:
                                    if feat_name in condition:
                                        # Will be processed in next step
                                        pass
                except:
                    pass
            elif 'label=' in line and 'Class' in line:
                # Extract class nodes
                try:
                    label_start = line.find('label="') + 7
                    label_end = line.find('"', label_start)
                    if label_start > 6 and label_end > label_start:
                        label = line[label_start:label_end]
                        if 'Class' in label:
                            class_name = label.replace('Class ', '').strip()
                            if class_name not in constraints_dict:
                                constraints_dict[class_name] = {}
                except:
                    pass
        
        # Initialize constraint structure for each class with feature bounds
        for class_label in np.unique(train_labels):
            class_str = str(class_label)
            if class_str not in constraints_dict:
                constraints_dict[class_str] = {}
            
            # For each feature, find min/max bounds from training data for this class
            class_mask = train_labels == class_label
            class_features = train_features[class_mask]
            
            for idx, feat_name in enumerate(feature_names):
                feat_values = class_features[:, idx]
                constraints_dict[class_str][feat_name] = {
                    'min': float(np.percentile(feat_values, 5)),  # Use 5th percentile as lower bound
                    'max': float(np.percentile(feat_values, 95))  # Use 95th percentile as upper bound
                }
        
        # Convert to expected format: {class: [{'feature': name, 'min': val, 'max': val}, ...]}
        formatted_constraints = {}
        for class_label, feature_bounds in constraints_dict.items():
            formatted_constraints[f"Class {class_label}"] = [
                {'feature': feat_name, 'min': bounds['min'], 'max': bounds['max']}
                for feat_name, bounds in feature_bounds.items()
            ]
        
        return formatted_constraints
