import os
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import Counter

def convert_to_bool_none(json_obj):
    for key, value in json_obj.items():
        if value == 'True' or value == 'true':
            json_obj[key] = True
        elif value == 'False' or value == 'false':
            json_obj[key] = False
        elif value == 'null' or value == 'Null':
            json_obj[key] = None
    return json_obj

def bool_to_binary(value):
    """ Convert boolean to binary, and handle None separately for detectability. """
    if value is True:
        return 1
    elif value is False:
        return 0
    return None


def evaluate_metrics(directory, llm_label='gpt-4o'):
    scores = {}
    visibility_stats = {'detectable': 0, 'undetectable': 0}
    answer_stats = {'true': 0, 'false': 0}
    total_true = []
    total_pred = []
    total_detectable_true = []
    total_detectable_pred = []
    wrong_detect_files = []

    # Walk through the directory and process each JSON file
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)

                # Load JSON data
                with open(file_path, 'r') as f:
                    json_data = json.load(f)

                gt_labels = json_data.get("gt", {})
                llm_labels = convert_to_bool_none(json_data.get("llm_label", {}).get(llm_label, {}))

                # Process each key in the data
                for key in set(gt_labels.keys()).union(llm_labels.keys()):
                    true_val = gt_labels.get(key)
                    pred_val = llm_labels.get(key)
                    detectable_true = 0 if true_val is None else 1
                    detectable_pred = 0 if pred_val is None else 1

                    # Initialize scoring dictionaries
                    if key not in scores:
                        scores[key] = {'true': [], 'pred': [], 'detectable_true': [], 'detectable_pred': []}

                    # Append for values' match metrics
                    if true_val is not None and pred_val is not None:
                        true_binary = bool_to_binary(true_val)
                        pred_binary = bool_to_binary(pred_val)
                        if pred_binary is None:
                            if file not in wrong_detect_files:
                                wrong_detect_files.append(file)
                            continue

                        scores[key]['true'].append(true_binary)
                        scores[key]['pred'].append(pred_binary)
                        total_true.append(true_binary)
                        total_pred.append(pred_binary)
                        answer_stats['true' if true_val else 'false'] += 1

                    # Append for detectability metrics
                    scores[key]['detectable_true'].append(detectable_true)
                    scores[key]['detectable_pred'].append(detectable_pred)
                    total_detectable_true.append(detectable_true)
                    total_detectable_pred.append(detectable_pred)
                    if detectable_true == 1:
                        visibility_stats['detectable'] += 1
                    else:
                        visibility_stats['undetectable'] += 1

    if wrong_detect_files:
        with open("wrong_" + llm_label + ".txt", 'w', encoding='utf-8') as log_file:
            for missing_file in wrong_detect_files:
                log_file.write(missing_file + '\n')
    # Calculate and print the metrics for each key
    results = {}
    value_sample_counts = Counter()
    detectability_sample_counts = Counter()
    for key, data in scores.items():
        value_sample_counts[key] = len(data['true'])
        detectability_sample_counts[key] = len(data['detectable_true'])

        # Scores for values' matches
        if data['true']:
            # print(key, data['true'], data['pred'])
            accuracy = accuracy_score(data['true'], data['pred'])
            precision = precision_score(data['true'], data['pred'], zero_division=0)
            recall = recall_score(data['true'], data['pred'], zero_division=0)
            f1 = f1_score(data['true'], data['pred'], zero_division=0)
        else:
            accuracy = precision = recall = f1 = None

        # Scores for detectability
        detect_accuracy = accuracy_score(data['detectable_true'], data['detectable_pred'])
        detect_precision = precision_score(data['detectable_true'], data['detectable_pred'], zero_division=0)
        detect_recall = recall_score(data['detectable_true'], data['detectable_pred'], zero_division=0)
        detect_f1 = f1_score(data['detectable_true'], data['detectable_pred'], zero_division=0)

        results[key] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'sample_count': value_sample_counts[key],
            'detectability': {
                'accuracy': detect_accuracy,
                'precision': detect_precision,
                'recall': detect_recall,
                'f1_score': detect_f1,
                'sample_count': detectability_sample_counts[key]
            }
        }

    # Calculate overall metrics
    overall_accuracy = accuracy_score(total_true, total_pred)
    overall_precision = precision_score(total_true, total_pred, zero_division=0)
    overall_recall = recall_score(total_true, total_pred, zero_division=0)
    overall_f1 = f1_score(total_true, total_pred, zero_division=0)

    # Calculate overall detectability metrics
    overall_detect_accuracy = accuracy_score(total_detectable_true, total_detectable_pred)
    overall_detect_precision = precision_score(total_detectable_true, total_detectable_pred, zero_division=0)
    overall_detect_recall = recall_score(total_detectable_true, total_detectable_pred, zero_division=0)
    overall_detect_f1 = f1_score(total_detectable_true, total_detectable_pred, zero_division=0)

    # Add overall visibility and answer stats, including detectability metrics
    results['overall'] = {
        'visibility_stats': visibility_stats,
        'answer_stats': answer_stats,
        'accuracy': overall_accuracy,
        'precision': overall_precision,
        'recall': overall_recall,
        'f1_score': overall_f1,
        'total_sample_count': len(total_true),
        'total_detectability_sample_count': len(total_detectable_true),
        'detectability': {
            'accuracy': overall_detect_accuracy,
            'precision': overall_detect_precision,
            'recall': overall_detect_recall,
            'f1_score': overall_detect_f1
        }
    }

    return results

# Specify the directory containing the JSON files
directory = "./dataset_v1/json_merge/"
llm_model = 'gemini-pro-1.5'
output_file = 'metrics_pro.json'
metrics = evaluate_metrics(directory, llm_model)
with open(output_file, 'w') as f:
    json.dump(metrics, f)