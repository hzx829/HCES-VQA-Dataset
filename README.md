## Hazardous Chemical Enterprise Safety VQA Dataset

---

### Overview

The **Hazardous Chemical Enterprise Safety VQA Dataset** is a comprehensive dataset designed for training and evaluating Visual Question Answering (VQA) models in the context of industrial safety. The dataset focuses on detecting safety violations and ensuring compliance with safety regulations across various high-risk operations.

This repository also includes an **evaluation script** to assess the performance of models on this dataset, providing detailed metrics for accuracy, precision, recall, and F1 score.

---

### Dataset Structure

#### Image Files

- **Path**: `./dataset_v1/images`
- **Naming Convention**: `<category>_file_<file_id>_frame_<frame_id>.png`

  Example: `0of8_file_85_frame_0001.png`

#### Annotation Files

- **Path**: `./dataset_v1/json_merge/`
- **Naming Convention**: `<image_filename>.json`

  Example: `0of8_file_85_frame_0001.png.json`

#### JSON Format

Each JSON file contains two main sections:

1. **`gt` (Ground Truth)**: Actual safety compliance labels.
2. **`llm_label` (Model Predictions)**: Predictions from different models.

Example:

```json
{
  "gt": {
    "wearing_helmets": false,
    "safety_line": true,
    ...
  },
  "llm_label": {
    "gpt-4o": {
      "wearing_helmets": false,
      "safety_line": true,
      ...
    },
    "gemini-flash-1.5": {
      "wearing_helmets": true,
      "safety_line": true,
      ...
    }
  }
}
```

---

### Evaluation Script

The evaluation script computes various performance metrics, including accuracy, precision, recall, and F1 score, for each safety question and provides overall metrics.

#### Key Features

- **Per-Question Metrics**: Detailed scores for each safety-related question.
- **Overall Metrics**: Aggregated performance across all questions.
- **Detectability Metrics**: Evaluates the model's ability to detect questions.

#### Evaluation Metrics

- **Accuracy**: Proportion of correct predictions.
- **Precision**: Proportion of true positives among predicted positives.
- **Recall**: Proportion of true positives among actual positives.
- **F1 Score**: Harmonic mean of precision and recall.

---

### Usage Instructions

#### Dataset

1. Download dataset from Google Drive(Ask me for permission)
```
https://drive.google.com/file/d/1qoiGJzE5opcNASrhlUVxLhtMjrdNUWo4/view?usp=sharing
```
#### Evaluation

1. Run the evaluation script:

```python
from your_script import evaluate

directory = "./dataset_v1/json_merge/"
llm_model = 'gemini-pro-1.5'
output_file = 'metrics_pro.json'
metrics = evaluate_metrics(directory, llm_model)
with open(output_file, 'w') as f:
   json.dump(metrics, f)
```

2. **Arguments**:
   - `directory`: Path to the JSON files.
   - `llm_model`: The model name (`gpt-4o`, `gemini-flash-1.5`, `gemini-pro-1.5`).
   - `output_file`: File to save the evaluation results.

3. **Output Metrics**: Saved in the specified JSON file.

Example output structure:

```json
{
  "wearing_helmets": {
    "accuracy": 0.85,
    "precision": 0.88,
    "recall": 0.92,
    "f1_score": 0.90,
    "sample_count": 100,
    "detectability": {
      "accuracy": 0.95,
      "precision": 0.90,
      "recall": 0.95,
      "f1_score": 0.93,
      "sample_count": 120
    }
  },
  "overall": {
    "accuracy": 0.87,
    "precision": 0.89,
    "recall": 0.91,
    "f1_score": 0.90,
    "total_sample_count": 500,
    "detectability": {
      "accuracy": 0.92,
      "precision": 0.91,
      "recall": 0.93,
      "f1_score": 0.92
    }
  }
}
```

4. **Log Files**: A log file `wrong_<model>.txt` will list JSON files where model predictions were not expected json format.

---

### Question Categories

1. **Common Safety**: General compliance, such as wearing helmets and safety lines.
2. **High Position Work**: Safety for working at heights.
3. **Fire Safety**: Hot work compliance and firefighting readiness.
4. **Confined Space Work**: Supervisor presence and safety rope usage.
5. **Lifting Operations**: Proper lifting procedures and object security.
6. **Excavation and Trenching**: Safe excavation practices.
7. **Road Closure Operations**: Use of cones and barrier tape.
8. **Electrical Work**: Compliance in temporary electrical setups.
9. **Blanking and Blinding**: Proper blind plate operations.

---

### Citation

If you use this dataset or evaluation script, please cite:

```
@Dataset{Hazardous_Chemical_VQA,
  title = {Hazardous Chemical Enterprise Safety VQA Dataset},
  year = {2024},
  author = {Your Organization Name},
  note = {Dataset and evaluation tools for industrial safety VQA tasks.}
}
```
