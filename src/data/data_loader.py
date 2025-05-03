import os
import json
import zipfile
from typing import Dict, List, Tuple, Any

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_json_from_zip(zip_path: str, json_path: str) -> Dict[str, Any]:
    """quick helper to load json from zip"""
    with zipfile.ZipFile(zip_path, 'r') as zf:
        with zf.open(json_path) as f:
            return json.load(f)

def load_math_dataset(zip_path: str) -> List[Dict[str, Any]]:
    """loads all math problems, skips macos files"""
    problems = []
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for file_info in zf.filelist:
            if file_info.filename.endswith('.json') and not file_info.filename.startswith('__MACOSX'):
                problem = load_json_from_zip(zip_path, file_info.filename)
                problem['file_path'] = file_info.filename
                problems.append(problem)
    return problems

def load_example_problems(zip_path: str) -> List[Dict[str, Any]]:
    """loads example problems for few-shot"""
    examples = []
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for file_info in zf.filelist:
            if file_info.filename.endswith('.json') and not file_info.filename.startswith('__MACOSX'):
                example = load_json_from_zip(zip_path, file_info.filename)
                examples.append(example)
    return examples

def create_train_test_split(
    problems: List[Dict[str, Any]], 
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """splits data, stratified by problem type"""
    train_problems, test_problems = train_test_split(
        problems,
        test_size=test_size,
        random_state=random_state,
        stratify=[p['type'] for p in problems]
    )
    return train_problems, test_problems

def save_to_csv(
    problems: List[Dict[str, Any]], 
    output_path: str,
    include_cols: List[str] = ['problem', 'type', 'level', 'solution']
) -> None:
    """dumps problems to csv, keeping only specified cols"""
    df = pd.DataFrame(problems)
    df[include_cols].to_csv(output_path, index=False)

def load_inference_results(npz_path: str) -> Dict[str, np.ndarray]:
    """loads model predictions from npz"""
    return dict(np.load(npz_path))

if __name__ == "__main__":
    # quick test
    import yaml
    
    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # load data
    math_problems = load_math_dataset(config['paths']['math_dataset'])
    example_problems = load_example_problems(config['paths']['example_problems'])
    
    # split and save
    train_problems, test_problems = create_train_test_split(math_problems)
    
    os.makedirs(config['paths']['processed_data'], exist_ok=True)
    save_to_csv(
        train_problems,
        os.path.join(config['paths']['processed_data'], 'train.csv'),
        include_cols=config['data']['include_cols']
    )
    save_to_csv(
        test_problems,
        os.path.join(config['paths']['processed_data'], 'test.csv'),
        include_cols=config['data']['include_cols']
    ) 