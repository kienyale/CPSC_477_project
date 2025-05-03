import ast
from typing import Dict, List, Tuple, Any

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer

class MathDataset(Dataset):
    """handles math problems dataset stuff"""
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 256
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = self.labels[idx]

        # tokenize with padding
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def build_aligned_longform_from_MATH(
    input_path: str,
) -> pd.DataFrame:
    """
    builds aligned format with human/ai solutions
    
    returns df with:
    - doc_id: unique problem id
    - prompt: problem text
    - variant: solution type
    - text: solution text 
    - label: 0=ai, 1=human
    """
    # load csv
    df = pd.read_csv(input_path, index_col=False)
    
    # parse composite id
    def parse_id(x):
        if isinstance(x, str):
            tup = tuple(ast.literal_eval(x))
        else:
            tup = tuple(x) if isinstance(x, (list, tuple)) else (x,)
        return tup
    
    df['ID_parsed'] = df['ID'].apply(parse_id)
    df['doc_id'] = df['ID_parsed'].apply(lambda t: t[0])
    
    # rename cols
    df = df.rename(columns={
        'Problem': 'prompt',
        'zeroshot': 'ans_base',
        'fewshot': 'ans_few', 
        'fewshot2': 'ans_hard',
        'ground_truth': 'ans_human'
    })
    
    # drop dupes
    df = df.drop_duplicates(
        subset=['doc_id', 'prompt', 'ans_base', 'ans_few', 'ans_hard', 'ans_human'],
        keep='first'
    )
    
    # group by doc_id + generated answers
    grouped = (
        df
        .groupby(['doc_id', 'prompt', 'ans_base', 'ans_few', 'ans_hard'], dropna=False)
        .agg({'ans_human': list})
        .reset_index()
    )
    
    # build long form
    rows = []
    for _, r in grouped.iterrows():
        doc = r['doc_id']
        prompt = r['prompt']
        
        # ai solutions (label=0)
        rows.append({
            'doc_id': doc,
            'prompt': prompt,
            'variant': 'baseline',
            'text': r['ans_base'],
            'label': 0
        })
        
        rows.append({
            'doc_id': doc,
            'prompt': prompt,
            'variant': 'few_shot',
            'text': r['ans_few'],
            'label': 0
        })

        rows.append({
            'doc_id': doc,
            'prompt': prompt,
            'variant': 'prompt_engineering',
            'text': r['ans_hard'],
            'label': 0
        })
        
        # human solutions (label=1)
        for human_text in r['ans_human']:
            rows.append({
                'doc_id': doc,
                'prompt': prompt,
                'variant': 'human',
                'text': human_text,
                'label': 1
            })
    
    long_df = pd.DataFrame(rows)
    return long_df

def create_dataloaders(
    df: pd.DataFrame,
    tokenizer: PreTrainedTokenizer,
    batch_size: int,
    max_length: int = 256,
    val_size: float = 0.1,
    random_state: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """creates train/val dataloaders with stratification"""
    # split data
    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        random_state=random_state,
        stratify=df['label']
    )
    
    # make datasets
    train_dataset = MathDataset(
        texts=train_df['text'].tolist(),
        labels=train_df['label'].tolist(),
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    val_dataset = MathDataset(
        texts=val_df['text'].tolist(),
        labels=val_df['label'].tolist(),
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    # make loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, val_loader

if __name__ == "__main__":
    # quick test
    import yaml
    from transformers import AutoTokenizer
    
    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # load and process
    df = build_aligned_longform_from_MATH(
        os.path.join(config['paths']['processed_data'], 'train.csv')
    )
    
    # init tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config['model']['base_model'],
        trust_remote_code=config['model']['tokenizer_trust_remote_code']
    )
    
    # make loaders
    train_loader, val_loader = create_dataloaders(
        df=df,
        tokenizer=tokenizer,
        batch_size=config['training']['batch_size'],
        max_length=config['model']['max_length']
    ) 