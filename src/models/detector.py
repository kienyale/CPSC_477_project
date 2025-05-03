import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    PreTrainedModel,
    PreTrainedTokenizer
)
from peft import LoraConfig, get_peft_model, PeftModel

class Detector:
    """handles ai vs human classification"""
    
    def __init__(
        self,
        model_name: str,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        device: Optional[torch.device] = None,
        config: Optional[Dict] = None
    ):
        self.model_name = model_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config or {}
        
        # init tokenizer
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=self.config.get('tokenizer_trust_remote_code', True)
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # init config
        self.model_config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=True,
            num_labels=2,
            pad_token_id=self.tokenizer.pad_token_id
        )
        
        # init base model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=self.model_config,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            device_map=None,
            low_cpu_mem_usage=False
        ).to(self.device)
        
    def add_lora(self, lora_config: Dict) -> None:
        """adds lora adapters for efficient tuning"""
        config = LoraConfig(
            r=lora_config.get('r', 8),
            lora_alpha=lora_config.get('lora_alpha', 32),
            target_modules=lora_config.get('target_modules', ["q_proj", "k_proj", "v_proj"]),
            lora_dropout=lora_config.get('lora_dropout', 0.1),
            bias=lora_config.get('bias', "none")
        )
        self.model = get_peft_model(self.model, config)
        
    def save(self, output_dir: str) -> None:
        """saves model and tokenizer"""
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        base_model: Optional[str] = None,
        device: Optional[torch.device] = None,
        config: Optional[Dict] = None
    ) -> 'Detector':
        """loads a saved detector"""
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # init tokenizer from base
        tokenizer = AutoTokenizer.from_pretrained(
            base_model or model_path,
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # make instance
        detector = cls(
            model_name=base_model or model_path,
            tokenizer=tokenizer,
            device=device,
            config=config
        )
        
        # load weights
        if base_model:
            # load as peft model
            detector.model = PeftModel.from_pretrained(
                detector.model,
                model_path,
                torch_dtype=torch.float32,
                device_map=None
            )
        else:
            # load full weights
            detector.model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                config=detector.model_config,
                trust_remote_code=True,
                torch_dtype=torch.float32,
                device_map=None
            )
            
        detector.model.to(device)
        return detector
    
    def predict(
        self,
        texts: list,
        batch_size: int = 16,
        max_length: int = 256
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        runs inference
        
        returns:
        - predictions (0/1)
        - probabilities
        - logits
        """
        self.model.eval()
        all_preds, all_probs, all_logits = [], [], []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoding = self.tokenizer(
                batch,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=max_length
            ).to(self.device)
            
            with torch.no_grad():
                logits = self.model(**encoding).logits
                probs = torch.softmax(logits, dim=-1)[:,1]
                preds = (probs >= 0.5).long()
                
            all_preds.append(preds.cpu())
            all_probs.append(probs.cpu())
            all_logits.append(logits.cpu())
            
            # cleanup
            del encoding, logits, probs, preds
            torch.cuda.empty_cache()
            
        return (
            torch.cat(all_preds, dim=0),
            torch.cat(all_probs, dim=0),
            torch.cat(all_logits, dim=0)
        )

if __name__ == "__main__":
    # quick test
    import yaml
    
    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # init model
    detector = Detector(
        model_name=config['model']['base_model'],
        config=config['model']
    )
    
    # add lora
    detector.add_lora(config['lora'])
    
    # save
    detector.save(config['paths']['detector_sft'])
    
    # load
    detector = Detector.from_pretrained(
        model_path=config['paths']['detector_sft'],
        base_model=config['model']['base_model'],
        config=config['model']
    ) 