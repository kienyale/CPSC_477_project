import os
from typing import Dict, Optional, Tuple, List
import logging
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from src.models.detector import Detector
from src.utils.metrics import compute_metrics

logger = logging.getLogger(__name__)

class DetectorTrainer:
    """handles training loop and evaluation"""
    
    def __init__(
        self,
        model: Detector,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        config: Dict,
        output_dir: str
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.output_dir = output_dir
        
        # training params
        self.num_epochs = config['training']['num_epochs']
        self.learning_rate = config['training']['learning_rate']
        self.weight_decay = config['training']['weight_decay']
        self.warmup_steps = config['training']['warmup_steps']
        self.gradient_accumulation_steps = config['training']['gradient_accumulation_steps']
        self.max_grad_norm = config['training']['max_grad_norm']
        
        # init optimizer and scheduler
        self.optimizer = AdamW(
            self.model.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        total_steps = len(self.train_dataloader) * self.num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps
        )
        
        # loss fn
        self.criterion = nn.CrossEntropyLoss()
        
        # tracking
        self.best_val_loss = float('inf')
        self.best_val_metrics = {}
        
    def train_epoch(self) -> Tuple[float, Dict]:
        """runs one training epoch"""
        self.model.model.train()
        total_loss = 0
        all_preds, all_labels = [], []
        
        progress_bar = tqdm(
            self.train_dataloader,
            desc="Training",
            leave=False
        )
        
        for step, batch in enumerate(progress_bar):
            # prep batch
            input_ids = batch['input_ids'].to(self.model.device)
            attention_mask = batch['attention_mask'].to(self.model.device)
            labels = batch['labels'].to(self.model.device)
            
            # forward
            outputs = self.model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss / self.gradient_accumulation_steps
            total_loss += loss.item()
            
            # backward
            loss.backward()
            
            # update
            if (step + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.model.parameters(),
                    self.max_grad_norm
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            # track metrics
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # update progress
            progress_bar.set_postfix({
                'loss': f"{total_loss/(step+1):.4f}"
            })
            
        # compute final metrics
        metrics = compute_metrics(all_preds, all_labels)
        metrics['loss'] = total_loss / len(self.train_dataloader)
        
        return metrics['loss'], metrics
    
    def evaluate(self) -> Tuple[float, Dict]:
        """runs evaluation pass"""
        self.model.model.eval()
        total_loss = 0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Evaluating", leave=False):
                # prep batch
                input_ids = batch['input_ids'].to(self.model.device)
                attention_mask = batch['attention_mask'].to(self.model.device)
                labels = batch['labels'].to(self.model.device)
                
                # forward
                outputs = self.model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += outputs.loss.item()
                
                # track metrics
                preds = torch.argmax(outputs.logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # compute final metrics
        metrics = compute_metrics(all_preds, all_labels)
        metrics['loss'] = total_loss / len(self.val_dataloader)
        
        return metrics['loss'], metrics
    
    def train(self) -> Dict:
        """runs full training loop"""
        logger.info("Starting training...")
        
        for epoch in range(self.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.num_epochs}")
            
            # train
            train_loss, train_metrics = self.train_epoch()
            logger.info(f"Training metrics: {train_metrics}")
            
            # eval
            val_loss, val_metrics = self.evaluate()
            logger.info(f"Validation metrics: {val_metrics}")
            
            # save best
            if val_loss < self.best_val_loss:
                logger.info("New best model! Saving...")
                self.best_val_loss = val_loss
                self.best_val_metrics = val_metrics
                
                # save
                os.makedirs(self.output_dir, exist_ok=True)
                self.model.save(self.output_dir)
        
        logger.info("Training completed!")
        logger.info(f"Best validation metrics: {self.best_val_metrics}")
        
        return self.best_val_metrics

if __name__ == "__main__":
    # quick test
    import yaml
    from src.data.data_processor import create_dataloaders
    
    # load config
    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # make loaders
    train_dataloader, val_dataloader = create_dataloaders(
        data_path=os.path.join(config['paths']['processed_data'], 'train.csv'),
        tokenizer_name=config['model']['base_model'],
        batch_size=config['training']['batch_size']
    )
    
    # init model
    detector = Detector(
        model_name=config['model']['base_model'],
        config=config['model']
    )
    
    # add lora
    detector.add_lora(config['lora'])
    
    # init trainer
    trainer = DetectorTrainer(
        model=detector,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=config,
        output_dir=config['paths']['detector_sft']
    )
    
    # train
    best_metrics = trainer.train() 