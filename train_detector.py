import os
import argparse
import yaml
import logging
import torch
from datetime import datetime

from src.data.data_loader import (
    load_math_dataset,
    load_example_problems,
    create_train_test_split,
    save_to_csv
)
from src.data.data_processor import create_dataloaders
from src.models.detector import Detector
from src.training.trainer import DetectorTrainer
from src.utils.logging import setup_logging

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a detector model to classify AI vs human-generated solutions"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--no-lora",
        action="store_true",
        help="Disable LoRA fine-tuning"
    )
    return parser.parse_args()

def main():
    # parse args
    args = parse_args()
    
    # load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(config['paths']['logs'], f'training_{timestamp}.log')
    setup_logging(config, log_file)
    logger = logging.getLogger(__name__)
    
    # set seed
    torch.manual_seed(config['training']['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['training']['seed'])
    
    # load data
    logger.info("loading datasets...")
    math_problems = load_math_dataset(config['paths']['math_dataset'])
    example_problems = load_example_problems(config['paths']['example_problems'])
    
    # split data
    train_problems, test_problems = create_train_test_split(
        math_problems,
        test_size=config['data']['test_size'],
        random_state=config['data']['random_state']
    )
    
    # save processed
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
    
    # make loaders
    train_dataloader, val_dataloader = create_dataloaders(
        data_path=os.path.join(config['paths']['processed_data'], 'train.csv'),
        tokenizer_name=config['model']['base_model'],
        batch_size=config['training']['batch_size']
    )
    
    # init model
    logger.info("initializing model...")
    detector = Detector(
        model_name=config['model']['base_model'],
        config=config['model']
    )
    
    # add lora if enabled
    if not args.no_lora:
        logger.info("adding lora adapters...")
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
    logger.info("starting training...")
    best_metrics = trainer.train()
    
    # log results
    logger.info("training completed!")
    logger.info(f"best validation metrics: {best_metrics}")

if __name__ == "__main__":
    main() 