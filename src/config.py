import torch

config = {
        'data_root': '.',
        'csv_path': './data/dish.csv',
        'project_name': 'calorie_estimation',
        'use_wandb': False,
        'seed': 42,
        'batch_size': 32,
        'lr': 2e-5,
        'weight_decay': 0.01,
        'epochs': 20,
        'max_text_length': 64,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'best_model_path': './best_model.pth'
    }