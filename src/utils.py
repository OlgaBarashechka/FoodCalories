import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from tqdm import tqdm
import wandb
import random
import numpy as np
from src.dataset import FoodDataset, get_transforms
from torch.utils.data import DataLoader

# Импорт ConvNeXt из torchvision
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

# Импорт ALBERT вместо DistilBERT
from transformers import AlbertModel, AlbertTokenizer


class CalorieEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        
        # ConvNeXt-Tiny с предобученными весами ImageNet-1K
        self.visual_encoder = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        # Удаляем классификационную головку
        self.visual_encoder.classifier = nn.Identity()
        # Адаптивный пулинг для фиксированного вектора признаков
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Заменяем DistilBERT на ALBERT
        self.text_encoder = AlbertModel.from_pretrained("albert-base-v2")
        
        # Размеры входных данных для классификатора:
        # - ConvNeXt-Tiny: 768 (выход последнего блока)
        # - ALBERT: 768 (hidden_size для albert-base-v2)
        # - mass: 1
        # Итого: 768 + 768 + 1 = 1537
        self.classifier = nn.Sequential(
            nn.Linear(1537, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
    
    def forward(self, image, text_ids, attention_mask, mass):
        # Обработка изображения через ConvNeXt
        img_features = self.visual_encoder.features(image)  # [B, 768, H, W]
        img_features = self.adaptive_pool(img_features)  # [B, 768, 1, 1]
        img_features = img_features.flatten(1)  # [B, 768]
        
        # Обработка текста через ALBERT
        text_features = self.text_encoder(
            input_ids=text_ids, 
            attention_mask=attention_mask
        ).last_hidden_state[:, 0, :]  # [B, 768] (CLS token)
        
        # Объединение признаков
        combined = torch.cat([img_features, text_features, mass.unsqueeze(1)], dim=1)  # [B, 1537]
        return self.classifier(combined).squeeze(-1)  # [B]

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(config):
    set_seed(config['seed'])
    
    # Инициализация W&B (опционально)
    if config['use_wandb']:
        wandb.init(project=config['project_name'], config=config)
    
    # Датасеты и даталоадеры
    train_dataset = FoodDataset(
        root_dir=config['data_root'],
        csv_path=config['csv_path'],
        split='train',
        transform=get_transforms(train=True),
        max_length=config['max_text_length']
    )
    val_dataset = FoodDataset(
        root_dir=config['data_root'],
        csv_path=config['csv_path'],
        split='test',
        transform=get_transforms(train=False),
        max_length=config['max_text_length']
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=2, pin_memory=True)
    
    # Модель, оптимизатор, scheduler
    model = CalorieEstimator().to(config['device'])
    optimizer = AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=config['epochs'])
    
    criterion = nn.L1Loss()  # MAE
    best_val_mae = float('inf')
    
    # Цикл обучения
    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0.0
        
        print(f"\nEpoch {epoch + 1}/{config['epochs']}")
        
        # Обучение
        with tqdm(train_loader, desc="Training", leave=False) as pbar:
            for batch in pbar:
                optimizer.zero_grad()
                
                # Перенос данных на устройство
                image = batch['image'].to(config['device'], non_blocking=True)
                text_ids = batch['text_ids'].to(config['device'], non_blocking=True)
                attention_mask = batch['attention_mask'].to(config['device'], non_blocking=True)
                mass = batch['mass'].to(config['device'], non_blocking=True)
                calories = batch['calories'].to(config['device'], non_blocking=True)
                
                # Прямой проход
                outputs = model(image, text_ids, attention_mask, mass)
                loss = criterion(outputs, calories)
                
                # Обратный проход
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        train_loss /= len(train_loader)
        
        # Валидация
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            with tqdm(val_loader, desc="Validation", leave=False) as pbar:
                for batch in pbar:
                    image = batch['image'].to(config['device'], non_blocking=True)
                    text_ids = batch['text_ids'].to(config['device'], non_blocking=True)
                    attention_mask = batch['attention_mask'].to(config['device'], non_blocking=True)
                    mass = batch['mass'].to(config['device'], non_blocking=True)
                    calories = batch['calories'].to(config['device'], non_blocking=True)
                    
                    outputs = model(image, text_ids, attention_mask, mass)
                    loss = criterion(outputs, calories)
                    
                    val_loss += loss.item()
                    pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        val_loss /= len(val_loader)
        
        # Логирование результатов
        if config['use_wandb']:
            wandb.log({
                'epoch': epoch + 1,
                'train_mae': train_loss,
                'val_mae': val_loss,
                'learning_rate': optimizer.param_groups[0]['lr']
            })
        
        print(f"  Train MAE: {train_loss:.4f}")
        print(f"  Val   MAE: {val_loss:.4f}")
        
        # Сохранение лучшей модели
        if val_loss < best_val_mae:
            best_val_mae = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                                'epoch': epoch,
                'val_mae': best_val_mae
            }, config['best_model_path'])
            print(f"  → Model saved with MAE: {best_val_mae:.4f}")
        
        # Обновление scheduler
        scheduler.step()
    
    print(f"\nTraining completed!")
    print(f"Best validation MAE: {best_val_mae:.4f}")
    
    if config['use_wandb']:
        wandb.finish()
    
    return best_val_mae

def evaluate(model, val_loader, device):
    """
    Функция для финальной оценки модели на валидационном наборе.
    Возвращает средний MAE и список предсказаний/истинных значений.
    """
    model.eval()
    total_loss = 0.0
    predictions = []
    targets = []
    
    criterion = nn.L1Loss()
    
    with torch.no_grad():
        with tqdm(val_loader, desc="Evaluation", leave=False) as pbar:
            for batch in pbar:
                image = batch['image'].to(device, non_blocking=True)
                text_ids = batch['text_ids'].to(device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(device, non_blocking=True)
                mass = batch['mass'].to(device, non_blocking=True)
                calories = batch['calories'].to(device, non_blocking=True)

                outputs = model(image, text_ids, attention_mask, mass)
                loss = criterion(outputs, calories)

                total_loss += loss.item()
                predictions.extend(outputs.cpu().numpy())
                targets.extend(calories.cpu().numpy())
                
                pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(val_loader)
    return avg_loss, predictions, targets

def load_and_evaluate(config):
    """
    Загружает сохранённую модель и проводит полную оценку.
    """
    # Инициализация модели
    model = CalorieEstimator()
    
    # Загрузка чекпоинта
    checkpoint = torch.load(config['best_model_path'], map_location=config['device'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config['device'])
    
    # Подготовка даталоадера
    val_dataset = FoodDataset(
        root_dir=config['data_root'],
        csv_path=config['csv_path'],
        split='test',
        transform=get_transforms(train=False),
        max_length=config['max_text_length']
    )
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4, pin_memory=True)
    
    # Оценка
    mae, preds, targets = evaluate(model, val_loader, config['device'])
    
    print(f"\nFinal Evaluation Results:")
    print(f"MAE on test set: {mae:.4f}")
    
    # Дополнительно можно вывести статистику по предсказаниям
    preds = np.array(preds)
    targets = np.array(targets)
    
    print(f"Mean predicted calories: {preds.mean():.2f}")
    print(f"Mean true calories: {targets.mean():.2f}")
    print(f"Std predicted calories: {preds.std():.2f}")
    print(f"Std true calories: {targets.std():.2f}")
    
    return mae, preds, targets



