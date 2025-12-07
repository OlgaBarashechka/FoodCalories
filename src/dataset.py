import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from transformers import AlbertTokenizer  # ИЗМЕНЕНО: импортируем AlbertTokenizer
import numpy as np
from scipy import stats


def get_transforms(train=True):
    normalize = A.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )
    
    if train:
        transform = A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(p=0.2),
            normalize
        ])
    else:
        transform = A.Compose([
            A.Resize(224, 224),
            normalize
        ])
    
    return transform


class FoodDataset(Dataset):
    def __init__(self, root_dir, csv_path, split, transform=None, max_length=128, z_threshold=3):
        self.root_dir = root_dir
        self.csv_path = csv_path
        self.split = split
        self.transform = transform
        self.max_length = max_length
        self.z_threshold = z_threshold  # Пороговое значение Z-статистики для выявления выбросов
        
        # Загрузка данных
        self.ingredients_df = pd.read_csv(os.path.join(root_dir, 'data', 'ingredients.csv'))
        self.dish_df = pd.read_csv(csv_path)
        
        # Фильтрация по split
        self.data = self.dish_df[self.dish_df['split'] == self.split].reset_index(drop=True)
        
        # Очистка данных от выбросов в табличных признаках
        self._remove_outliers()
        
        # Токенизатор для текста
        # ИЗМЕНЕНО: используем AlbertTokenizer и модель albert-base-v1
        self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v1')
    
    def _remove_outliers(self):
        """
        Удаляет строки с выбросами в столбцах 'total_mass' и 'total_calories'
        с использованием Z-оценки (стандартных отклонений).
        Сохраняются только строки, где Z-оценка < z_threshold.
        """
        # Вычисляем Z-оценки для числовых столбцов
        z_scores = np.abs(stats.zscore(self.data[['total_mass', 'total_calories']].dropna()))
        
        # Создаём маску: True для строк без выбросов
        non_outlier_mask = (z_scores < self.z_threshold).all(axis=1)
        
        # Фильтруем данные
        self.data = self.data[non_outlier_mask].reset_index(drop=True)
        print(f"После очистки от выбросов осталось {len(self.data)} образцов.")
    
    def _get_ingredients_str(self, ingredients_ids):
        ids = ingredients_ids.split(';')
        names = []
        for iid in ids:
            if iid in self.ingredients_df['id'].values:
                name = self.ingredients_df.loc[self.ingredients_df['id'] == iid, 'ingr'].iloc[0]
                names.append(name)
        return ', '.join(names)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        dish_id = row['dish_id']
        
        # Изображение
        img_path = os.path.join(self.root_dir, 'data', 'images', str(dish_id), 'rgb.png')
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)  # Конвертируем в numpy array для albumentations
        
        # Применяем трансформации albumentations
        if self.transform:
            image = self.transform(image=image)['image']
        
        # Постобработка: конвертация в тензор и перестановка осей
        image = torch.tensor(image).permute(2, 0, 1).float()  # Добавляем .float() для согласованности типов
        
        # Текст (ингредиенты)
        ingr_str = self._get_ingredients_str(row['ingredients'])
        text_inputs = self.tokenizer(
            ingr_str,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        text_ids = text_inputs['input_ids'].squeeze(0)
        attention_mask = text_inputs['attention_mask'].squeeze(0)
        
        # Табличные данные (уже очищены от выбросов на этапе __init__)
        mass = torch.tensor(row['total_mass'], dtype=torch.float32)
        calories = torch.tensor(row['total_calories'], dtype=torch.float32)
        
        return {
            'image': image,
            'text_ids': text_ids,
            'attention_mask': attention_mask,
            'mass': mass,
            'calories': calories
        }
