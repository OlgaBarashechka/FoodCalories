from utils import train
from config import config

if __name__ == '__main__':
   # Обучение модели
    best_mae = train(config)
    
    # Финальная оценка
    final_mae, predictions, targets = load_and_evaluate(config)

    print(f"Final best MAE on validation set: {best_mae:.4f}")