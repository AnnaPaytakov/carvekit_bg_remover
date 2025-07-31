from carvekit.web.schemas.config import MLConfig
from carvekit.web.predict import init_interface
from PIL import Image
import torch

# Настройка конфигурации
config = MLConfig(
    segmentation_network="u2net",  # Можно "basnet", "tracer_b7"
    preprocessing_method="none",
    post_processing_method="fba",  # или "none"
    device="cuda" if torch.cuda.is_available() else "cpu",
)

# Инициализация модели
interface = init_interface(config)

# Загрузка и обработка изображения
input_image = Image.open("input.jpg")
result = interface([input_image])[0]

# Сохранение результата
result.save("output.png")
