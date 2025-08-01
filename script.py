import torch
from carvekit.api.high import HiInterface  # корректный импорт

# Настройка параметров
interface = HiInterface(
    object_type="object",        # "object" для общих объектов, "hairs-like" для волос/людей
    batch_size_seg=1,
    batch_size_matting=1,
    device="cuda" if torch.cuda.is_available() else "cpu",
    seg_mask_size=640,           # 640 для tracer_b7, 320 для u2net/basnet
    matting_mask_size=2048,
    trimap_prob_threshold=231,
    trimap_dilation=30,
    trimap_erosion_iters=5,
    fp16=False
)

# Обработка изображения:
images = ["input.jpg"]  # или путь к своей картинке
results = interface(images)
result = results[0]
result.save("output.png")

print("✅ Фон успешно удалён и сохранён как output.png")
