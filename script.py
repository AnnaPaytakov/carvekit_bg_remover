import torch
from carvekit.api.high import HiInterface

interface = HiInterface(
    object_type="object",
    batch_size_seg=1,
    batch_size_matting=1,
    device="cuda" if torch.cuda.is_available() else "cpu",
    seg_mask_size=640,           # 640 for tracer_b7 model
    matting_mask_size=2048,
    trimap_prob_threshold=231,
    trimap_dilation=30,
    trimap_erosion_iters=5,
    fp16=False
)

images = ["input.jpg"] 
results = interface(images)
result = results[0]
result.save("output.png")

print("Removed successfully and Saved as output.png")
