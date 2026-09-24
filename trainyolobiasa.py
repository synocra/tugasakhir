from ultralytics import YOLO

# 1. Load model YOLOv11s (small)
model = YOLO("yolo11s.pt")

# 2. Mulai Training dengan spesifikasi kamu
results = model.train(
    data="/content/tugasakhir/dataset/data.yaml",  # Sesuaikan dengan lokasi file yaml kamu
    epochs=100,  # Total 100 iterasi/epochs
    imgsz=640,  # Ukuran input 640x640
    batch=8,  # Batch size 8
    momentum=0.94,  # Momentum
    lr0=0.01,  # Initial learning rate
    lrf=0.01,  # Final learning rate (untuk cyclic/decay strategy)
    weight_decay=0.0005,  # Weight decay coefficient
    device=0,  # Menggunakan GPU (CUDA)
    project="tugas_akhir_yolo11",  # Nama project
    name="fruit_ripeness_exp",  # Nama eksperimen
    cos_lr=True,  # Mengaktifkan Cosine/Cyclic LR scheduler
)
