from ultralytics import YOLO

# Inisialisasi model dari file konfigurasi arsitektur YAML
model = YOLO("ultralytics/cfg/models/11/yolo11.yaml")

# Memulai proses pelatihan dengan parameter dari paper referensi
model.train(
    data="data.yaml",         # Ganti dengan path ke file dataset YAML kamu
    epochs=60,                # Jumlah epoch
    imgsz=640,                # Resolusi gambar input
    batch=16,                 # Ukuran batch
    optimizer="Adam",         # Optimizer
    lr0=0.001,                # Initial learning rate
    device=0,                 # Gunakan GPU 0
    project="yolo11_project", # Direktori utama penyimpanan hasil
    name="train_baseline"     # Nama sub-direktori run ini
)

print("Pelatihan selesai!")