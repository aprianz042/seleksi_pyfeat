from feat import Detector
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

detector = Detector(au_model="xgb", emotion_model="resmasknet")

test_data_dir = "FINAL/4_dataset_affectnet_rafdb_seleksi_wajah_lurus/angry"
single_face_img_path = os.path.join(test_data_dir, "angry_0026.jpg")

if not os.path.exists(single_face_img_path):
    raise FileNotFoundError(f"File tidak ditemukan: {single_face_img_path}")

try:
    fex = detector.detect_image(single_face_img_path)
except AttributeError:
    fex = detector.detect(single_face_img_path, data_type="image")

# Plot detections → hasil berupa list of figures
figs = fex.plot_detections()

# Simpan semua fig (atau hanya fig[0] kalau cuma 1 wajah)
for i, fig in enumerate(figs):
    out_path = f"face_landmarks_{i}.jpg"
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Selesai simpan: {out_path}")
