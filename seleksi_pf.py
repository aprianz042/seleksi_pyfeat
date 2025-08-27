import os
import shutil
from feat import Detector 

detector = Detector(
    face_model="retinaface",
    landmark_model="mobilefacenet",
    au_model='xgb',
    emotion_model="resmasknet",
    facepose_model="img2pose",
)

def convert_emotion(x):
    mapping = {
        'anger': 'angry',
        'disgust': 'disgust',
        'fear': 'fear',
        'happiness': 'happy',
        'sadness': 'sad',
        'surprise': 'surprise',
        'neutral': 'neutral'
    }
    return mapping.get(x, x)  # jika x tidak ada di mapping, kembalikan x itu sendiri

# Fungsi analisis emosi
def analisis_emo_pyfeat(img_path):
    try:
        analisis = detector.detect_image(img_path)
        emo_ = analisis.emotions
        emosi = emo_.idxmax(axis=1).iloc[0]
        result = convert_emotion(emosi)
        return result
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

dataset_path = "datasetFIX"
label = ["surprise"]
output_path = f"hasil_seleksi_{dataset_path}"
limit = 2000

#for label in os.listdir(dataset_path):
for label in label:
    label_path = os.path.join(dataset_path, label)
    if os.path.isdir(label_path):
        print(f"Memproses label: {label}")
        hit = 0
        list_dir = os.listdir(label_path)
        files_to_process = list_dir[4158:]
        #for image_name in os.listdir(label_path):
        for image_name in files_to_process:
            image_path = os.path.join(label_path, image_name)
            if os.path.isfile(image_path):
                detected_emotion = analisis_emo_pyfeat(image_path)
                if detected_emotion and (detected_emotion == label):
                    selected_label_path = os.path.join(output_path, label)
                    os.makedirs(selected_label_path, exist_ok=True)

                    shutil.copy(image_path, os.path.join(selected_label_path, image_name))
                    print(f"  ✅ {image_name} disimpan ke {selected_label_path}")
                    hit+=1
            if hit > limit:
                break