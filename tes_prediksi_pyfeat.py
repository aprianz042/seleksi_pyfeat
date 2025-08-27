import os
import shutil
from feat import Detector
import pandas as pd

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
    return mapping.get(x, x)

def analisis_emo_pyfeat(img_path):
    try:
        analysis = detector.detect_image(img_path)
        emotions = analysis.emotions
        dominant_emotion = emotions.idxmax(axis=1).iloc[0]
        result = convert_emotion(dominant_emotion)
        
        # Pisahkan dominant emotion dan probabilities untuk masing-masing emosi
        emotion_probabilities = emotions.iloc[0].to_dict()  # Mengubah ke dictionary
        
        # Membuat dictionary dengan key untuk dominant_emotion dan masing-masing emosi
        emotion_dict = {
            'angry': round(emotion_probabilities['anger'], 4),
            'disgust': round(emotion_probabilities['disgust'], 4),
            'fear': round(emotion_probabilities['fear'], 4),
            'happy': round(emotion_probabilities['happiness'], 4),
            'sadness': round(emotion_probabilities['sadness'], 4),
            'surprise': round(emotion_probabilities['surprise'], 4),
            'neutral': round(emotion_probabilities['neutral'], 4),
            'dominant_emotion': result,
        }
        
        return emotion_dict
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None


def merge_with_suffix(dict1, dict2):
    merged = {}
    for k, v in dict1.items():
        if k in dict2:
            merged[f"{k}_before"] = v
        else:
            merged[k] = v
    for k, v in dict2.items():
        if k in dict1:
            merged[f"{k}_after"] = v
        else:
            merged[k] = v
    return merged

def analysis(file_img):
    try:
        #image_path_before = f'FINAL/6_dataset_affectnet_rafdb_seleksi_wajah_lurus_hand_sintesis/{file_img}'
        #image_path_after = f'FINAL/10_dataset_affectnet_rafdb_seleksi_wajah_lurus_hand_sintesis_frontal/{file_img}'
        
        image_path_before = f'FINAL/5_dataset_affectnet_rafdb_seleksi_wajah_miring/{file_img}'
        image_path_after = f'FINAL/11_dataset_affectnet_rafdb_seleksi_wajah_miring_frontal/{file_img}'

        file = {"file" : file_img}
        analysis_before = analisis_emo_pyfeat(image_path_before)
        analysis_after = analisis_emo_pyfeat(image_path_after)
        analysis_merged = merge_with_suffix(analysis_before, analysis_after)
        full_analysis = file | analysis_merged
        return full_analysis
    except Exception as e:
        return None


dataset_path = "FINAL/11_dataset_affectnet_rafdb_seleksi_wajah_miring_frontal"
#dataset_path = "FINAL/10_dataset_affectnet_rafdb_seleksi_wajah_lurus_hand_sintesis_frontal"
label_results = []
max_images_per_label = 5
for label in os.listdir(dataset_path):
    label_path = os.path.join(dataset_path, label)
    if os.path.isdir(label_path):
        print(f"Memproses label: {label}")
        
        image_count = 0 
        for image_name in os.listdir(label_path):
            image_path = os.path.join(label_path, image_name)
            if os.path.isfile(image_path):
                if image_count < max_images_per_label:
                    image_ = os.path.join(label, image_name)
                    result = analysis(image_)
                    if result is not None:
                        #print(result)
                        label_results.append(result)
                    else:
                        print("Gagal proses gambar")
                    image_count += 1
                else:
                    break 
        

df = pd.DataFrame(label_results)

df.to_csv('1emotion_analysis_results.csv', index=False)
print(df)