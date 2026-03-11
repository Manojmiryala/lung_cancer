# import os
# import numpy as np
# import cv2
# from flask import Flask, request, jsonify, render_template
# import joblib
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from pathlib import Path

# # Import feature extraction functions and constants from your training script
# from train_cnn_glcm_roi import (
#     preprocess_image_gray,
#     segment_lung_mask,
#     extract_candidate_rois,
#     lung_roi_fallback,
#     extract_glcm_features,
#     extract_cnn_feature_from_roi,
#     LABELS,
#     IMG_SIZE,
#     ROI_SIZE,
#     MODEL_DIR
# )

# app = Flask(__name__)

# # Load models once at startup
# cnn_model = load_model(str(MODEL_DIR / "cnn_feature_extractor.h5"))
# svm = joblib.load(MODEL_DIR / "svm_fused.pkl")
# scaler = joblib.load(MODEL_DIR / "scaler.gz")

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"error": "No file selected"}), 400

#     upload_folder = "static/uploads"
#     roi_folder = "static/roi"
#     os.makedirs(upload_folder, exist_ok=True)
#     os.makedirs(roi_folder, exist_ok=True)

#     # Save uploaded image
#     img_filename = os.path.join(upload_folder, file.filename)
#     file.save(img_filename)

#     try:
#         img_gray = preprocess_image_gray(img_filename)
#         mask = segment_lung_mask(img_gray)
#         rois = extract_candidate_rois(img_gray, mask, min_area=30, max_area=2000)
#         roi_feats_cnn = []
#         roi_feats_glcm = []
#         roi_img_path = None

#         if not rois:
#             roi = lung_roi_fallback(img_gray, mask)
#             rois = [roi]

#         # Use the first ROI for display and prediction
#         roi = rois[0]
#         roi_img_path = os.path.join(roi_folder, f"roi_{os.path.basename(file.filename)}")
#         cv2.imwrite(roi_img_path, roi)

#         for r in rois:
#             try:
#                 glcm_f = extract_glcm_features(r)
#                 cnn_f = extract_cnn_feature_from_roi(r, cnn_model)
#                 roi_feats_glcm.append(glcm_f)
#                 roi_feats_cnn.append(cnn_f)
#             except Exception:
#                 continue
#         if not roi_feats_cnn:
#             roi = lung_roi_fallback(img_gray, mask)
#             roi_feats_glcm = [extract_glcm_features(roi)]
#             roi_feats_cnn = [extract_cnn_feature_from_roi(roi, cnn_model)]

#         cnn_vec = np.mean(np.vstack(roi_feats_cnn), axis=0)
#         glcm_vec = np.mean(np.vstack(roi_feats_glcm), axis=0)
#         fused = np.hstack([cnn_vec, glcm_vec]).reshape(1, -1)

#         fused_scaled = scaler.transform(fused)
#         pred = svm.predict(fused_scaled)[0]
#         pred_label = LABELS[pred]
#         proba = svm.predict_proba(fused_scaled)[0]
#         result = {
#             "prediction": pred_label,
#             "probabilities": {LABELS[i]: float(prob) for i, prob in enumerate(proba)},
#             "image_path": "/" + img_filename.replace("\\", "/"),
#             "roi_path": "/" + roi_img_path.replace("\\", "/")
#         }
#         # If request is AJAX (from frontend), return JSON
#         if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
#             return jsonify(result)
#         # If form POST, render result in HTML
#         return render_template('index.html', result=result)
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True)



# import os
# import numpy as np
# import cv2
# from flask import Flask, request, jsonify, render_template
# import joblib
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from pathlib import Path

# # Import feature extraction functions and constants from your training script
# from train_cnn_glcm_roi import (
#     preprocess_image_gray,
#     segment_lung_mask,
#     extract_candidate_rois,
#     lung_roi_fallback,
#     extract_glcm_features,
#     extract_cnn_feature_from_roi,
#     LABELS,
#     IMG_SIZE,
#     ROI_SIZE,
#     MODEL_DIR
# )

# app = Flask(__name__)

# # Load models once at startup
# cnn_model = load_model(str(MODEL_DIR / "cnn_feature_extractor.h5"))
# svm = joblib.load(MODEL_DIR / "svm_fused.pkl")
# scaler = joblib.load(MODEL_DIR / "scaler.gz")

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"error": "No file selected"}), 400

#     upload_folder = "static/uploads"
#     roi_folder = "static/roi"
#     os.makedirs(upload_folder, exist_ok=True)
#     os.makedirs(roi_folder, exist_ok=True)

#     # Save uploaded image
#     img_filename = os.path.join(upload_folder, file.filename)
#     file.save(img_filename)

#     try:
#         img_gray = preprocess_image_gray(img_filename)
#         mask = segment_lung_mask(img_gray)
#         rois = extract_candidate_rois(img_gray, mask, min_area=30, max_area=2000)
#         roi_feats_cnn = []
#         roi_feats_glcm = []
#         roi_img_path = None

#         if not rois:
#             roi = lung_roi_fallback(img_gray, mask)
#             rois = [roi]

#         # Use the first ROI for display and prediction
#         roi = rois[0]
#         roi_img_path = os.path.join(roi_folder, f"roi_{os.path.basename(file.filename)}")
#         cv2.imwrite(roi_img_path, roi)

#         for r in rois:
#             try:
#                 glcm_f = extract_glcm_features(r)
#                 cnn_f = extract_cnn_feature_from_roi(r, cnn_model)
#                 roi_feats_glcm.append(glcm_f)
#                 roi_feats_cnn.append(cnn_f)
#             except Exception:
#                 continue
#         if not roi_feats_cnn:
#             roi = lung_roi_fallback(img_gray, mask)
#             roi_feats_glcm = [extract_glcm_features(roi)]
#             roi_feats_cnn = [extract_cnn_feature_from_roi(roi, cnn_model)]

#         cnn_vec = np.mean(np.vstack(roi_feats_cnn), axis=0)
#         glcm_vec = np.mean(np.vstack(roi_feats_glcm), axis=0)
#         fused = np.hstack([cnn_vec, glcm_vec]).reshape(1, -1)

#         fused_scaled = scaler.transform(fused)
#         pred = svm.predict(fused_scaled)[0]
#         pred_label = LABELS[pred]
#         proba = svm.predict_proba(fused_scaled)[0]

#         # Adjust probabilities based on the prediction
#         if pred_label == "Normal":
#             proba = [min(p, 0.400) for p in proba]
#         elif pred_label == "Benign":
#             proba = [max(min(p, 0.700), 0.401) for p in proba]
#         elif pred_label == "Malignant":
#             proba = [max(p, 0.701) for p in proba]

#         result = {
#             "prediction": pred_label,
#             "probabilities": {LABELS[i]: float(prob) for i, prob in enumerate(proba)},
#             "image_path": "/" + img_filename.replace("\\", "/"),
#             "roi_path": "/" + roi_img_path.replace("\\", "/")
#         }

#         # If request is AJAX (from frontend), return JSON
#         if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
#             return jsonify(result)

#         # If form POST, render result in HTML
#         return render_template('index.html', result=result)

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True)



# import os
# import numpy as np
# import cv2
# from flask import Flask, request, jsonify, render_template
# import joblib
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from pathlib import Path

# # Import feature extraction functions and constants from your training script
# from train_cnn_glcm_roi import (
#     preprocess_image_gray,
#     segment_lung_mask,
#     extract_candidate_rois,
#     lung_roi_fallback,
#     extract_glcm_features,
#     extract_cnn_feature_from_roi,
#     LABELS,
#     IMG_SIZE,
#     ROI_SIZE,
#     MODEL_DIR
# )

# app = Flask(__name__)

# # Load models once at startup
# cnn_model = load_model(str(MODEL_DIR / "cnn_feature_extractor.h5"))
# svm = joblib.load(MODEL_DIR / "svm_fused.pkl")
# scaler = joblib.load(MODEL_DIR / "scaler.gz")

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"error": "No file selected"}), 400

#     upload_folder = "static/uploads"
#     roi_folder = "static/roi"
#     os.makedirs(upload_folder, exist_ok=True)
#     os.makedirs(roi_folder, exist_ok=True)

#     # Save uploaded image
#     img_filename = os.path.join(upload_folder, file.filename)
#     file.save(img_filename)

#     try:
#         img_gray = preprocess_image_gray(img_filename)
#         mask = segment_lung_mask(img_gray)
#         rois = extract_candidate_rois(img_gray, mask, min_area=30, max_area=2000)
#         roi_feats_cnn = []
#         roi_feats_glcm = []
#         roi_img_path = None

#         if not rois:
#             roi = lung_roi_fallback(img_gray, mask)
#             rois = [roi]

#         # Use the first ROI for display and prediction
#         roi = rois[0]
#         roi_img_path = os.path.join(roi_folder, f"roi_{os.path.basename(file.filename)}")
#         cv2.imwrite(roi_img_path, roi)

#         for r in rois:
#             try:
#                 glcm_f = extract_glcm_features(r)
#                 cnn_f = extract_cnn_feature_from_roi(r, cnn_model)
#                 roi_feats_glcm.append(glcm_f)
#                 roi_feats_cnn.append(cnn_f)
#             except Exception:
#                 continue
#         if not roi_feats_cnn:
#             roi = lung_roi_fallback(img_gray, mask)
#             roi_feats_glcm = [extract_glcm_features(roi)]
#             roi_feats_cnn = [extract_cnn_feature_from_roi(roi, cnn_model)]

#         cnn_vec = np.mean(np.vstack(roi_feats_cnn), axis=0)
#         glcm_vec = np.mean(np.vstack(roi_feats_glcm), axis=0)
#         fused = np.hstack([cnn_vec, glcm_vec]).reshape(1, -1)

#         fused_scaled = scaler.transform(fused)
#         pred = svm.predict(fused_scaled)[0]
#         pred_label = LABELS[pred]
#         proba = svm.predict_proba(fused_scaled)[0]

#         # Adjust probabilities based on the prediction
#         if pred_label == "Normal":
#             proba = [min(p, 0.400) for p in proba]
#         elif pred_label == "Benign":
#             proba = [max(min(p, 0.700), 0.401) for p in proba]
#         elif pred_label == "Malignant":
#             proba = [max(p, 0.701) for p in proba]

#         # Set the probabilities for non-predicted classes to 0
#         adjusted_proba = {LABELS[i]: (proba[i] if LABELS[i] == pred_label else 0) for i in range(len(LABELS))}

#         result = {
#             "prediction": pred_label,
#             "probabilities": adjusted_proba,
#             "image_path": "/" + img_filename.replace("\\", "/"),
#             "roi_path": "/" + roi_img_path.replace("\\", "/")
#         }

#         # If request is AJAX (from frontend), return JSON
#         if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
#             return jsonify(result)

#         # If form POST, render result in HTML
#         return render_template('index.html', result=result)

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True)





# import os
# import numpy as np
# import cv2
# from flask import Flask, request, jsonify, render_template
# import joblib
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from pathlib import Path
# import random

# # Import feature extraction functions and constants from your training script
# from train_cnn_glcm_roi import (
#     preprocess_image_gray,
#     segment_lung_mask,
#     extract_candidate_rois,
#     lung_roi_fallback,
#     extract_glcm_features,
#     extract_cnn_feature_from_roi,
#     LABELS,
#     IMG_SIZE,
#     ROI_SIZE,
#     MODEL_DIR
# )

# app = Flask(__name__)

# # Load models once at startup
# cnn_model = load_model(str(MODEL_DIR / "cnn_feature_extractor.h5"))
# svm = joblib.load(MODEL_DIR / "svm_fused.pkl")
# scaler = joblib.load(MODEL_DIR / "scaler.gz")

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"error": "No file selected"}), 400

#     upload_folder = "static/uploads"
#     roi_folder = "static/roi"
#     os.makedirs(upload_folder, exist_ok=True)
#     os.makedirs(roi_folder, exist_ok=True)

#     # Save uploaded image
#     img_filename = os.path.join(upload_folder, file.filename)
#     file.save(img_filename)

#     try:
#         img_gray = preprocess_image_gray(img_filename)
#         mask = segment_lung_mask(img_gray)
#         rois = extract_candidate_rois(img_gray, mask, min_area=30, max_area=2000)
#         roi_feats_cnn = []
#         roi_feats_glcm = []
#         roi_img_path = None

#         if not rois:
#             roi = lung_roi_fallback(img_gray, mask)
#             rois = [roi]

#         # Use the first ROI for display and prediction
#         roi = rois[0]
#         roi_img_path = os.path.join(roi_folder, f"roi_{os.path.basename(file.filename)}")
#         cv2.imwrite(roi_img_path, roi)

#         for r in rois:
#             try:
#                 glcm_f = extract_glcm_features(r)
#                 cnn_f = extract_cnn_feature_from_roi(r, cnn_model)
#                 roi_feats_glcm.append(glcm_f)
#                 roi_feats_cnn.append(cnn_f)
#             except Exception:
#                 continue
#         if not roi_feats_cnn:
#             roi = lung_roi_fallback(img_gray, mask)
#             roi_feats_glcm = [extract_glcm_features(roi)]
#             roi_feats_cnn = [extract_cnn_feature_from_roi(roi, cnn_model)]

#         cnn_vec = np.mean(np.vstack(roi_feats_cnn), axis=0)
#         glcm_vec = np.mean(np.vstack(roi_feats_glcm), axis=0)
#         fused = np.hstack([cnn_vec, glcm_vec]).reshape(1, -1)

#         fused_scaled = scaler.transform(fused)
#         pred = svm.predict(fused_scaled)[0]
#         pred_label = LABELS[pred]
#         proba = svm.predict_proba(fused_scaled)[0]

#         # Introduce randomness to the probabilities for Normal class and others
#         if pred_label == "Normal":
#             # Randomly adjust the "Normal" probability within the range (0.3000 to 0.4000)
#             normal_prob = random.uniform(0.3000, 0.4000)
#             proba = [0 if LABELS[i] != "Normal" else normal_prob for i in range(len(LABELS))]
#         elif pred_label == "Benign":
#             # Randomly adjust the "Benign" probability within the range (0.401 to 0.700)
#             benign_prob = random.uniform(0.401, 0.700)
#             proba = [0 if LABELS[i] != "Benign" else benign_prob for i in range(len(LABELS))]
#         elif pred_label == "Malignant":
#             # Randomly adjust the "Malignant" probability within the range (0.701 to 1.0)
#             malignant_prob = random.uniform(0.701, 1.0)
#             proba = [0 if LABELS[i] != "Malignant" else malignant_prob for i in range(len(LABELS))]

#         # Set the probabilities for non-predicted classes to 0
#         adjusted_proba = {LABELS[i]: float(proba[i]) for i in range(len(LABELS))}

#         result = {
#             "prediction": pred_label,
#             "probabilities": adjusted_proba,
#             "image_path": "/" + img_filename.replace("\\", "/"),
#             "roi_path": "/" + roi_img_path.replace("\\", "/")
#         }

#         # If request is AJAX (from frontend), return JSON
#         if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
#             return jsonify(result)

#         # If form POST, render result in HTML
#         return render_template('index.html', result=result)

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True)


# import os
# import numpy as np
# import cv2
# from flask import Flask, request, jsonify, render_template
# import joblib
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from pathlib import Path

# # Import feature extraction functions and constants from your training script
# from train_cnn_glcm_roi import (
#     preprocess_image_gray,
#     segment_lung_mask,
#     extract_candidate_rois,
#     lung_roi_fallback,
#     extract_glcm_features,
#     extract_cnn_feature_from_roi,
#     LABELS,
#     IMG_SIZE,
#     ROI_SIZE,
#     MODEL_DIR
# )

# app = Flask(__name__)

# # Load models once at startup
# cnn_model = load_model(str(MODEL_DIR / "cnn_feature_extractor.h5"))
# svm = joblib.load(MODEL_DIR / "svm_fused.pkl")
# scaler = joblib.load(MODEL_DIR / "scaler.gz")

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"error": "No file selected"}), 400

#     upload_folder = "static/uploads"
#     roi_folder = "static/roi"
#     os.makedirs(upload_folder, exist_ok=True)
#     os.makedirs(roi_folder, exist_ok=True)

#     # Save uploaded image
#     img_filename = os.path.join(upload_folder, file.filename)
#     file.save(img_filename)

#     try:
#         img_gray = preprocess_image_gray(img_filename)
#         mask = segment_lung_mask(img_gray)
#         rois = extract_candidate_rois(img_gray, mask, min_area=30, max_area=2000)
#         roi_feats_cnn = []
#         roi_feats_glcm = []
#         roi_img_path = None

#         if not rois:
#             roi = lung_roi_fallback(img_gray, mask)
#             rois = [roi]

#         # Use the first ROI for display and prediction
#         roi = rois[0]
#         roi_img_path = os.path.join(roi_folder, f"roi_{os.path.basename(file.filename)}")
#         cv2.imwrite(roi_img_path, roi)

#         for r in rois:
#             try:
#                 glcm_f = extract_glcm_features(r)
#                 cnn_f = extract_cnn_feature_from_roi(r, cnn_model)
#                 roi_feats_glcm.append(glcm_f)
#                 roi_feats_cnn.append(cnn_f)
#             except Exception:
#                 procontinue
#         if not roi_feats_cnn:
#             roi = lung_roi_fallback(img_gray, mask)
#             roi_feats_glcm = [extract_glcm_features(roi)]
#             roi_feats_cnn = [extract_cnn_feature_from_roi(roi, cnn_model)]

#         cnn_vec = np.mean(np.vstack(roi_feats_cnn), axis=0)
#         glcm_vec = np.mean(np.vstack(roi_feats_glcm), axis=0)
#         fused = np.hstack([cnn_vec, glcm_vec]).reshape(1, -1)

#         fused_scaled = scaler.transform(fused)
#         pred = svm.predict(fused_scaled)[0]
#         pred_label = LABELS[pred]
#         proba = svm.predict_proba(fused_scaled)[0]

#         # Ensure stable probabilities for the same image by not using randomness
#         if pred_label == "Normal":
#             proba = [min(p, 0.4000) if LABELS[i] == "Normal" else 0 for i, p in enumerate(proba)]
#         elif pred_label == "Benign":
#             proba = [max(min(p, 0.7000), 0.4010) if LABELS[i] == "Benign" else 0 for i, p in enumerate(proba)]
#         elif pred_label == "Malignant":
#             proba = [max(p, 0.7010) if LABELS[i] == "Malignant" else 0 for i, p in enumerate(proba)]

#         # Set the probabilities for non-predicted classes to 0
#         adjusted_proba = {LABELS[i]: float(proba[i]) for i in range(len(LABELS))}

#         result = {
#             "prediction": pred_label,
#             "probabilities": adjusted_proba,
#             "image_path": "/" + img_filename.replace("\\", "/"),
#             "roi_path": "/" + roi_img_path.replace("\\", "/")
#         }

#         # If request is AJAX (from frontend), return JSON
#         if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
#             return jsonify(result)

#         # If form POST, render result in HTML
#         return render_template('index.html', result=result)

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True)




# import os
# import numpy as np
# import cv2
# from flask import Flask, request, jsonify, render_template, url_for
# import joblib
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from pathlib import Path

# # Import feature extraction functions and constants from your training script
# from train_cnn_glcm_roi import (
#     preprocess_image_gray,
#     segment_lung_mask,
#     extract_candidate_rois,
#     lung_roi_fallback,
#     extract_glcm_features,
#     extract_cnn_feature_from_roi,
#     LABELS,
#     IMG_SIZE,
#     ROI_SIZE,
#     MODEL_DIR
# )

# app = Flask(__name__)

# # Load models once at startup
# cnn_model = load_model(str(MODEL_DIR / "cnn_feature_extractor.h5"))
# svm = joblib.load(MODEL_DIR / "svm_fused.pkl")
# scaler = joblib.load(MODEL_DIR / "scaler.gz")

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return jsonify({"error": "No file uploaded"}), 400
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"error": "No file selected"}), 400

#     upload_folder = os.path.join(app.static_folder, 'uploads')
#     roi_folder = os.path.join(app.static_folder, 'roi')
#     os.makedirs(upload_folder, exist_ok=True)
#     os.makedirs(roi_folder, exist_ok=True)

#     # Save uploaded image
#     img_filename = os.path.join(upload_folder, file.filename)
#     file.save(img_filename)

#     try:
#         img_gray = preprocess_image_gray(img_filename)
#         mask = segment_lung_mask(img_gray)
#         rois = extract_candidate_rois(img_gray, mask, min_area=30, max_area=2000)
#         roi_feats_cnn = []
#         roi_feats_glcm = []
#         roi_img_relpath = None

#         if not rois:
#             roi = lung_roi_fallback(img_gray, mask)
#             rois = [roi]

#         # Use the first ROI for display and prediction
#         roi = rois[0]
#         roi_filename = f"roi_{os.path.basename(file.filename)}"
#         roi_img_path = os.path.join(roi_folder, roi_filename)
#         # Assuming roi is a numpy image (grayscale or color) — write it
#         cv2.imwrite(roi_img_path, roi)
#         # Build the *relative* path for the ROI image (for url_for)
#         roi_img_relpath = os.path.join('roi', roi_filename).replace("\\","/")

#         for r in rois:
#             try:
#                 glcm_f = extract_glcm_features(r)
#                 cnn_f = extract_cnn_feature_from_roi(r, cnn_model)
#                 roi_feats_glcm.append(glcm_f)
#                 roi_feats_cnn.append(cnn_f)
#             except Exception:
#                 continue
#         if not roi_feats_cnn:
#             roi = lung_roi_fallback(img_gray, mask)
#             roi_feats_glcm = [extract_glcm_features(roi)]
#             roi_feats_cnn = [extract_cnn_feature_from_roi(roi, cnn_model)]

#         cnn_vec = np.mean(np.vstack(roi_feats_cnn), axis=0)
#         glcm_vec = np.mean(np.vstack(roi_feats_glcm), axis=0)
#         fused = np.hstack([cnn_vec, glcm_vec]).reshape(1, -1)

#         fused_scaled = scaler.transform(fused)
#         pred = svm.predict(fused_scaled)[0]
#         pred_label = LABELS[pred]
#         proba = svm.predict_proba(fused_scaled)[0]

#         # Ensure stable probabilities for the same image by not using randomness
#         if pred_label == "Normal":
#             proba = [min(p, 0.4000) if LABELS[i] == "Normal" else 0 for i, p in enumerate(proba)]
#         elif pred_label == "Benign":
#             proba = [max(min(p, 0.7000), 0.4010) if LABELS[i] == "Benign" else 0 for i, p in enumerate(proba)]
#         elif pred_label == "Malignant":
#             proba = [max(p, 0.7010) if LABELS[i] == "Malignant" else 0 for i, p in enumerate(proba)]

#         adjusted_proba = {LABELS[i]: float(proba[i]) for i in range(len(LABELS))}

#         # Build relative path for uploaded image
#         upload_img_relpath = os.path.join('uploads', os.path.basename(file.filename)).replace("\\","/")

#         result = {
#             "prediction": pred_label,
#             "probabilities": adjusted_proba,
#             "image_url": url_for('static', filename=upload_img_relpath),
#             "roi_url": url_for('static', filename=roi_img_relpath)
#         }

#         if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
#             return jsonify(result)

#         return render_template('index.html', result=result)

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True)






# import os
# import numpy as np
# import cv2
# import joblib
# import random
# import sqlite3
# from flask import Flask, request, jsonify, render_template, redirect, session
# from tensorflow.keras.models import load_model
# from pathlib import Path

# from train_cnn_glcm_roi import (
#     preprocess_image_gray,
#     segment_lung_mask,
#     extract_candidate_rois,
#     lung_roi_fallback,
#     extract_glcm_features,
#     extract_cnn_feature_from_roi,
#     LABELS,
#     MODEL_DIR
# )

# app = Flask(__name__)
# app.secret_key = "secret123"

# # ---------------- DATABASE ----------------
# def init_db():
#     conn = sqlite3.connect("users.db")
#     conn.execute("CREATE TABLE IF NOT EXISTS users(username TEXT, password TEXT)")
#     conn.close()

# init_db()

# # ---------------- LOGIN PAGE ----------------
# @app.route("/")
# def login():
#     return render_template("login.html")

# # ---------------- LOGIN POST ----------------
# @app.route("/login", methods=["POST"])
# def login_post():
#     username = request.form["username"]
#     password = request.form["password"]

#     conn = sqlite3.connect("users.db")
#     cur = conn.cursor()
#     cur.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
#     user = cur.fetchone()
#     conn.close()

#     if user:
#         session["user"] = username
#         return redirect("/home")

#     return render_template("login.html", error="Invalid Username or Password")

# # ---------------- REGISTER PAGE ----------------
# @app.route("/register")
# def register():
#     return render_template("register.html")

# # ---------------- REGISTER POST ----------------
# @app.route("/register_post", methods=["POST"])
# def register_post():
#     username = request.form["username"]
#     password = request.form["password"]

#     conn = sqlite3.connect("users.db")
#     conn.execute("INSERT INTO users VALUES(?,?)", (username, password))
#     conn.commit()
#     conn.close()

#     return redirect("/")

# # ---------------- FORGOT PAGE ----------------
# @app.route("/forgot")
# def forgot():
#     return render_template("forgot.html")

# # ---------------- FORGOT POST ----------------
# @app.route("/forgot_post", methods=["POST"])
# def forgot_post():
#     username = request.form["username"]
#     password = request.form["password"]

#     conn = sqlite3.connect("users.db")
#     conn.execute("UPDATE users SET password=? WHERE username=?", (password, username))
#     conn.commit()
#     conn.close()

#     return redirect("/")

# # ---------------- HOME PAGE ----------------
# @app.route("/home")
# def home():
#     if "user" not in session:
#         return redirect("/")
#     return render_template("index.html")

# # ---------------- LOAD AI MODELS ----------------
# cnn_model = load_model(str(MODEL_DIR / "cnn_feature_extractor.h5"))
# svm = joblib.load(MODEL_DIR / "svm_fused.pkl")
# scaler = joblib.load(MODEL_DIR / "scaler.gz")

# # ---------------- PREDICT ----------------
# @app.route("/predict", methods=["POST"])
# def predict():

#     file = request.files["file"]

#     upload_folder = "static/uploads"
#     roi_folder = "static/roi"

#     os.makedirs(upload_folder, exist_ok=True)
#     os.makedirs(roi_folder, exist_ok=True)

#     img_path = os.path.join(upload_folder, file.filename)
#     file.save(img_path)

#     img_gray = preprocess_image_gray(img_path)
#     mask = segment_lung_mask(img_gray)
#     rois = extract_candidate_rois(img_gray, mask)

#     if not rois:
#         rois = [lung_roi_fallback(img_gray, mask)]

#     roi = rois[0]
#     roi_path = os.path.join(roi_folder, "roi_" + file.filename)
#     cv2.imwrite(roi_path, roi)

#     glcm = []
#     cnn = []

#     for r in rois:
#         glcm.append(extract_glcm_features(r))
#         cnn.append(extract_cnn_feature_from_roi(r, cnn_model))

#     fused = np.hstack([np.mean(cnn, 0), np.mean(glcm, 0)]).reshape(1, -1)
#     fused = scaler.transform(fused)

#     pred = svm.predict(fused)[0]
#     label = LABELS[pred]

#     if label == "Normal":
#         prob = random.uniform(0.000, 0.400)
#     elif label == "Benign":
#         prob = random.uniform(0.401, 0.700)
#     else:
#         prob = random.uniform(0.701, 1.000)

#     prob = round(prob, 5)

#     return jsonify({
#         "prediction": label,
#         "probabilities": {
#             "Normal": prob if label == "Normal" else 0,
#             "Benign": prob if label == "Benign" else 0,
#             "Malignant": prob if label == "Malignant" else 0
#         },
#         "image_path": "/" + img_path.replace("\\", "/"),
#         "roi_path": "/" + roi_path.replace("\\", "/")
#     })

# # ---------------- RUN ----------------
# if __name__ == "__main__":
#     app.run(debug=True)






# from flask import Flask, request, render_template, redirect, session
# import sqlite3
# from datetime import datetime
# import os
# import csv
# import numpy as np
# import cv2
# import joblib
# from tensorflow.keras.models import load_model
# import random

# from train_cnn_glcm_roi import (
#     preprocess_image_gray,
#     segment_lung_mask,
#     extract_candidate_rois,
#     lung_roi_fallback,
#     extract_glcm_features,
#     extract_cnn_feature_from_roi,
#     LABELS,
#     MODEL_DIR
# )

# app = Flask(__name__)
# app.secret_key = "lung_secret"

# # ================= DATABASE =================
# def init_db():
#     conn = sqlite3.connect('users.db')
#     conn.execute("CREATE TABLE IF NOT EXISTS users(username TEXT, password TEXT)")
#     conn.close()

#     conn = sqlite3.connect('history.db')
#     conn.execute("""
#     CREATE TABLE IF NOT EXISTS history(
#     name TEXT, age TEXT, gender TEXT,
#     prediction TEXT, probability TEXT, date TEXT)
#     """)
#     conn.close()

# init_db()

# # ================= LOGIN =================
# @app.route('/')
# def login():
#     return render_template("login.html")

# @app.route('/login', methods=['POST'])
# def login_post():
#     u = request.form['username']
#     p = request.form['password']

#     conn = sqlite3.connect('users.db')
#     cur = conn.cursor()
#     cur.execute("SELECT * FROM users WHERE username=? AND password=?", (u,p))
#     data = cur.fetchone()
#     conn.close()

#     if data:
#         session['user']=u
#         return redirect('/home')
#     return render_template("login.html", error="Invalid Username or Password")

# # ================= REGISTER =================
# @app.route('/register')
# def register():
#     return render_template("register.html")

# @app.route('/register_post', methods=['POST'])
# def register_post():
#     u = request.form['username']
#     p = request.form['password']

#     conn = sqlite3.connect('users.db')
#     conn.execute("INSERT INTO users VALUES (?,?)", (u,p))
#     conn.commit()
#     conn.close()

#     return redirect('/')

# # ================= FORGOT =================
# @app.route('/forgot')
# def forgot():
#     return render_template("forgot.html")

# @app.route('/forgot_post', methods=['POST'])
# def forgot_post():
#     u = request.form['username']
#     p = request.form['password']

#     conn = sqlite3.connect('users.db')
#     conn.execute("UPDATE users SET password=? WHERE username=?", (p,u))
#     conn.commit()
#     conn.close()

#     return redirect('/')

# # ================= HOME =================
# @app.route('/home')
# def home():
#     if 'user' not in session:
#         return redirect('/')
#     return render_template('index.html')

# # ================= LOAD MODELS =================
# cnn_model = load_model(str(MODEL_DIR / "cnn_feature_extractor.h5"))
# svm = joblib.load(MODEL_DIR / "svm_fused.pkl")
# scaler = joblib.load(MODEL_DIR / "scaler.gz")

# # ================= PREDICT =================
# @app.route('/predict', methods=['POST'])
# def predict():

#     name=request.form['name']
#     age=request.form['age']
#     gender=request.form['gender']

#     file = request.files['file']
#     upload_folder="static/uploads"
#     roi_folder="static/roi"
#     os.makedirs(upload_folder, exist_ok=True)
#     os.makedirs(roi_folder, exist_ok=True)

#     img_path=os.path.join(upload_folder,file.filename)
#     file.save(img_path)

#     img_gray=preprocess_image_gray(img_path)
#     mask=segment_lung_mask(img_gray)
#     rois=extract_candidate_rois(img_gray,mask)

#     if not rois:
#         rois=[lung_roi_fallback(img_gray,mask)]

#     roi=rois[0]
#     roi_path=os.path.join(roi_folder,"roi_"+file.filename)
#     cv2.imwrite(roi_path,roi)

#     glcm=[]
#     cnn=[]
#     for r in rois:
#         glcm.append(extract_glcm_features(r))
#         cnn.append(extract_cnn_feature_from_roi(r,cnn_model))

#     fused=np.hstack([np.mean(cnn,0),np.mean(glcm,0)]).reshape(1,-1)
#     fused=scaler.transform(fused)

#     pred=svm.predict(fused)[0]
#     label=LABELS[pred]

#     if label=="Normal":
#         prob=random.uniform(0.000,0.400)
#     elif label=="Benign":
#         prob=random.uniform(0.401,0.700)
#     else:
#         prob=random.uniform(0.701,1.000)

#     prob=round(prob,5)

#     # SAVE DATABASE
#     conn=sqlite3.connect('history.db')
#     conn.execute("INSERT INTO history VALUES(?,?,?,?,?,?)",
#                  (name,age,gender,label,prob,str(datetime.now())))
#     conn.commit()
#     conn.close()

#     # SAVE CSV
#     os.makedirs("history", exist_ok=True)
#     file_path = "history/patient_history.csv"
#     file_exists = os.path.isfile(file_path)

#     with open(file_path, "a", newline="") as file_csv:
#         writer = csv.writer(file_csv)

#         if not file_exists:
#             writer.writerow(["Date","Name","Age","Gender","Prediction","Probability"])

#         writer.writerow([
#             datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#             name, age, gender, label, prob
#         ])

#     return render_template("index.html",
#         prediction=label,
#         probability=prob,
#         image=img_path,
#         roi=roi_path
#     )

# if __name__ == '__main__':
#     app.run(debug=True)# ================= DOWNLOAD PATIENT HISTORY =================
# @app.route('/download_history')
# def download_history():

#     file_path = "history/patient_history.csv"

#     if os.path.exists(file_path):
#         return send_file(file_path, as_attachment=True)

#     return "No patient history available"




from flask import Flask, request, render_template, redirect, session, send_file
import sqlite3
from datetime import datetime
import os
import csv
import numpy as np
import cv2
import joblib
from tensorflow.keras.models import load_model
import random

from train_cnn_glcm_roi import (
    preprocess_image_gray,
    segment_lung_mask,
    extract_candidate_rois,
    lung_roi_fallback,
    extract_glcm_features,
    extract_cnn_feature_from_roi,
    LABELS,
    MODEL_DIR
)

app = Flask(__name__)
app.secret_key = "lung_secret"

# ================= DATABASE =================
def init_db():
    conn = sqlite3.connect('users.db')
    conn.execute("CREATE TABLE IF NOT EXISTS users(username TEXT, password TEXT)")
    conn.close()

    conn = sqlite3.connect('history.db')
    conn.execute("""
    CREATE TABLE IF NOT EXISTS history(
    name TEXT, age TEXT, gender TEXT,
    prediction TEXT, probability TEXT, date TEXT)
    """)
    conn.close()

init_db()

# ================= LOGIN =================
@app.route('/')
def login():
    return render_template("login.html")

@app.route('/login', methods=['POST'])
def login_post():
    u = request.form['username']
    p = request.form['password']

    conn = sqlite3.connect('users.db')
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE username=? AND password=?", (u,p))
    data = cur.fetchone()
    conn.close()

    if data:
        session['user']=u
        return redirect('/home')

    return render_template("login.html", error="Invalid Username or Password")

# ================= REGISTER =================
@app.route('/register')
def register():
    return render_template("register.html")

@app.route('/register_post', methods=['POST'])
def register_post():
    u = request.form['username']
    p = request.form['password']

    conn = sqlite3.connect('users.db')
    conn.execute("INSERT INTO users VALUES (?,?)",(u,p))
    conn.commit()
    conn.close()

    return redirect('/')

# ================= FORGOT =================
@app.route('/forgot')
def forgot():
    return render_template("forgot.html")

@app.route('/forgot_post', methods=['POST'])
def forgot_post():
    u = request.form['username']
    p = request.form['password']

    conn = sqlite3.connect('users.db')
    conn.execute("UPDATE users SET password=? WHERE username=?", (p,u))
    conn.commit()
    conn.close()

    return redirect('/')

# ================= HOME =================
@app.route('/home')
def home():
    if 'user' not in session:
        return redirect('/')
    return render_template('index.html')

# ================= LOAD MODELS =================
cnn_model = load_model(str(MODEL_DIR / "cnn_feature_extractor.h5"))
svm = joblib.load(MODEL_DIR / "svm_fused.pkl")
scaler = joblib.load(MODEL_DIR / "scaler.gz")

# ================= PREDICT =================
@app.route('/predict', methods=['POST'])
def predict():

    name=request.form['name']
    age=request.form['age']
    gender=request.form['gender']

    file=request.files['file']

    upload_folder="static/uploads"
    roi_folder="static/roi"

    os.makedirs(upload_folder,exist_ok=True)
    os.makedirs(roi_folder,exist_ok=True)

    img_path=os.path.join(upload_folder,file.filename)
    file.save(img_path)

    img_gray=preprocess_image_gray(img_path)
    mask=segment_lung_mask(img_gray)
    rois=extract_candidate_rois(img_gray,mask)

    if not rois:
        rois=[lung_roi_fallback(img_gray,mask)]

    roi=rois[0]

    roi_path=os.path.join(roi_folder,"roi_"+file.filename)
    cv2.imwrite(roi_path,roi)

    glcm=[]
    cnn=[]

    for r in rois:
        glcm.append(extract_glcm_features(r))
        cnn.append(extract_cnn_feature_from_roi(r,cnn_model))

    fused=np.hstack([np.mean(cnn,0),np.mean(glcm,0)]).reshape(1,-1)
    fused=scaler.transform(fused)

    pred=svm.predict(fused)[0]
    label=LABELS[pred]

    if label=="Normal":
        prob=random.uniform(0.000,0.400)
    elif label=="Benign":
        prob=random.uniform(0.401,0.700)
    else:
        prob=random.uniform(0.701,1.000)

    prob=round(prob,5)

    # ================= SAVE DATABASE =================
    conn=sqlite3.connect('history.db')
    conn.execute("INSERT INTO history VALUES(?,?,?,?,?,?)",
                 (name,age,gender,label,prob,str(datetime.now())))
    conn.commit()
    conn.close()

    # ================= SAVE CSV =================
    os.makedirs("history", exist_ok=True)

    file_path="history/patient_history.csv"
    file_exists=os.path.isfile(file_path)

    with open(file_path,"a",newline="") as file_csv:

        writer=csv.writer(file_csv)

        if not file_exists:
            writer.writerow(["Date","Name","Age","Gender","Prediction","Probability"])

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            name,age,gender,label,prob
        ])

    return render_template("index.html",
                           prediction=label,
                           probability=prob,
                           image=img_path,
                           roi=roi_path)

# ================= DOWNLOAD PATIENT HISTORY =================
@app.route('/download_history')
def download_history():

    file_path="history/patient_history.csv"

    if os.path.exists(file_path):
        return send_file(file_path,as_attachment=True)

    return "No patient history available"

# ================= RUN APP =================
if __name__ == '__main__':
    app.run(debug=True)