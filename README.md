# Eye Retina Disease Detection

This project aims to detect eye retina diseases using deep learning models. The process involves data preprocessing, noise removal, feature extraction, model training, and evaluation. The final model is designed to provide accurate classification of retina diseases.

---

## Features

- **Data Preprocessing**:  
  Various algorithms and filters were applied to clean and enhance the dataset for improved model performance.
  
- **Noise Removal**:  
  Advanced noise removal techniques, such as Gaussian filtering and Median filtering, were used to improve image quality.

- **Deep Learning Models**:  
  State-of-the-art deep learning architectures were implemented and trained for the classification task. These include CNNs and other advanced networks.

- **Model Evaluation**:  
  The models were evaluated based on metrics such as accuracy, precision, recall, and F1-score. The best-performing model was selected as the final classifier.

- **Final Model**:  
  The final model integrates the preprocessing pipeline and optimized deep learning model for robust disease detection.

---

## Project Workflow

1. **Data Collection**  
   - Retina images were collected from publicly available datasets.
   - Images were labeled for supervised training.

2. **Data Preprocessing**  
   - Applied contrast adjustment, resizing, and normalization.
   - Used filters like Gaussian and Median for noise reduction.

3. **Feature Extraction**  
   - Leveraged convolutional layers in deep learning models for feature extraction.

4. **Model Training**  
   - Trained multiple deep learning models using TensorFlow/Keras.
   - Fine-tuned hyperparameters for optimal performance.

5. **Model Evaluation**  
   - Evaluated the models using standard metrics.
   - Selected the best model based on performance.

6. **Deployment-Ready Model**  
   - Integrated preprocessing and trained model into a pipeline.
   - Saved the final model in `.h5` format for deployment.

---

## Technologies Used

- **Programming Languages**: Python  
- **Deep Learning Frameworks**: TensorFlow, Keras  
- **Libraries**: NumPy, Pandas, Matplotlib, Seaborn  
- **Image Processing Tools**: OpenCV, Scikit-image  
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score  

---
