# Computer Vision Project: Blind Aid - Color Recognition

## Project Overview
The Omdena Romania Chapter team embarked on a mission to develop a Computer Vision model designed to assist individuals with mobile phones in navigating their surroundings. The primary objective is to accurately identify objects, people, and cars in real-time, empowering users to better perceive their environment. Additionally, the project aims to enhance daily indoor tasks, such as shopping, selecting clothing, distinguishing colors, managing finances, retrieving product information, and providing indoor navigation support.

## Dataset
For this project, we utilized a Kaggle dataset containing 250 images, covering 10 distinct color classes. The dataset was split into training, validation, and testing sets, comprising 150, 50, and 50 images, respectively. To facilitate model training, we used `tf.keras.utils.image_dataset_from_directory` data loader and encoded the target labels into one-hot vectors using `tf.one_hot`.

## Model
We employed the VGG16 model, pre-trained with ImageNet weights, and extended it by adding additional layers including Average Pooling, Dropout, and an output layer with sigmoid activation. Weight decay (WD) was used to mitigate overfitting, and a learning rate scheduler with logarithmic warm-up and cosine decay was applied.

### Model Parameters
- `N_EPOCHS`: 5
- `VERBOSE`: 1
- `N_REPLICAS`: 2 (number of GPUs)
- `LR_MAX`: 5e-6 * `N_REPLICAS`
- `WD_RATIO`: 1e-5
- `N_WARMUP_EPOCHS`: 0
- `num_classes`: 10

## Evaluation Metrics
On the test dataset, the model's performance was assessed using the following evaluation metrics:

- **VGG16:**
  - **Loss**: Measuring how well the model minimized errors.
  - **F1 Score**: Indicating the model's ability to correctly classify colors.
  - **Precision**: Reflecting the model's success in reducing false positives.
  - **Recall**: Highlighting the model's effectiveness in minimizing false negatives.
  - **AUC**: Showcasing the model's performance in distinguishing between positive and negative instances.
  - **Binary Accuracy**: Signifying the model's proficiency in binary classification tasks.

- **EfficientNet B5:**
  - **Loss**: Measuring how well the model minimized errors.
  - **F1 Score**: Indicating the model's ability to correctly classify colors.
  - **Precision**: Reflecting the model's success in reducing false positives.
  - **Recall**: Highlighting the model's effectiveness in minimizing false negatives.
  - **AUC**: Showcasing the model's performance in distinguishing between positive and negative instances.
  - **Binary Accuracy**: Signifying the model's proficiency in binary classification tasks.

- **EfficientNet B7:**
  - **Loss**: Measuring how well the model minimized errors.
  - **F1 Score**: Indicating the model's ability to correctly classify colors.
  - **Precision**: Reflecting the model's success in reducing false positives.
  - **Recall**: Highlighting the model's effectiveness in minimizing false negatives.
  - **AUC**: Showcasing the model's performance in distinguishing between positive and negative instances.
  - **Binary Accuracy**: Signifying the model's proficiency in binary classification tasks.

## Conclusion
This Computer Vision project successfully developed models, including VGG16, EfficientNet B5, and EfficientNet B7, to assist visually impaired individuals in recognizing colors and objects. The models demonstrated different levels of performance, with VGG16 achieving a balance between precision and recall. By utilizing a diverse dataset and fine-tuning pre-trained models, we enhanced the quality of real-time assistance provided to users.
