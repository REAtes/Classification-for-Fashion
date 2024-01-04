# Fashion MNIST: Data Exploration and Deep Learning Model

This project aims to create a deep learning model for classifying fashion items using the Fashion MNIST dataset. Below, you can find the steps of the project and the results obtained.

## Steps

1. **Data Exploration**: Detailed information about the Fashion MNIST dataset is provided. In this step, the dataset is loaded and visualized.

2. **Model Creation**: Two different models are created:
   - A Convolutional Neural Network (CNN) model with 32 filters
   - A CNN model with 64 filters

3. **Training and Evaluation**: Both models are trained and their accuracy scores are recorded.

4. **New Model**: A new model with an added MaxPooling layer is created and trained.

5. **Hyperparameter Tuning**: Hyperparameter tuning is performed for the new model and the model is retrained.

6. **Results**: The test dataset is examined, and predictions are made. The predictions are visualized and a classification report is generated.

## Results

- 32-filter CNN model:
  - Training loss: 0.1421
  - Training accuracy: 0.9482
  - Validation loss: 0.2511
  - Validation accuracy: 0.9165

- 64-filter CNN model:
  - Training loss: 0.1114
  - Training accuracy: 0.9586
  - Validation loss: 0.2710
  - Validation accuracy: 0.9119

- New Model (MaxPooling):
  - Training loss: 0.0156
  - Training accuracy: 0.9963
  - Validation loss: 0.4081
  - Validation accuracy: 0.9152

- Hyperparameter Tuning (RMSprop optimizer):
  - Training loss: 0.0048
  - Training accuracy: 0.9986
  - Validation loss: 1.3150
  - Validation accuracy: 0.9067


## Images

The best-performing model, which is the new model, is visualized with predictions made on the test dataset. The predictions are displayed alongside the true labels.

## Classification Report

A classification report was generated using the predictions made on the test dataset. It can be found below:

```
             precision    recall  f1-score   support
     Class 0       0.87      0.84      0.86      1000
     Class 1       0.98      0.98      0.98      1000
     Class 2       0.89      0.86      0.87      1000
     Class 3       0.92      0.91      0.91      1000
     Class 4       0.88      0.85      0.86      1000
     Class 5       0.98      0.96      0.97      1000
     Class 6       0.74      0.82      0.78      1000
     Class 7       0.94      0.96      0.95      1000
     Class 8       0.98      0.97      0.98      1000
     Class 9       0.96      0.97      0.96      1000

    accuracy                           0.91     10000
   macro avg       0.91      0.91      0.91     10000
weighted avg       0.91      0.91      0.91     10000

```

---

The model that yields **the best results** appears to be **the New Model (MaxPooling)**. This model stands out with the lowest training loss and the highest training accuracy. Additionally, the validation loss and validation accuracy are also at satisfactory levels. Therefore, this model seems to have performed better compared to the others.

***Please note*** that evaluating additional metrics such as the classification report or performance on the test dataset is important for a comprehensive comparison.
