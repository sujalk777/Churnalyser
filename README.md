
# Churn Modelling of Bank by Deep Learning

## Overview


https://github.com/user-attachments/assets/2432edd3-d75a-427c-9aa0-c9f5ed4b93f0


This project leverages deep learning techniques to predict customer churn in the banking sector. It employs a comprehensive pipeline for data preprocessing, model optimization, and user interface development. The goal is to identify customers who are likely to churn, enabling proactive retention strategies.

## Project Steps and Methods

1. **Data Preprocessing:**
   - **SMOTE (Synthetic Minority Over-sampling Technique):** Applied to address class imbalance by generating synthetic samples for the minority class to improve model performance.

2. **Model Optimization:**
   - **Keras Tuner:** Used to find the best hyperparameters for the deep learning model, including optimizing the number of layers, units per layer, and dropout rates for improved accuracy and generalization.

3. **Model Deployment:**
   - **Streamlit:** Implemented to create an interactive web application that allows users to input data and receive predictions on customer churn. The Streamlit app provides a user-friendly interface for real-time model predictions and visualizations.

## Requirements

- Python 3.7+
- TensorFlow 2.x
- Keras Tuner
- Imbalanced-learn
- Streamlit
- Pandas
- Numpy
- Scikit-learn

## Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/shubh123a3/-Churn-Modelling-of-Bank-by-Deep-Learning.git
cd Churn-Modelling-of-Bank-by-Deep-Learning
pip install -r requirements.txt
```

## Usage

1. **Train the Model:**
   - Run the training script to preprocess the data, apply SMOTE, tune hyperparameters using Keras Tuner, and train the deep learning model.

2. **Launch the Streamlit App:**
   - Start the Streamlit app to interact with the model and visualize predictions:
   ```bash
   streamlit run app.py
   ```

3. **Interact with the Model:**
   - Use the Streamlit interface to input customer data and obtain churn predictions.

## Files

- `data_preprocessing.py`: Data preprocessing and SMOTE application.
- `model_tuning.py`: Hyperparameter tuning with Keras Tuner.
- `train_model.py`: Model training script.
- `app.py`: Streamlit application for model interaction.

## Contribution

Feel free to contribute to this project by opening issues, submitting pull requests, or providing feedback.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html) for handling class imbalance.
- [Keras Tuner](https://keras.io/keras_tuner/) for hyperparameter optimization.
- [Streamlit](https://streamlit.io/) for creating interactive web applications.
