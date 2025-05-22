# 🚗 Car Price Prediction using TensorFlow

This project uses TensorFlow to build and train a neural network model that predicts car prices based on various features like mileage, horsepower, torque, and others. The dataset is preprocessed, normalized, and split into training, validation, and testing subsets. Performance is evaluated using MAE and RMSE.

---

## 📂 Project Structure

car-price-prediction/
├── train.csv
├── model.png
├── car_price_prediction.ipynb (or .py)
└── README.md

yaml
Copy
Edit

---

## 📊 Dataset Description

The dataset used in this project is assumed to be a CSV file named `train.csv`. It contains the following columns:

| Column         | Description                          |
|----------------|--------------------------------------|
| `years`        | Age of the car                       |
| `km`           | Kilometers driven                    |
| `rating`       | User rating of the car               |
| `condition`    | Numeric value representing condition |
| `economy`      | Fuel economy (km/l or mpg)           |
| `top speed`    | Maximum speed of the car             |
| `hp`           | Horsepower                           |
| `torque`       | Torque of the engine                 |
| `current price`| Target variable (car price)          |

---

## 🧰 Dependencies

Make sure the following Python packages are installed:

```bash
pip install tensorflow pandas seaborn matplotlib numpy
🛠️ How It Works
1. Data Loading and Exploration
Reads the dataset using pandas.

Plots pairwise relationships using Seaborn for feature visualization.

2. Data Preparation
Converts the dataset into a TensorFlow tensor and shuffles it.

Extracts features (x) and labels (y).

Splits the data into training (80%), validation (10%), and test (10%) sets.

Converts each split into tf.data.Dataset for efficient training.

3. Normalization
Applies feature-wise normalization using TensorFlow's Normalization layer.

4. Model Architecture
A feed-forward neural network:

InputLayer(input_shape=(8,))

Normalization

Dense(128, activation='relu')

Dense(128, activation='relu')

Dense(128, activation='relu')

Dense(1) – Output layer for regression

5. Compilation and Training
Loss: MeanAbsoluteError

Optimizer: Adam

Metric: RootMeanSquaredError

Trained for 100 epochs.

6. Evaluation and Visualization
Loss and RMSE are plotted over epochs.

Model is evaluated on the test dataset.

Predictions are compared to actual prices using bar plots.

📈 Results and Metrics
Evaluation metrics:

MAE (Mean Absolute Error): Measures average absolute difference between predictions and true values.

RMSE (Root Mean Squared Error): Penalizes larger errors more than MAE.

Both are tracked during training for model performance analysis.

📷 Model Visualization
The model architecture is saved to model.png using:

python
Copy
Edit
tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True)
🧪 Sample Prediction
To predict the price for one car from the test set:

python
Copy
Edit
model.predict(tf.expand_dims(X_test[0], axis=0))
Compare with the actual value:

python
Copy
Edit
Y_test[0]
📊 Prediction vs Actual (Top 100 Samples)
A bar chart comparing predicted and actual car prices for the first 100 samples in the test set:

python
Copy
Edit
ind = np.arange(100)
plt.figure(figsize=(10,5))
width = 1
plt.bar(ind, y_pred, width, label='Predicted Price')
plt.bar(ind + width, y_true, width, label='Actual Price')
plt.xlabel('Sample Index')
plt.ylabel('Car Price')
plt.legend()
plt.show()
📝 Notes
Ensure the dataset is cleaned and preprocessed as needed before training.

Consider experimenting with different architectures and learning rates for better results.

Additional metrics like R² score can be used for a deeper evaluation of regression performance.

📬 Contact
For questions, suggestions, or contributions, feel free to open an issue or submit a pull request.

vbnet
Copy
Edit

Let me know if you'd like a downloadable file or a version tailored for a Jupyter Notebook project structure.
