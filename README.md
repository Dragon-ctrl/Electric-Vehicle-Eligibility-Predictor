# Electric Vehicle Eligibility Prediction

This project aims to predict the Clean Alternative Fuel Vehicle (CAFV) eligibility of electric vehicles using machine learning. The prediction is based on several features such as vehicle make, model, year, electric range, and more.

The project is divided into several stages: data preprocessing, feature selection, model training, and deployment. Each stage is implemented in a separate Python script.

## Installation

1. Clone this repository: `git clone https://github.com/your-username/ev-eligibility-prediction.git`
2. Change into the project directory: `cd ev-eligibility-prediction`
3. Create a virtual environment: `python3 -m venv env`
4. Activate the virtual environment: `source env/bin/activate`
5. Install the required packages: `pip install -r requirements.txt`

## Usage

### Data Preprocessing

The `preprocess_data.py` script loads the Electric Vehicle Population Data CSV file, drops unnecessary columns, drops rows with missing values, and converts categorical variables to numerical using one-hot encoding. The preprocessed data is saved to `preprocessed_data.csv`. To run the script, execute:

python preprocess_data.py

### Feature Selection

The `feature_selection.py` script loads the preprocessed data, drops unnecessary columns, drops rows with missing values, separates numerical and categorical columns, standardizes numerical features, and one-hot encodes categorical features. The selected features are saved to `selected_features.csv`. To run the script, execute:

python feature_selection.py

### Model Training

The `train_model.py` script loads the preprocessed and selected features data, splits the data into training and testing sets, defines several regression models to test, and performs hyperparameter tuning using grid search. The best model is selected based on the mean squared error score on the testing set, and is saved to `model.joblib`. To run the script, execute:

python train_model.py

### Model Deployment

The `app.py` script loads the trained model and exposes a Flask endpoint at `/predict` that accepts a JSON payload with the selected features and returns the CAFV eligibility prediction as a JSON response. To run the Flask app, execute:

```javascript
export FLASK_APP=app.py
flask run
```
## License

This project is licensed under the MIT License - see the [License](/LICENSE) file for details.
