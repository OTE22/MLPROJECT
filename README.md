# 🎓 Student Exam Performance Prediction System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**An End-to-End Machine Learning Pipeline for Predicting Student Mathematics Performance**

[Features](#-features) • [Architecture](#-architecture) • [Installation](#-installation) • [Usage](#-usage) • [API Reference](#-api-reference) • [Documentation](#-documentation)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Project Architecture](#-architecture)
- [Technology Stack](#-technology-stack)
- [Installation](#-installation)
- [Usage](#-usage)
- [Pipeline Components](#-pipeline-components)
- [Model Training](#-model-training)
- [API Reference](#-api-reference)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [Logging & Exception Handling](#-logging--exception-handling)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## 🔍 Overview

The **Student Exam Performance Prediction System** is a comprehensive machine learning solution designed to predict students' mathematics exam scores based on various demographic and academic factors. This end-to-end pipeline encompasses data ingestion, transformation, model training, and deployment through a user-friendly web interface.

### Problem Statement

Educational institutions need data-driven insights to:
- Identify students who may need additional support
- Understand factors affecting academic performance
- Optimize resource allocation for student success
- Provide early intervention for at-risk students

### Solution

This ML pipeline analyzes multiple features including:
- **Demographic Information**: Gender, race/ethnicity
- **Parental Education**: Level of parents' educational attainment
- **Academic Factors**: Reading scores, writing scores
- **Support Programs**: Lunch type, test preparation course completion

The system employs multiple machine learning algorithms and automatically selects the best-performing model through hyperparameter tuning.

---

## ✨ Features

### 🚀 Core Capabilities

- **End-to-End ML Pipeline**: Automated workflow from data ingestion to model deployment
- **Multi-Model Training**: Evaluates 8+ machine learning algorithms simultaneously
- **Hyperparameter Optimization**: GridSearchCV for optimal model parameters
- **Production-Ready API**: RESTful Flask application for real-time predictions
- **Interactive Web Interface**: User-friendly form-based prediction system
- **Robust Error Handling**: Custom exception handling with detailed logging
- **Modular Architecture**: Reusable components for easy maintenance and scaling

### 🎯 Machine Learning Models

The system evaluates the following algorithms:

1. **Random Forest Classifier**
2. **Gradient Boosting Classifier**
3. **Logistic Regression**
4. **Decision Tree Classifier**
5. **XGBoost Classifier**
6. **CatBoost Regressor**
7. **AdaBoost Classifier**
8. **Support Vector Classifier (SVC)**

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Data Ingestion Layer                     │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Raw Data → Train/Test Split → CSV Storage         │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                  Data Transformation Layer                   │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Numerical Pipeline → StandardScaler               │    │
│  │  Categorical Pipeline → OneHotEncoder → Scaler     │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                    Model Training Layer                      │
│  ┌────────────────────────────────────────────────────┐    │
│  │  GridSearchCV → Best Model Selection → Serialization│   │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│                   Prediction/API Layer                       │
│  ┌────────────────────────────────────────────────────┐    │
│  │  Flask API → Load Models → Transform → Predict     │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Technology Stack

### Backend & ML
- **Python 3.8+**: Core programming language
- **Flask**: Web framework for API development
- **scikit-learn**: Machine learning algorithms and preprocessing
- **XGBoost**: Gradient boosting framework
- **CatBoost**: Categorical boosting algorithm
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Dill**: Advanced object serialization

### Frontend
- **HTML5/CSS3**: Modern responsive UI
- **JavaScript (ES6+)**: Interactive user experience
- **Font Awesome**: Icon library

### Development Tools
- **Logging**: Built-in Python logging with custom configuration
- **Exception Handling**: Custom exception classes for debugging
- **setuptools**: Package management and distribution

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### Step-by-Step Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/OTE22/MLPROJECT.git
   cd mlproject
   ```

2. **Create Virtual Environment**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Package in Editable Mode**
   ```bash
   pip install -e .
   ```

### Quick Install (One-Liner)
```bash
git clone https://github.com/OTE22/MLPROJECT.git && cd mlproject && python -m venv venv && source venv/bin/activate && pip install -r requirements.txt
```

---

## 🚀 Usage

### Training the Model

To train the ML models from scratch:

```python
from src.component.data_ingestion import DataIngestion
from src.component.data_transformation import DataTransformation
from src.component.model_trainer import ModelTrainer

# Step 1: Data Ingestion
data_ingestion = DataIngestion()
train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

# Step 2: Data Transformation
data_transformation = DataTransformation()
train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(
    train_data_path, test_data_path
)

# Step 3: Model Training
model_trainer = ModelTrainer()
metrics = model_trainer.initiate_model_trainer(train_arr, test_arr)
print(f"Model Metrics: {metrics}")
```

### Running the Web Application

1. **Start the Flask Server**
   ```bash
   python application.py
   ```

2. **Access the Application**
   - Open your browser and navigate to: `http://localhost:5000`
   - Or: `http://127.0.0.1:5000`

3. **Make Predictions**
   - Fill in the student information form
   - Click "Predict Math Score"
   - View the predicted score instantly

### Making Predictions via API

```python
import requests
import json

# Prepare student data
student_data = {
    'gender': 'female',
    'ethnicity': 'group B',
    'parental_level_of_education': "bachelor's degree",
    'lunch': 'standard',
    'test_preparation_course': 'completed',
    'reading_score': 72,
    'writing_score': 74
}

# Send POST request
response = requests.post('http://localhost:5000/predict', data=student_data)

# Get prediction
print(f"Predicted Math Score: {response.text}")
```

---

## 🔧 Pipeline Components

### 1. Data Ingestion (`src/component/data_ingestion.py`)

**Purpose**: Load raw data and split into training and testing sets.

**Key Features**:
- Reads data from CSV files
- Performs 80-20 train-test split
- Saves processed data to artifacts directory
- Implements logging for tracking

**Configuration**:
```python
@dataclass(frozen=True)
class DataIngestionConfig:
    train_data_path: str = 'artifacts/train.csv'
    test_data_path: str = 'artifacts/test.csv'
    raw_data_path: str = 'artifacts/data.csv'
```

**Usage**:
```python
data_ingestion = DataIngestion()
train_path, test_path = data_ingestion.initiate_data_ingestion()
```

---

### 2. Data Transformation (`src/component/data_transformation.py`)

**Purpose**: Preprocess and transform raw data for model training.

**Numerical Pipeline**:
1. **Imputation**: Missing values filled with median
2. **Scaling**: StandardScaler normalization

**Categorical Pipeline**:
1. **Imputation**: Missing values filled with most frequent value
2. **Encoding**: OneHotEncoder for categorical variables
3. **Scaling**: StandardScaler (with_mean=False for sparse matrices)

**Features**:
- **Numerical**: `writing_score`, `reading_score`
- **Categorical**: `gender`, `race_ethnicity`, `parental_level_of_education`, `lunch`, `test_preparation_course`

**Configuration**:
```python
@dataclass(frozen=True)
class DataTransformationConfig:
    preprocessor_obj_file_path: str = 'artifacts/preprocessor.pkl'
```

**Output**:
- Transformed training array
- Transformed testing array
- Serialized preprocessor object

---

### 3. Model Trainer (`src/component/model_trainer.py`)

**Purpose**: Train multiple ML models and select the best performer.

**Training Process**:

1. **Model Initialization**: 8 different algorithms
2. **Hyperparameter Tuning**: GridSearchCV with 3-fold cross-validation
3. **Model Evaluation**: Train/test accuracy comparison
4. **Best Model Selection**: Highest test accuracy
5. **Serialization**: Save best model for deployment

**Hyperparameter Grids**:

```python
params = {
    'RandomForestClassifier': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'GradientBoostingClassifier': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    # ... additional models
}
```

**Performance Metrics**:
- Accuracy
- F1 Score (weighted)
- Precision (weighted)
- Recall (weighted)

**Configuration**:
```python
@dataclass(frozen=True)
class ModelTrainerConfig:
    trained_model_file_path: str = 'artifacts/model.pkl'
```

---

### 4. Prediction Pipeline (`src/pipeline/predict_pipeline.py`)

**Purpose**: Load trained models and make real-time predictions.

**Components**:

**a) PredictPipeline Class**:
- Loads preprocessor and model from artifacts
- Transforms input data
- Returns predictions

**b) CustomData Class**:
- Encapsulates student information
- Converts data to DataFrame format
- Validates input data

**Usage Example**:
```python
# Create student data object
student = CustomData(
    gender='male',
    race_ethnicity='group C',
    parental_level_of_education='some college',
    lunch='standard',
    test_preparation_course='none',
    reading_score=67,
    writing_score=70
)

# Convert to DataFrame
df = student.get_data_as_dataframe()

# Make prediction
pipeline = PredictPipeline()
prediction = pipeline.predict(df)
print(f"Predicted Math Score: {prediction[0]}")
```

---

## 🤖 Model Training

### Evaluation Metrics

The `evaluate_models` function in `src/utils.py` performs comprehensive model evaluation:

```python
def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    report = {}

    for model_name, model in models.items():
        # GridSearchCV for hyperparameter tuning
        gs = GridSearchCV(model, param_grid=params[model_name], cv=3)
        gs.fit(X_train, y_train)

        # Train with best parameters
        model.set_params(**gs.best_params_)
        model.fit(X_train, y_train)

        # Evaluate
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)

        report[model_name] = {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc
        }

    return report
```

### Model Selection Criteria

- **Minimum Threshold**: 60% accuracy
- **Selection Metric**: Test set accuracy
- **Validation**: 3-fold cross-validation during GridSearchCV
- **Overfitting Check**: Comparison of train vs. test accuracy

---

## 📡 API Reference

### Base URL
```
http://localhost:5000
```

### Endpoints

#### 1. Home Page
```http
GET /
```

**Response**: HTML landing page

---

#### 2. Prediction Page
```http
GET /predict
```

**Response**: HTML form for student data input

---

#### 3. Make Prediction
```http
POST /predict
```

**Request Body** (Form Data):
```json
{
  "gender": "male|female",
  "ethnicity": "group A|group B|group C|group D|group E",
  "parental_level_of_education": "associate's degree|bachelor's degree|high school|master's degree|some college|some high school",
  "lunch": "free/reduced|standard",
  "test_preparation_course": "none|completed",
  "reading_score": 0-100,
  "writing_score": 0-100
}
```

**Response**: HTML page with predicted math score

**Example cURL**:
```bash
curl -X POST http://localhost:5000/predict \
  -d "gender=female" \
  -d "ethnicity=group B" \
  -d "parental_level_of_education=bachelor's degree" \
  -d "lunch=standard" \
  -d "test_preparation_course=completed" \
  -d "reading_score=72" \
  -d "writing_score=74"
```

---

## 📁 Project Structure

```
mlproject/
│
├── application.py              # Flask application entry point
├── setup.py                    # Package installation configuration
├── requirements.txt            # Project dependencies
├── README.md                   # This documentation
│
├── src/                        # Source code directory
│   ├── __init__.py
│   │
│   ├── component/              # ML pipeline components
│   │   ├── __init__.py
│   │   ├── data_ingestion.py          # Data loading and splitting
│   │   ├── data_transformation.py     # Feature engineering
│   │   └── model_trainer.py           # Model training and selection
│   │
│   ├── pipeline/               # Prediction pipelines
│   │   ├── __init__.py
│   │   ├── train_pipeline.py          # Training orchestration
│   │   └── predict_pipeline.py        # Prediction interface
│   │
│   ├── exception.py            # Custom exception handling
│   ├── logger.py               # Logging configuration
│   └── utils.py                # Utility functions
│
├── templates/                  # HTML templates
│   ├── index.html              # Landing page
│   └── home.html               # Prediction form
│
├── artifacts/                  # Generated artifacts (created during runtime)
│   ├── data.csv                # Raw data
│   ├── train.csv               # Training set
│   ├── test.csv                # Testing set
│   ├── preprocessor.pkl        # Trained preprocessor
│   └── model.pkl               # Trained model
│
├── logs/                       # Application logs (created during runtime)
│   └── MM_DD_YYYY_HH_MM_SS.log
│
└── research/                   # Research and notebooks
    └── data/
        └── data.csv            # Source dataset
```

---

## ⚙️ Configuration

### Data Configuration

**Ingestion Settings** (`src/component/data_ingestion.py`):
```python
train_data_path = 'artifacts/train.csv'      # 80% of data
test_data_path = 'artifacts/test.csv'        # 20% of data
raw_data_path = 'artifacts/data.csv'         # Original dataset
```

**Transformation Settings** (`src/component/data_transformation.py`):
```python
preprocessor_obj_file_path = 'artifacts/preprocessor.pkl'

numerical_columns = ['writing_score', 'reading_score']
categorical_columns = [
    'gender',
    'race_ethnicity',
    'parental_level_of_education',
    'lunch',
    'test_preparation_course'
]
```

**Model Settings** (`src/component/model_trainer.py`):
```python
trained_model_file_path = 'artifacts/model.pkl'
min_accuracy_threshold = 0.6
```

### Flask Configuration

**Server Settings** (`application.py`):
```python
host = '0.0.0.0'    # Accessible from network
port = 5000         # Default Flask port
debug = False       # Set to True for development
```

---

## 📊 Logging & Exception Handling

### Logging System

**Configuration** (`src/logger.py`):

```python
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
```

**Log Levels**:
- **INFO**: Normal operations, pipeline progress
- **ERROR**: Exceptions and failures
- **DEBUG**: Detailed diagnostic information

**Example Log Output**:
```
[2025-10-14 18:45:32] 28 src.component.data_ingestion - INFO - Entered the data ingestion method
[2025-10-14 18:45:33] 31 src.component.data_ingestion - INFO - Read the dataset as dataframe
[2025-10-14 18:45:34] 43 src.component.data_ingestion - INFO - Ingestion of the data is completed
```

### Custom Exception Handling

**CustomException Class** (`src/exception.py`):

```python
class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message
```

**Exception Format**:
```
Error occurred in python script name [filename.py] at line number [X] due to: [error message]
```

**Usage**:
```python
try:
    # Your code
    result = risky_operation()
except Exception as e:
    raise CustomException(e, sys)
```

---

## 🧪 Testing

### Manual Testing

1. **Test Data Ingestion**:
```python
from src.component.data_ingestion import DataIngestion

ingestion = DataIngestion()
train_path, test_path = ingestion.initiate_data_ingestion()
print(f"Train data: {train_path}\nTest data: {test_path}")
```

2. **Test Data Transformation**:
```python
from src.component.data_transformation import DataTransformation

transformation = DataTransformation()
train_arr, test_arr, _ = transformation.initiate_data_transformation(
    train_path, test_path
)
print(f"Transformed data shape: {train_arr.shape}")
```

3. **Test Model Training**:
```python
from src.component.model_trainer import ModelTrainer

trainer = ModelTrainer()
metrics = trainer.initiate_model_trainer(train_arr, test_arr)
print(f"Model Performance: {metrics}")
```

4. **Test Prediction**:
```python
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

student = CustomData(
    gender='male',
    race_ethnicity='group B',
    parental_level_of_education='some college',
    lunch='standard',
    test_preparation_course='completed',
    reading_score=80,
    writing_score=85
)

pipeline = PredictPipeline()
prediction = pipeline.predict(student.get_data_as_dataframe())
print(f"Predicted Score: {prediction[0]}")
```

### Web Application Testing

1. Start the Flask server
2. Navigate to `http://localhost:5000`
3. Test the landing page
4. Fill the prediction form with various inputs
5. Verify predictions are returned correctly

---

## 🔄 Workflow

### Complete Training Pipeline

```python
"""
Complete end-to-end training workflow
"""

from src.component.data_ingestion import DataIngestion
from src.component.data_transformation import DataTransformation
from src.component.model_trainer import ModelTrainer
from src.logger import logging

def train_model():
    """Execute the complete training pipeline"""

    try:
        # Step 1: Data Ingestion
        logging.info("Starting data ingestion")
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
        logging.info("Data ingestion completed")

        # Step 2: Data Transformation
        logging.info("Starting data transformation")
        data_transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(
            train_data_path, test_data_path
        )
        logging.info("Data transformation completed")

        # Step 3: Model Training
        logging.info("Starting model training")
        model_trainer = ModelTrainer()
        metrics = model_trainer.initiate_model_trainer(train_arr, test_arr)
        logging.info("Model training completed")

        print("\n" + "="*50)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("="*50)
        print(f"\nBest Model: {metrics['model_name']}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print("\n" + "="*50)

        return metrics

    except Exception as e:
        logging.error(f"Training pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    train_model()
```

---

## 🎨 User Interface

### Landing Page Features

- **Modern Design**: Gradient background with floating animations
- **Responsive Layout**: Works on desktop, tablet, and mobile
- **Interactive Elements**: Hover effects and smooth transitions
- **Clear CTA**: Prominent call-to-action button
- **Feature Cards**: Highlights key capabilities
- **Statistics**: Display accuracy and usage metrics

### Prediction Form Features

- **Progress Bar**: Visual feedback on form completion
- **Input Validation**: Real-time validation with error feedback
- **Icon Labels**: Clear field identification
- **Dropdown Menus**: Easy selection for categorical inputs
- **Number Inputs**: Constrained ranges for scores (0-100)
- **Instant Results**: Prediction displayed immediately after submission

---

## 📈 Performance Optimization

### Best Practices Implemented

1. **Efficient Data Processing**:
   - Vectorized operations with NumPy/Pandas
   - Sparse matrix handling for one-hot encoding
   - Minimal data copying

2. **Model Optimization**:
   - GridSearchCV for hyperparameter tuning
   - Cross-validation to prevent overfitting
   - Model comparison for best selection

3. **Code Efficiency**:
   - Modular components for reusability
   - Lazy loading of models
   - Cached preprocessor and model objects

4. **Error Handling**:
   - Try-except blocks at critical points
   - Detailed error messages for debugging
   - Graceful failure handling

---

## 🚧 Troubleshooting

### Common Issues and Solutions

#### 1. Import Errors
```
ModuleNotFoundError: No module named 'src'
```
**Solution**: Install the package in editable mode
```bash
pip install -e .
```

#### 2. Model Not Found
```
FileNotFoundError: artifacts/model.pkl not found
```
**Solution**: Train the model first
```python
from src.component.data_ingestion import DataIngestion
from src.component.data_transformation import DataTransformation
from src.component.model_trainer import ModelTrainer

# Run complete pipeline
# ... (see Workflow section)
```

#### 3. Flask Port Already in Use
```
OSError: [Errno 48] Address already in use
```
**Solution**: Kill the process or change port
```bash
# Kill process on port 5000
lsof -ti:5000 | xargs kill -9

# Or change port in application.py
application.run(host='0.0.0.0', port=8080)
```

#### 4. Prediction Errors
```
ValueError: X has different shape
```
**Solution**: Ensure all required features are provided and match training data format

---

## 🔮 Future Enhancements

### Planned Features

- [ ] **Model Versioning**: Track and compare multiple model versions
- [ ] **A/B Testing**: Deploy multiple models simultaneously
- [ ] **Real-time Monitoring**: Dashboard for prediction analytics
- [ ] **Batch Predictions**: API endpoint for bulk predictions
- [ ] **Model Explainability**: SHAP values for feature importance
- [ ] **Docker Deployment**: Containerized application
- [ ] **CI/CD Pipeline**: Automated testing and deployment
- [ ] **Database Integration**: Store predictions and user data
- [ ] **Authentication**: User login and session management
- [ ] **Advanced UI**: React/Vue.js frontend

### Potential Improvements

- Implement ensemble methods
- Add deep learning models (Neural Networks)
- Feature engineering automation
- Automated data drift detection
- Multi-target prediction (reading, writing scores)

---

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

### How to Contribute

1. **Fork the Repository**
   ```bash
   git clone https://github.com/OTE22/MLPROJECT.git.git
   ```

2. **Create a Feature Branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```

3. **Make Changes**
   - Write clean, documented code
   - Follow PEP 8 style guidelines
   - Add tests for new features

4. **Commit Changes**
   ```bash
   git commit -m "Add amazing feature"
   ```

5. **Push to Branch**
   ```bash
   git push origin feature/amazing-feature
   ```

6. **Open Pull Request**
   - Describe changes clearly
   - Reference any related issues

### Code Style

- Follow PEP 8
- Use type hints where applicable
- Write docstrings for all functions/classes
- Keep functions focused and small
- Use meaningful variable names

### Testing Requirements

- All new features must include tests
- Maintain >80% code coverage
- Ensure all tests pass before submitting PR

---

## 📄 License

This project is licensed under the MIT License - see below for details:

```
MIT License

Copyright (c) 2025 Ali Abbass

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 📞 Contact

**Author**: Ali Abbass
**Email**: ali@gmail.com
**Project Link**: [https://github.com/OTE22/MLPROJECT.git](https://github.com/OTE22/MLPROJECT.git)

### Support

For questions, issues, or suggestions:

- 📧 Email: ali@gmail.com
- 🐛 Issues: [GitHub Issues](https://github.com/OTE22/MLPROJECT.git/issues)
- 💬 Discussions: [GitHub Discussions](https://github/discussions)

---

## 🙏 Acknowledgments

- **scikit-learn**: For comprehensive ML algorithms and tools
- **Flask**: For the lightweight web framework
- **XGBoost & CatBoost**: For advanced gradient boosting implementations
- **The Open Source Community**: For continuous inspiration and support

---

## 📚 Additional Resources

### Documentation
- [scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

### Tutorials
- [End-to-End ML Projects](https://www.youtube.com/results?search_query=end+to+end+ml+project)
- [Flask for Machine Learning](https://www.youtube.com/results?search_query=flask+machine+learning)
- [scikit-learn Tutorials](https://scikit-learn.org/stable/tutorial/index.html)

---

<div align="center">

**⭐ If you find this project useful, please consider giving it a star! ⭐**

Made with ❤️ by Ali Abbass

</div>
