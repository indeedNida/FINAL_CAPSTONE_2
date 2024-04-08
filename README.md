# Text Prediction Model

This repository contains a Flask web application that utilizes a deep learning model to predict subsequent text based on a given seed text. The application demonstrates the integration of TensorFlow models within a Flask framework, showcasing the capability to generate text predictions in real-time.

## Features

- Text prediction based on seed input
- Integration of TensorFlow models with Flask
- Real-time prediction with adjustable parameters

## Installation

To get this project up and running on your local machine, follow these steps:

1. **Clone the Repository**

    ```
    git clone https://github.com/<your-username>/Next-Word-Prediction-Model.git
    cd Next-Word-Prediction-Model
    ```

2. **Set Up a Virtual Environment** (Optional but recommended)

    - For Unix/macOS:

        ```
        python3 -m venv venv
        source venv/bin/activate
        ```

    - For Windows:

        ```
        python -m venv venv
        .\venv\Scripts\activate
        ```

3. **Install Required Packages**

    ```
    pip install -r requirements.txt
    ```

## Usage

To run the Flask application:

```
python run.py
```

The application will be accessible at `http://127.0.0.1:5000`.

## Model

The text prediction model is built using TensorFlow and Keras. It's designed to predict the next set of words based on a given seed text. The model is trained on a dataset of text and utilizes sequence models for prediction.

## API Endpoints

- `POST /generate`: Accepts seed text and other parameters to generate predicted text.

## Contributing

Contributions are welcome! For major changes, please open an issue first to discuss what you would like to change.

## Screenshots

<img width="940" alt="Screenshot 2024-04-08 at 1 08 16 PM" src="https://github.com/mayankhurana/Next-Word-Predictor/assets/146905878/dfa71b46-9d78-4efd-8b83-37d5b8417644">
<img width="488" alt="Screenshot 2024-04-08 at 1 08 30 PM" src="https://github.com/mayankhurana/Next-Word-Predictor/assets/146905878/4024e5dc-92ae-4183-878c-b8dc8d4e57c1">
<img width="241" alt="Screenshot 2024-04-08 at 1 08 42 PM" src="https://github.com/mayankhurana/Next-Word-Predictor/assets/146905878/62865be3-008b-463c-8128-90abe1148b56">

