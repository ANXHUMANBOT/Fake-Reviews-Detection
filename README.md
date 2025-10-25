🛡️ Fake Product Review Detector 🛡️

A real-time web app to detect and flag deceptive product reviews using natural language processing and machine learning.

✨ Core Features

Real-Time Classification: Instantly classifies product reviews as "Genuine" or "Fake".

Confidence Score: Provides a percentage to show how confident the model is in its decision.

Graffiti Theme UI: A funky, dark-mode interface for a better user experience.

ML Backend: Powered by a Logistic Regression model trained on 40,000+ reviews using Scikit-learn and TF-IDF.

🚀 Tech Stack

Backend: Python, Flask

Machine Learning: Scikit-learn, Pandas, NLTK

Frontend: HTML5, CSS3, JavaScript

Deployment: (Local Flask Server)

🔧 Installation & Setup

Here's how to get the project running on your local machine.

1. Clone the Repository

Clone this project to your local machine.

git clone [https://github.com/YourUsername/Your-Repo-Name.git](https://github.com/YourUsername/Your-Repo-Name.git)
cd Your-Repo-Name


2. Create a Virtual Environment

It's highly recommended to use a virtual environment.

# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate


3. Install Dependencies

Install all the required Python libraries using the requirements.txt file.

pip install -r requirements.txt


4. (If needed) Train the Model

The trained models (.joblib files) are included, but if you want to retrain on new data, run the training script:

python train_model.py


5. Run the Web App

Start the Flask server!

python app.py


Open your browser and navigate to http://127.0.0.1:5000 to see the app in action!

📜 License

This project is open-source and distributed under the MIT License. See the LICENSE file for more details.
