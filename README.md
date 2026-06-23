Credit Card Fraud Detection with Concept Drift (CCFD)



Overview



CCFD is a distributed fraud detection framework designed to handle concept drift in financial transaction data. The system combines fraud detection models, drift monitoring, adaptive retraining, and federated learning concepts to maintain model performance as data distributions evolve over time.



Features



\-> Fraud detection pipeline

\-> Concept drift detection and monitoring

\-> Adaptive model retraining

\-> Federated learning simulation

\-> Multi-bank client architecture

\-> Performance visualization and analytics

\-> REST API integration



&#x20;Project Structure



fraud-concept-drift/

│

├── clients/

├── server/

├── api\_server.py

├── frauddrift.py

├── drift\_detector.py

├── drift\_monitor.py

├── model\_manager.py

├── run\_all.py

├── requirements.txt

└── README.md



Dataset Download



The datasets are hosted externally to keep the GitHub repository lightweight.



Download Dataset



Download:



https://drive.google.com/file/d/1U-2ySc3mJEtV\_qIaOpMf2lmWk20ovJzi/view?usp=sharing



&#x20;Setup Dataset



After downloading:



1\. Extract the downloaded archive.

2\. Place the `clients` folder in the project root directory.



Expected structure:





fraud-concept-drift/

│

├── clients/

│   ├── bank1/

│   │   └── data.csv

│   ├── bank2/

│   │   └── data.csv

│   ├── bank3/

│   │   └── data.csv

│   ├── bank4/

│   │   └── data.csv

│   └── bank5/

│       └── data.csv





Installation



Clone the repository:





git clone https://github.com/chirumanipradyumnareddy/CCFD.git

cd CCFD





Create a virtual environment:





python -m venv venv





Activate the environment:



&#x20;Windows





venv\\Scripts\\activate





&#x20;Linux / macOS





source venv/bin/activate





Install dependencies:





pip install -r requirements.txt





Running the Project



Run the complete system:





python run\_all.py





Run the API server:





python api\_server.py





Technologies Used



\-> Python

\-> Machine Learning

\-> Federated Learning Concepts

\-> Concept Drift Detection

\-> REST APIs

\-> Data Analytics

\-> Pandas

\-> NumPy

\-> Scikit-learn



Applications



\-> Financial fraud detection

\-> Real-time risk assessment

\-> Adaptive machine learning systems

\-> Federated analytics environments

\-> Concept drift research



&#x20;Future Enhancements



\-> Real-time streaming support

\-> Cloud deployment

\-> Containerized microservices

\-> Advanced federated learning integration

\-> Automated model lifecycle management



&#x20;Author



Pradyumna Reddy



