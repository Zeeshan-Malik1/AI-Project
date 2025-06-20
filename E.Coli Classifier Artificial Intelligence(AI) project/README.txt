  E.coli Promoter Classifier

 AI project built using Python and Flask. It classifies **E.coli DNA sequences** into promoter or non-promoter categories using multiple machine learning models including Random Forest,Naive Bayes,SVM, and ensemble classifier . The project also performs biological analysis n9-like motif detection,GC content analysis, and clustering visualization using PCA and K-Means.

  Features

- One-hot encoding of DNA sequences
- Machine learning models:
  -  Random Forest
  -  Naive Bayes
  -  Support Vector Machine (SVM)
  -  Ensemble (Voting Classifier)
- Cross-validation accuracy comparison
- PCA & K-Means clustering
- Nucleotide composition (GC, AT, A, T, G, C content)
- Motif discovery in promoters
- Web interface for user-friendly predictions
- Batch and single-sequence prediction
- Data visualizations

 Project Structure

Ecoli-Promoter-Classifier/
├── app.py                       # Flask application
├── promotor_genes/
│   ├── Index                    # Dataset index file
│   ├── promoters.data           # Sequence dataset
│   ├── promoters.names          # Feature names
│   └── promoters.theory         # Theory documentation
├── templates/
│   ├── index.html               # Main interface
│   └── examples.html            # Examples page
├── static/                      # (optional: for styling or images)
├── requirements.txt             # Python dependencies (create this)
└── README.md                    # Project documentation


 Setup Instructions

 1. Clone the Repository
 bash
git clone https://github.com/Zeeshan-Malik1/Ecoli-Promoter-Classifier.git
cd Ecoli-Promoter-Classifier

 3. Install Dependencies
Create a `requirements.txt` file and add:

 flask
 numpy
 pandas
 scikit-learn
 matplotlib
 seaborn


 Then run:
 bash
 pip install -r requirements.txt


 4. Run the Flask Application
 bash
 python app.py


 Visit: [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

  Input Format

 - Sequence must be 57 characters long
 - Valid characters: A, T, G, C
 - Example:

 TTGACAATCATCGTGTTAGGTCACGGTAGCTCGCAGACAGTGATCGTGTGATAAA


 Visualizations

 - PCA + K-Means cluster plot
 - Confusion Matrices
 - Cross-validation accuracy bars
 - Feature Importance plot
 - GC and AT content distribution
 - Motif enrichment analysis

 Technologies Used

 - Python 3
 - Flask
 - Pandas & NumPy
 - scikit-learn
 - Matplotlib & Seaborn
 - HTML & Jinja Templates

 License
 This project is licensed under the [MIT License](LICENSE).

 Author

 Zeeshan Malik  
 BSCS Student   
GitHub: [Zeeshan-Malik1](https://github.com/Zeeshan-Malik1)
