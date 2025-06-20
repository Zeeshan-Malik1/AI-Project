import os
import io
import base64
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, adjusted_rand_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

app = Flask(__name__)

# Configuration
SEQUENCE_LENGTH = 57
NUCLEOTIDES = ['a', 'c', 'g', 't']

class PromoterClassifier:
    def __init__(self):
        self.models = {}
        self.ensemble_model = None
        self.label_encoder = LabelEncoder()
        self.kmeans = None
        self.pca = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.train_df = None
        self.cv_scores = {}
        
    def one_hot_encode(self, sequences):
        """Convert DNA sequences to one-hot encoded arrays"""
        # Ensure all sequences are exactly 57 characters
        sequences = [str(s)[:SEQUENCE_LENGTH].ljust(SEQUENCE_LENGTH, 'n') for s in sequences]
        
        # Convert to list of characters
        seq_arrays = [list(seq.lower()) for seq in sequences]
        
        # One-hot encode
        encoder = OneHotEncoder(categories=[NUCLEOTIDES], sparse_output=False, handle_unknown='ignore')
        encoded_seqs = []
        
        for seq in seq_arrays:
            # Replace unknown nucleotides with 'a'
            seq = [nuc if nuc in NUCLEOTIDES else 'a' for nuc in seq]
            seq_encoded = encoder.fit_transform(np.array(seq).reshape(-1, 1))
            encoded_seqs.append(seq_encoded.flatten())
            
        return np.array(encoded_seqs)
    
    def analyze_nucleotide_composition(self, sequences, labels):
        """Analyze nucleotide composition for different classes"""
        composition_data = []
        
        for seq, label in zip(sequences, labels):
            seq_lower = seq.lower()
            total_len = len(seq_lower)
            
            composition = {
                'label': label,
                'A_content': seq_lower.count('a') / total_len * 100,
                'T_content': seq_lower.count('t') / total_len * 100,
                'G_content': seq_lower.count('g') / total_len * 100,
                'C_content': seq_lower.count('c') / total_len * 100,
                'GC_content': (seq_lower.count('g') + seq_lower.count('c')) / total_len * 100,
                'AT_content': (seq_lower.count('a') + seq_lower.count('t')) / total_len * 100
            }
            composition_data.append(composition)
        
        return pd.DataFrame(composition_data)
    
    def find_motifs(self, sequences, labels, motif_length=6):
        """Find common motifs in promoter vs non-promoter sequences"""
        promoter_seqs = [seq for seq, label in zip(sequences, labels) if label == '+']
        non_promoter_seqs = [seq for seq, label in zip(sequences, labels) if label == '-']
        
        def extract_motifs(seqs, length):
            motifs = []
            for seq in seqs:
                for i in range(len(seq) - length + 1):
                    motifs.append(seq[i:i+length].upper())
            return Counter(motifs)
        
        promoter_motifs = extract_motifs(promoter_seqs, motif_length)
        non_promoter_motifs = extract_motifs(non_promoter_seqs, motif_length)
        
        # Find enriched motifs (appear more in promoters)
        enriched_motifs = {}
        for motif, count in promoter_motifs.most_common(20):
            promoter_freq = count / len(promoter_seqs)
            non_promoter_freq = non_promoter_motifs.get(motif, 0) / len(non_promoter_seqs)
            if promoter_freq > non_promoter_freq * 1.5:  # At least 1.5x more frequent
                enriched_motifs[motif] = {
                    'promoter_freq': promoter_freq,
                    'non_promoter_freq': non_promoter_freq,
                    'enrichment': promoter_freq / (non_promoter_freq + 1e-10)
                }
        
        return dict(sorted(enriched_motifs.items(), key=lambda x: x[1]['enrichment'], reverse=True)[:10])
    
    def get_feature_importance(self):
        """Get feature importance from Random Forest"""
        if 'Random Forest' in self.models:
            rf_model = self.models['Random Forest']
            importances = rf_model.feature_importances_
            
            # Map back to nucleotide positions
            position_importance = []
            for pos in range(SEQUENCE_LENGTH):
                pos_importance = np.sum(importances[pos*4:(pos+1)*4])
                position_importance.append(pos_importance)
            
            return position_importance
        return None
    
    def load_data(self, file_path):
        """Load and preprocess promoter data"""
        try:
            # Read the data file
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            data = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split(',')
                    if len(parts) >= 3:
                        label = parts[0].strip()
                        name = parts[1].strip()
                        sequence = parts[2].strip()
                        data.append([label, name, sequence])
            
            self.train_df = pd.DataFrame(data, columns=['label', 'name', 'sequence'])
            self.train_df['sequence'] = self.train_df['sequence'].str.lower()
            
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def train_models(self):
        """Train all ML models with cross-validation"""
        # Prepare features and labels
        X = self.one_hot_encode(self.train_df['sequence'])
        y = self.label_encoder.fit_transform(self.train_df['label'])
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Train individual classifiers
        self.models['Random Forest'] = RandomForestClassifier(n_estimators=100, random_state=42)
        self.models['Naive Bayes'] = MultinomialNB()
        self.models['SVM'] = SVC(kernel='rbf', probability=True, random_state=42)
        
        # Cross-validation scores
        for name, model in self.models.items():
            model.fit(self.X_train, self.y_train)
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='accuracy')
            self.cv_scores[name] = {
                'mean': cv_scores.mean(),
                'std': cv_scores.std(),
                'scores': cv_scores.tolist()
            }
        
        # Create ensemble model
        self.ensemble_model = VotingClassifier(
            estimators=[
                ('rf', self.models['Random Forest']),
                ('nb', self.models['Naive Bayes']),
                ('svm', self.models['SVM'])
            ],
            voting='soft'
        )
        self.ensemble_model.fit(self.X_train, self.y_train)
        
        # Add ensemble to models
        self.models['Ensemble'] = self.ensemble_model
        cv_scores = cross_val_score(self.ensemble_model, self.X_train, self.y_train, cv=5, scoring='accuracy')
        self.cv_scores['Ensemble'] = {
            'mean': cv_scores.mean(),
            'std': cv_scores.std(),
            'scores': cv_scores.tolist()
        }
        
        # Train clustering
        n_clusters = len(np.unique(y))
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.kmeans.fit(X)
        
        # PCA for visualization
        self.pca = PCA(n_components=2)
        self.pca.fit(X)
    
    def get_model_performance(self):
        """Get performance metrics for all models"""
        results = {}
        
        for name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            report = classification_report(
                self.y_test, y_pred, 
                target_names=self.label_encoder.classes_, 
                output_dict=True, zero_division=0
            )
            
            # Add cross-validation scores
            cv_info = self.cv_scores.get(name, {})
            
            results[name] = {
                'accuracy': accuracy,
                'report': report,
                'cv_mean': cv_info.get('mean', 0),
                'cv_std': cv_info.get('std', 0)
            }
        
        return results
    
    def predict_sequence(self, sequence):
        """Predict class for a given sequence with confidence"""
        if len(sequence) != SEQUENCE_LENGTH:
            return {"error": f"Sequence must be exactly {SEQUENCE_LENGTH} characters long"}
        
        if not all(c.lower() in NUCLEOTIDES for c in sequence):
            return {"error": "Sequence must contain only A, C, G, T nucleotides"}
        
        try:
            X_user = self.one_hot_encode([sequence])
            
            predictions = {}
            probabilities = {}
            confidences = {}
            
            for name, model in self.models.items():
                pred = model.predict(X_user)[0]
                predictions[name] = self.label_encoder.inverse_transform([pred])[0]
                
                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(X_user)[0]
                    probabilities[name] = {
                        self.label_encoder.classes_[i]: round(prob[i], 3) 
                        for i in range(len(prob))
                    }
                    # Confidence as max probability
                    confidences[name] = round(max(prob), 3)
            
            # Get cluster
            cluster = self.kmeans.predict(X_user)[0]
            
            # Analyze sequence composition
            seq_composition = self.analyze_nucleotide_composition([sequence], ['+'])
            
            return {
                'predictions': predictions,
                'probabilities': probabilities,
                'confidences': confidences,
                'cluster': int(cluster),
                'composition': seq_composition.iloc[0].to_dict()
            }
            
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}
    
    def predict_batch(self, sequences):
        """Predict multiple sequences at once"""
        results = []
        for i, seq in enumerate(sequences):
            result = self.predict_sequence(seq.strip())
            result['sequence_id'] = i + 1
            result['sequence'] = seq.strip()
            results.append(result)
        return results
    
    def create_visualizations(self):
        """Create enhanced visualization plots"""
        plots = {}
        
        # Set the style
        plt.style.use('default')
        
        # 1. Enhanced cluster plot with better styling
        X_all = self.one_hot_encode(self.train_df['sequence'])
        y_all = self.label_encoder.transform(self.train_df['label'])
        cluster_labels = self.kmeans.predict(X_all)
        
        X_pca = self.pca.transform(X_all)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot by clusters
        scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, 
                                 cmap='viridis', alpha=0.7, s=30)
        axes[0].set_title('K-Means Clustering (PCA Visualization)', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('PC1')
        axes[0].set_ylabel('PC2')
        plt.colorbar(scatter1, ax=axes[0])
        
        # Plot by true labels
        colors = ['#e74c3c' if label == '+' else '#3498db' for label in self.train_df['label']]
        axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.7, s=30)
        axes[1].set_title('True Labels (Red: Promoter, Blue: Non-Promoter)', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('PC1')
        axes[1].set_ylabel('PC2')
        
        plt.tight_layout()
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        plots['cluster_plot'] = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        # 2. Performance comparison with cross-validation
        performance = self.get_model_performance()
        model_names = list(performance.keys())
        accuracies = [performance[name]['accuracy'] for name in model_names]
        cv_means = [performance[name]['cv_mean'] for name in model_names]
        cv_stds = [performance[name]['cv_std'] for name in model_names]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Test accuracy
        bars1 = ax1.bar(model_names, accuracies, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'])
        ax1.set_title('Test Set Accuracy', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        
        for bar, acc in zip(bars1, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Cross-validation accuracy with error bars
        bars2 = ax2.bar(model_names, cv_means, yerr=cv_stds, 
                       color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'],
                       capsize=5)
        ax2.set_title('Cross-Validation Accuracy (5-fold)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Accuracy')
        ax2.set_ylim(0, 1)
        
        for bar, mean in zip(bars2, cv_means):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{mean:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        plots['performance_plot'] = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        # 3. Confusion matrices
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for idx, (name, model) in enumerate(self.models.items()):
            y_pred = model.predict(self.X_test)
            cm = confusion_matrix(self.y_test, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Non-Promoter', 'Promoter'],
                       yticklabels=['Non-Promoter', 'Promoter'],
                       ax=axes[idx])
            axes[idx].set_title(f'{name} Confusion Matrix', fontweight='bold')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
        
        plt.tight_layout()
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        plots['confusion_matrix'] = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        # 4. Feature importance (nucleotide positions)
        importance = self.get_feature_importance()
        if importance:
            plt.figure(figsize=(12, 6))
            
            positions = list(range(1, SEQUENCE_LENGTH + 1))
            bars = plt.bar(positions, importance, color='#3498db', alpha=0.7)
            
            plt.title('Feature Importance by Nucleotide Position', fontsize=14, fontweight='bold')
            plt.xlabel('Nucleotide Position')
            plt.ylabel('Importance Score')
            plt.xticks(range(1, SEQUENCE_LENGTH + 1, 5))
            
            # Highlight top 5 positions
            top_positions = sorted(enumerate(importance), key=lambda x: x[1], reverse=True)[:5]
            for pos, imp in top_positions:
                bars[pos].set_color('#e74c3c')
            
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            plots['feature_importance'] = base64.b64encode(img_buffer.getvalue()).decode()
            plt.close()
        
        # 5. Nucleotide composition analysis
        composition_df = self.analyze_nucleotide_composition(
            self.train_df['sequence'], self.train_df['label']
        )
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # GC content comparison
        promoter_gc = composition_df[composition_df['label'] == '+']['GC_content']
        non_promoter_gc = composition_df[composition_df['label'] == '-']['GC_content']
        
        axes[0,0].hist(promoter_gc, bins=20, alpha=0.7, label='Promoter', color='#e74c3c')
        axes[0,0].hist(non_promoter_gc, bins=20, alpha=0.7, label='Non-Promoter', color='#3498db')
        axes[0,0].set_title('GC Content Distribution', fontweight='bold')
        axes[0,0].set_xlabel('GC Content (%)')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].legend()
        
        # Individual nucleotide content
        nucleotides = ['A_content', 'T_content', 'G_content', 'C_content']
        for i, nuc in enumerate(nucleotides):
            ax = axes[0,1] if i < 2 else axes[1, i-2]
            
            promoter_data = composition_df[composition_df['label'] == '+'][nuc]
            non_promoter_data = composition_df[composition_df['label'] == '-'][nuc]
            
            ax.boxplot([promoter_data, non_promoter_data], 
                      labels=['Promoter', 'Non-Promoter'])
            ax.set_title(f'{nuc.split("_")[0]} Content Comparison', fontweight='bold')
            ax.set_ylabel('Content (%)')
        
        plt.tight_layout()
        
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        plots['composition_analysis'] = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return plots

# Initialize classifier
classifier = PromoterClassifier()

@app.route('/', methods=['GET', 'POST'])
def index():
    # Load data if not already loaded
    if classifier.train_df is None:
        data_path = os.path.join('promotor_genes', 'promoters.data')
        if not os.path.exists(data_path):
            return render_template('index.html', error="Data file not found!")
        
        if not classifier.load_data(data_path):
            return render_template('index.html', error="Failed to load data!")
        
        classifier.train_models()
    
    # Get dataset statistics
    result = {
        'total_sequences': len(classifier.train_df),
        'labels': list(classifier.label_encoder.classes_),
        'clustering_score': adjusted_rand_score(
            classifier.label_encoder.transform(classifier.train_df['label']),
            classifier.kmeans.predict(classifier.one_hot_encode(classifier.train_df['sequence']))
        )
    }
    
    # Add composition analysis
    composition_df = classifier.analyze_nucleotide_composition(
        classifier.train_df['sequence'], classifier.train_df['label']
    )
    
    result['composition_stats'] = {
        'promoter_gc_mean': composition_df[composition_df['label'] == '+']['GC_content'].mean(),
        'non_promoter_gc_mean': composition_df[composition_df['label'] == '-']['GC_content'].mean()
    }
    
    # Find enriched motifs
    motifs = classifier.find_motifs(classifier.train_df['sequence'], classifier.train_df['label'])
    result['top_motifs'] = motifs
    
    user_prediction = None
    user_sequence = ""
    show_results = False
    
    if request.method == 'POST':
        user_sequence = request.form.get('user_sequence', '').strip()
        if user_sequence:
            user_prediction = classifier.predict_sequence(user_sequence)
            show_results = True
    
    # Get model performance
    performance = classifier.get_model_performance()
    
    # Create visualizations
    plots = classifier.create_visualizations()
    
    return render_template(
        'index.html',
        result=result,
        performance=performance,
        user_prediction=user_prediction,
        user_sequence=user_sequence,
        show_results=show_results,
        plots=plots
    )

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Handle batch prediction of multiple sequences"""
    try:
        # Check if data is loaded and train if needed
        if classifier.train_df is None:
            data_path = os.path.join('promotor_genes', 'promoters.data')
            if not os.path.exists(data_path):
                return jsonify({'error': 'Data file not found!'})
            
            if not classifier.load_data(data_path):
                return jsonify({'error': 'Failed to load data!'})
            
            classifier.train_models()
        
        sequences = request.form.get('sequences', '').strip().split('\n')
        sequences = [seq.strip() for seq in sequences if seq.strip()]
        
        if not sequences:
            return jsonify({'error': 'No sequences provided'})
        
        if len(sequences) > 50:  # Limit batch size
            return jsonify({'error': 'Maximum 50 sequences allowed'})
        
        results = classifier.predict_batch(sequences)
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': f'Batch prediction failed: {str(e)}'})


# Updated predict_batch method for the PromoterClassifier class
def predict_batch_updated(self, sequences):
    """Predict multiple sequences at once"""
    results = []
    for i, seq in enumerate(sequences):
        result = self.predict_sequence(seq.strip())
        result['sequence_id'] = i + 1
        result['sequence'] = seq.strip()
        results.append(result)
    return results

@app.route('/examples')
def examples():
    """Show some example sequences"""
    if classifier.train_df is None:
        return "No data loaded"
    
    examples = []
    for label in ['+', '-']:
        sample = classifier.train_df[classifier.train_df['label'] == label].head(3)
        for _, row in sample.iterrows():
            examples.append({
                'label': 'Promoter' if row['label'] == '+' else 'Non-Promoter',
                'name': row['name'],
                'sequence': row['sequence'].upper()
            })
    
    return render_template('examples.html', examples=examples)

if __name__ == '__main__':
    app.run(debug=True)