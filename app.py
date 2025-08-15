"""
Food Quality Classifier - Main Application
A simple and clean web application for classifying food quality
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import json
from datetime import datetime
from food_classifier import FoodQualityClassifier
from utils import allowed_file, save_uploaded_file, get_model_performance

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize the food classifier
classifier = FoodQualityClassifier()

@app.route('/')
def index():
    """Main page with food quality classification interface"""
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify_food():
    """Handle food image classification"""
    try:
        # Check if file was uploaded
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check file type
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload JPG, PNG, or JPEG'}), 400
        
        # Get food type from form
        food_type = request.form.get('food_type', 'tomato')
        
        # Check if requested model is available (either loaded or can be loaded)
        if food_type not in classifier.model_paths:
            return jsonify({
                'error': f'Model for {food_type} not available',
                'available_models': list(classifier.model_paths.keys())
            }), 400
        
        # Save uploaded file
        filename = save_uploaded_file(file, app.config['UPLOAD_FOLDER'])
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Classify the image
        result = classifier.classify_image(file_path, food_type)
        
        # Add metadata
        result['timestamp'] = datetime.now().isoformat()
        result['filename'] = filename
        result['food_type'] = food_type
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'error': 'Classification failed',
            'details': str(e),
            'models_loaded': len(classifier.models) if hasattr(classifier, 'models') else 0,
            'available_models': len(classifier.model_paths) if hasattr(classifier, 'model_paths') else 0
        }), 500

@app.route('/models')
def models_info():
    """Get information about available models"""
    models_info = classifier.get_models_info()
    return jsonify(models_info)

@app.route('/performance')
def performance():
    """Get model performance metrics"""
    performance_data = get_model_performance()
    return jsonify(performance_data)

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy', 
        'timestamp': datetime.now().isoformat(),
        'models_loaded': len(classifier.models),
        'available_models': list(classifier.model_paths.keys()),
        'python_version': '3.13.4',
        'tensorflow_version': '2.20.0',
        'memory_optimization': 'lazy_loading_enabled'
    })

if __name__ == '__main__':
    print("🚀 Starting Food Quality Classifier...")
    print("📱 Web interface available at: http://localhost:5000")
    print("🔍 API endpoints:")
    print("   - GET  / - Main interface")
    print("   - POST /classify - Classify food image")
    print("   - GET  /models - Model information")
    print("   - GET  /performance - Performance metrics")
    print("   - GET  /health - Health check")
    
    # Get port from environment variable (for deployment)
    port = int(os.environ.get('PORT', 5000))
    
    app.run(debug=False, host='0.0.0.0', port=port)