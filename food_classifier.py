"""
Food Quality Classifier - Core Classification Logic
Handles loading models and classifying food images
"""

import os
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import json
from datetime import datetime

class FoodQualityClassifier:
    """
    Main class for food quality classification
    Supports multiple food types with individual trained models
    """
    
    def __init__(self):
        """Initialize the classifier and load all available models"""
        self.models = {}
        self.labels = ['average', 'bad', 'good']
        self.model_info = {}
        self.load_all_models()
    
    def load_all_models(self):
        """Load all available food classification models"""
        models_dir = 'models'
        
        if not os.path.exists(models_dir):
            print(f"⚠️  Models directory '{models_dir}' not found!")
            return
        
        # Get all available model directories
        available_models = [d for d in os.listdir(models_dir) 
                          if os.path.isdir(os.path.join(models_dir, d))]
        
        print(f"🔍 Found {len(available_models)} models: {available_models}")
        
        for model_name in available_models:
            model_path = os.path.join(models_dir, model_name)
            try:
                # Load TensorFlow model
                model = tf.keras.models.load_model(model_path)
                
                # Get model input shape
                input_shape = model.input_shape[1:3]  # (height, width)
                
                # Store model and metadata
                self.models[model_name] = {
                    'model': model,
                    'input_shape': input_shape,
                    'loaded_at': datetime.now().isoformat()
                }
                
                # Store model information
                self.model_info[model_name] = {
                    'input_shape': input_shape,
                    'num_classes': len(self.labels),
                    'labels': self.labels,
                    'status': 'loaded'
                }
                
                print(f"✅ Loaded {model_name} model (input: {input_shape})")
                
            except Exception as e:
                print(f"❌ Failed to load {model_name} model: {str(e)}")
                self.model_info[model_name] = {
                    'status': 'error',
                    'error': str(e)
                }
    
    def preprocess_image(self, image_path, target_size):
        """
        Preprocess image for model input
        
        Args:
            image_path (str): Path to the image file
            target_size (tuple): Target size (height, width)
        
        Returns:
            numpy.ndarray: Preprocessed image array
        """
        try:
            # Open and resize image
            with Image.open(image_path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize image to target size
                img = ImageOps.fit(img, target_size, Image.Resampling.LANCZOS)
                
                # Convert to numpy array and normalize
                img_array = np.array(img).astype(np.float32)
                normalized_array = (img_array / 127.0) - 1
                
                # Add batch dimension
                return np.expand_dims(normalized_array, axis=0)
                
        except Exception as e:
            raise Exception(f"Image preprocessing failed: {str(e)}")
    
    def classify_image(self, image_path, food_type):
        """
        Classify the quality of a food image
        
        Args:
            image_path (str): Path to the image file
            food_type (str): Type of food (e.g., 'tomato', 'apple')
        
        Returns:
            dict: Classification results with probabilities
        """
        # Check if model exists for the food type
        if food_type not in self.models:
            available_types = list(self.models.keys())
            raise Exception(f"Model for '{food_type}' not found. Available types: {available_types}")
        
        # Get model and input shape
        model_data = self.models[food_type]
        model = model_data['model']
        input_shape = model_data['input_shape']
        
        # Preprocess image
        processed_image = self.preprocess_image(image_path, input_shape)
        
        # Make prediction
        try:
            predictions = model.predict(processed_image, verbose=0)
            probabilities = predictions[0] * 100  # Convert to percentages
            
            # Create results dictionary
            results = {
                'food_type': food_type,
                'predictions': dict(zip(self.labels, probabilities.tolist())),
                'top_prediction': self.labels[np.argmax(probabilities)],
                'confidence': float(np.max(probabilities)),
                'all_probabilities': probabilities.tolist()
            }
            
            return results
            
        except Exception as e:
            raise Exception(f"Prediction failed: {str(e)}")
    
    def get_models_info(self):
        """Get information about all available models"""
        return {
            'available_models': list(self.models.keys()),
            'model_details': self.model_info,
            'total_models': len(self.models),
            'labels': self.labels
        }
    
    def get_model_status(self, food_type):
        """Get status of a specific model"""
        if food_type in self.model_info:
            return self.model_info[food_type]
        return {'status': 'not_found'}
    
    def reload_model(self, food_type):
        """Reload a specific model"""
        if food_type in self.models:
            try:
                model_path = os.path.join('models', food_type)
                model = tf.keras.models.load_model(model_path)
                input_shape = model.input_shape[1:3]
                
                self.models[food_type] = {
                    'model': model,
                    'input_shape': input_shape,
                    'loaded_at': datetime.now().isoformat()
                }
                
                self.model_info[food_type]['status'] = 'reloaded'
                self.model_info[food_type]['loaded_at'] = datetime.now().isoformat()
                
                return {'status': 'success', 'message': f'{food_type} model reloaded'}
            except Exception as e:
                return {'status': 'error', 'message': str(e)}
        
        return {'status': 'error', 'message': f'Model {food_type} not found'}

# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing Food Quality Classifier...")
    
    classifier = FoodQualityClassifier()
    
    # Print model information
    print("\n📊 Model Information:")
    for food_type, info in classifier.model_info.items():
        print(f"  {food_type}: {info['status']}")
    
    print(f"\n🎯 Available food types: {list(classifier.models.keys())}")
    print(f"🏷️  Quality labels: {classifier.labels}")
