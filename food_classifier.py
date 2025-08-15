import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

class FoodQualityClassifier:
    def __init__(self):
        self.models = {}
        self.load_all_models()
    
    def load_all_models(self):
        """Load all food quality models using TFSMLayer for TensorFlow 2.20.0"""
        model_paths = {
            'tomato': 'models/tomato',
            'mango': 'models/mango', 
            'potato': 'models/potato',
            'apple': 'models/apple'
        }
        
        for food_type, model_path in model_paths.items():
            try:
                if os.path.exists(model_path):
                    # Use TFSMLayer for TensorFlow 2.20.0 compatibility
                    model = keras.layers.TFSMLayer(model_path, call_endpoint='serving_default')
                    self.models[food_type] = model
                    print(f"✅ Loaded {food_type} model using TFSMLayer")
                else:
                    print(f"❌ Model path not found: {model_path}")
            except Exception as e:
                print(f"❌ Failed to load {food_type} model: {e}")
        
        print(f"🔍 Found {len(self.models)} models: {list(self.models.keys())}")
    
    def preprocess_image(self, image_path, target_size=(224, 224)):
        """Preprocess image for model input"""
        try:
            # Load and resize image
            img = Image.open(image_path)
            img = img.resize(target_size)
            
            # Convert to array and normalize
            img_array = np.array(img)
            img_array = img_array.astype('float32') / 255.0
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array
        except Exception as e:
            raise Exception(f"Image preprocessing failed: {e}")
    
    def classify_image(self, image_path, food_type):
        """Classify food quality using the specified model"""
        try:
            if food_type not in self.models:
                raise Exception(f"Model for {food_type} not available")
            
            # Preprocess image
            processed_image = self.preprocess_image(image_path)
            
            # Get model
            model = self.models[food_type]
            
            # Make prediction using TFSMLayer
            prediction = model(processed_image)
            
            # Extract prediction values (adjust based on your model output)
            if isinstance(prediction, dict):
                # If prediction is a dictionary, get the first value
                prediction_values = list(prediction.values())[0]
            else:
                prediction_values = prediction
            
            # Convert to numpy array if needed
            if hasattr(prediction_values, 'numpy'):
                prediction_values = prediction_values.numpy()
            
            # Get quality score (assuming single value output)
            quality_score = float(prediction_values[0][0])
            
            # Determine quality category
            if quality_score >= 0.7:
                quality = "Good"
                confidence = quality_score
            elif quality_score >= 0.4:
                quality = "Average"
                confidence = quality_score
            else:
                quality = "Poor"
                confidence = quality_score
            
            return {
                'quality': quality,
                'confidence': round(confidence, 3),
                'score': round(quality_score, 3),
                'food_type': food_type,
                'model_used': f"{food_type}_model"
            }
            
        except Exception as e:
            raise Exception(f"Classification failed: {e}")
    
    def get_models_info(self):
        """Get information about loaded models"""
        models_info = {}
        for food_type, model in self.models.items():
            models_info[food_type] = {
                'status': 'loaded',
                'type': 'TFSMLayer',
                'path': f'models/{food_type}'
            }
        return models_info
    
    def get_model_status(self):
        """Get overall model loading status"""
        return {
            'total_models': 4,
            'loaded_models': len(self.models),
            'available_models': list(self.models.keys()),
            'status': 'ready' if len(self.models) > 0 else 'failed'
        }
    
    def reload_model(self, food_type):
        """Reload a specific model"""
        try:
            model_path = f'models/{food_type}'
            if os.path.exists(model_path):
                model = keras.layers.TFSMLayer(model_path, call_endpoint='serving_default')
                self.models[food_type] = model
                return {'status': 'success', 'message': f'{food_type} model reloaded'}
            else:
                return {'status': 'error', 'message': f'Model path not found: {model_path}'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}