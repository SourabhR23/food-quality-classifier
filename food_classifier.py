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
        """Load all food quality models using tf.saved_model.load"""
        model_paths = {
            'tomato': 'models/tomato',
            'mango': 'models/mango', 
            'potato': 'models/potato',
            'apple': 'models/apple'
        }
        
        for food_type, model_path in model_paths.items():
            try:
                if os.path.exists(model_path):
                    # Use tf.saved_model.load for TensorFlow 2.13.0 compatibility
                    model = tf.saved_model.load(model_path)
                    self.models[food_type] = model
                    print(f"✅ Loaded {food_type} model using saved_model.load")
                else:
                    print(f"❌ Model path not found: {model_path}")
            except Exception as e:
                print(f"❌ Failed to load {food_type} model: {e}")
        
        print(f"🔍 Found {len(self.models)} models: {list(self.models.keys())}")
    
    def preprocess_image(self, image_path, target_size=(300, 300)):
        """Preprocess image for model input"""
        try:
            # Load and resize image
            img = Image.open(image_path)
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
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
            
            # Make prediction using SavedModel
            # Convert numpy array to tensor
            input_tensor = tf.convert_to_tensor(processed_image, dtype=tf.float32)
            
            # Get the default serving signature
            infer = model.signatures['serving_default']
            
            # Get input key (first key in the signature)
            input_key = list(infer.structured_input_signature[1].keys())[0]
            
            # Make prediction
            prediction = infer(**{input_key: input_tensor})
            
            # Extract prediction values (get first output)
            output_key = list(prediction.keys())[0]
            prediction_values = prediction[output_key]
            
            # Convert to numpy array
            prediction_values = prediction_values.numpy()
            
            # Model outputs 3 classes: [Bad/Poor, Average, Good]
            class_names = ["Poor", "Average", "Good"]
            class_probabilities = prediction_values[0]
            
            # Get the predicted class (highest probability)
            predicted_class_idx = np.argmax(class_probabilities)
            quality = class_names[predicted_class_idx]
            confidence = float(class_probabilities[predicted_class_idx])
            
            # Also get the raw score for the predicted class
            quality_score = confidence
            
            # Create detailed probability breakdown
            probabilities = {
                'poor': round(float(class_probabilities[0]), 3),
                'average': round(float(class_probabilities[1]), 3), 
                'good': round(float(class_probabilities[2]), 3)
            }
            
            return {
                'quality': quality,
                'confidence': round(confidence, 3),
                'score': round(quality_score, 3),
                'food_type': food_type,
                'model_used': f"{food_type}_model",
                'probabilities': probabilities,
                'predicted_class_index': int(predicted_class_idx)
            }
            
        except Exception as e:
            raise Exception(f"Classification failed: {e}")
    
    def get_models_info(self):
        """Get information about loaded models"""
        models_info = {}
        for food_type, model in self.models.items():
            models_info[food_type] = {
                'status': 'loaded',
                'type': 'SavedModel',
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
                model = tf.saved_model.load(model_path)
                self.models[food_type] = model
                return {'status': 'success', 'message': f'{food_type} model reloaded'}
            else:
                return {'status': 'error', 'message': f'Model path not found: {model_path}'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}