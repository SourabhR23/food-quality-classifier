import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

class FoodQualityClassifier:
    def __init__(self):
        self.models = {}
        self.model_paths = {
            'tomato': 'models/tomato',
            'mango': 'models/mango', 
            'potato': 'models/potato',
            'apple': 'models/apple'
        }
        # Use lazy loading to reduce memory usage
        print("🚀 Food Quality Classifier initialized with lazy loading")
    
    def load_all_models(self):
        """Load all food quality models with fallback loading strategies"""
        model_paths = {
            'tomato': 'models/tomato',
            'mango': 'models/mango', 
            'potato': 'models/potato',
            'apple': 'models/apple'
        }
        
        for food_type, model_path in model_paths.items():
            try:
                if os.path.exists(model_path):
                    # Try multiple loading strategies for compatibility
                    model = self._load_model_with_fallback(model_path, food_type)
                    if model is not None:
                        self.models[food_type] = model
                        print(f"✅ Loaded {food_type} model successfully")
                    else:
                        print(f"❌ Failed to load {food_type} model with all strategies")
                else:
                    print(f"❌ Model path not found: {model_path}")
            except Exception as e:
                print(f"❌ Failed to load {food_type} model: {e}")
        
        print(f"🔍 Found {len(self.models)} models: {list(self.models.keys())}")
    
    def _load_model_with_fallback(self, model_path, food_type):
        """Try multiple model loading strategies for compatibility"""
        
        # Strategy 1: Try keras.models.load_model first (most compatible)
        try:
            model = keras.models.load_model(model_path)
            print(f"✅ Loaded {food_type} using keras.models.load_model")
            return model
        except Exception as e1:
            print(f"⚠️ keras.models.load_model failed for {food_type}: {e1}")
        
        # Strategy 2: Try tf.saved_model.load
        try:
            model = tf.saved_model.load(model_path)
            print(f"✅ Loaded {food_type} using tf.saved_model.load")
            return model
        except Exception as e2:
            print(f"⚠️ tf.saved_model.load failed for {food_type}: {e2}")
        
        # Strategy 3: Try TFSMLayer for Keras 3 compatibility
        try:
            from tensorflow import keras
            model = keras.layers.TFSMLayer(model_path, call_endpoint='serving_default')
            print(f"✅ Loaded {food_type} using keras.layers.TFSMLayer")
            return model
        except Exception as e3:
            print(f"⚠️ keras.layers.TFSMLayer failed for {food_type}: {e3}")
        
        # Strategy 4: Try loading with custom objects (for compatibility)
        try:
            model = keras.models.load_model(model_path, compile=False)
            print(f"✅ Loaded {food_type} using keras.models.load_model (no compile)")
            return model
        except Exception as e4:
            print(f"⚠️ keras.models.load_model (no compile) failed for {food_type}: {e4}")
        
        return None
    
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
    
    def _load_model_if_needed(self, food_type):
        """Load model on demand to save memory"""
        if food_type not in self.models:
            if food_type in self.model_paths:
                model_path = self.model_paths[food_type]
                if os.path.exists(model_path):
                    print(f"📥 Loading {food_type} model on demand...")
                    model = self._load_model_with_fallback(model_path, food_type)
                    if model is not None:
                        self.models[food_type] = model
                        print(f"✅ Loaded {food_type} model successfully")
                        return model
                    else:
                        raise Exception(f"Failed to load {food_type} model with all strategies")
                else:
                    raise Exception(f"Model path not found: {model_path}")
            else:
                raise Exception(f"Model for {food_type} not configured")
        return self.models[food_type]

    def classify_image(self, image_path, food_type):
        """Classify food quality using the specified model"""
        try:
            # Load model on demand
            model = self._load_model_if_needed(food_type)
            
            # Preprocess image
            processed_image = self.preprocess_image(image_path)
            
            # Try different prediction strategies based on model type
            prediction_values = self._predict_with_model(model, processed_image, food_type)
            
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
    
    def _predict_with_model(self, model, processed_image, food_type):
        """Make prediction with model, handling different model types"""
        
        # Try Keras model prediction first
        try:
            if hasattr(model, 'predict'):
                prediction_values = model.predict(processed_image, verbose=0)
                print(f"✅ Used Keras model.predict() for {food_type}")
                return prediction_values
        except Exception as e1:
            print(f"⚠️ Keras model.predict() failed for {food_type}: {e1}")
        
        # Try TFSMLayer call (Keras 3 SavedModel wrapper)
        try:
            if hasattr(model, '__call__') and 'TFSMLayer' in str(type(model)):
                input_tensor = tf.convert_to_tensor(processed_image, dtype=tf.float32)
                prediction_values = model(input_tensor).numpy()
                print(f"✅ Used TFSMLayer call for {food_type}")
                return prediction_values
        except Exception as e2:
            print(f"⚠️ TFSMLayer call failed for {food_type}: {e2}")
        
        # Try SavedModel signature
        try:
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
            print(f"✅ Used SavedModel signature for {food_type}")
            return prediction_values
            
        except Exception as e3:
            print(f"⚠️ SavedModel signature failed for {food_type}: {e3}")
        
        # Try direct call (for some SavedModels)
        try:
            input_tensor = tf.convert_to_tensor(processed_image, dtype=tf.float32)
            prediction_values = model(input_tensor).numpy()
            print(f"✅ Used direct model call for {food_type}")
            return prediction_values
        except Exception as e4:
            print(f"⚠️ Direct model call failed for {food_type}: {e4}")
        
        raise Exception(f"All prediction strategies failed for {food_type}")
    
    def get_models_info(self):
        """Get information about available and loaded models"""
        models_info = {}
        for food_type in self.model_paths.keys():
            if food_type in self.models:
                models_info[food_type] = {
                    'status': 'loaded',
                    'type': 'Keras/SavedModel',
                    'path': self.model_paths[food_type]
                }
            else:
                models_info[food_type] = {
                    'status': 'available_lazy',
                    'type': 'Keras/SavedModel', 
                    'path': self.model_paths[food_type]
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
                model = self._load_model_with_fallback(model_path, food_type)
                if model is not None:
                    self.models[food_type] = model
                    return {'status': 'success', 'message': f'{food_type} model reloaded'}
                else:
                    return {'status': 'error', 'message': f'Failed to reload {food_type} model with all strategies'}
            else:
                return {'status': 'error', 'message': f'Model path not found: {model_path}'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}