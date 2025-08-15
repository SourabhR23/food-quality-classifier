"""
Utility functions for the Food Quality Classifier
Handles file operations, validation, and performance metrics
"""

import os
import uuid
from datetime import datetime
from werkzeug.utils import secure_filename

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    """
    Check if uploaded file has allowed extension
    
    Args:
        filename (str): Name of the uploaded file
    
    Returns:
        bool: True if file extension is allowed
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_uploaded_file(file, upload_folder):
    """
    Save uploaded file with unique filename
    
    Args:
        file: Uploaded file object
        upload_folder (str): Directory to save file
    
    Returns:
        str: Unique filename
    """
    # Generate unique filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    unique_id = str(uuid.uuid4())[:8]
    original_ext = file.filename.rsplit('.', 1)[1].lower()
    
    filename = f"{timestamp}_{unique_id}.{original_ext}"
    filepath = os.path.join(upload_folder, filename)
    
    # Save file
    file.save(filepath)
    
    return filename

def get_file_size(file_path):
    """
    Get file size in human-readable format
    
    Args:
        file_path (str): Path to the file
    
    Returns:
        str: File size in human-readable format
    """
    size_bytes = os.path.getsize(file_path)
    
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    
    return f"{size_bytes:.1f} TB"

def cleanup_old_files(upload_folder, max_age_hours=24):
    """
    Clean up old uploaded files
    
    Args:
        upload_folder (str): Directory containing uploaded files
        max_age_hours (int): Maximum age of files in hours
    """
    if not os.path.exists(upload_folder):
        return
    
    current_time = datetime.now()
    
    for filename in os.listdir(upload_folder):
        file_path = os.path.join(upload_folder, filename)
        
        if os.path.isfile(file_path):
            file_time = datetime.fromtimestamp(os.path.getctime(file_path))
            age_hours = (current_time - file_time).total_seconds() / 3600
            
            if age_hours > max_age_hours:
                try:
                    os.remove(file_path)
                    print(f"🗑️  Cleaned up old file: {filename}")
                except Exception as e:
                    print(f"❌ Failed to delete {filename}: {str(e)}")

def get_model_performance():
    """
    Get performance metrics for all models
    
    Returns:
        dict: Performance data for each model
    """
    # This would typically load from a database or metrics file
    # For now, returning sample data structure
    
    performance_data = {
        'overview': {
            'total_models': 4,
            'total_predictions': 0,
            'average_accuracy': 0.0,
            'last_updated': datetime.now().isoformat()
        },
        'models': {
            'tomato': {
                'accuracy': 0.92,
                'precision': 0.89,
                'recall': 0.91,
                'f1_score': 0.90,
                'total_samples': 150,
                'confusion_matrix': {
                    'good': {'good': 45, 'average': 3, 'bad': 2},
                    'average': {'good': 2, 'average': 42, 'bad': 3},
                    'bad': {'good': 1, 'average': 2, 'bad': 50}
                }
            },
            'apple': {
                'accuracy': 0.88,
                'precision': 0.86,
                'recall': 0.87,
                'f1_score': 0.86,
                'total_samples': 120,
                'confusion_matrix': {
                    'good': {'good': 38, 'average': 4, 'bad': 2},
                    'average': {'good': 3, 'average': 35, 'bad': 4},
                    'bad': {'good': 2, 'average': 3, 'bad': 31}
                }
            },
            'mango': {
                'accuracy': 0.85,
                'precision': 0.83,
                'recall': 0.84,
                'f1_score': 0.83,
                'total_samples': 100,
                'confusion_matrix': {
                    'good': {'good': 28, 'average': 5, 'bad': 2},
                    'average': {'good': 4, 'average': 30, 'bad': 3},
                    'bad': {'good': 2, 'average': 4, 'bad': 22}
                }
            },
            'potato': {
                'accuracy': 0.90,
                'precision': 0.88,
                'recall': 0.89,
                'f1_score': 0.88,
                'total_samples': 130,
                'confusion_matrix': {
                    'good': {'good': 40, 'average': 3, 'bad': 2},
                    'average': {'good': 2, 'average': 38, 'bad': 3},
                    'bad': {'good': 1, 'average': 2, 'bad': 43}
                }
            }
        }
    }
    
    # Calculate overall metrics
    total_accuracy = sum(model['accuracy'] for model in performance_data['models'].values())
    performance_data['overview']['average_accuracy'] = total_accuracy / len(performance_data['models'])
    
    return performance_data

def format_percentage(value):
    """
    Format decimal value as percentage
    
    Args:
        value (float): Decimal value (0.0 to 1.0)
    
    Returns:
        str: Formatted percentage string
    """
    return f"{value * 100:.1f}%"

def get_quality_color(quality):
    """
    Get color for quality label
    
    Args:
        quality (str): Quality label ('good', 'average', 'bad')
    
    Returns:
        str: CSS color value
    """
    colors = {
        'good': '#28a745',      # Green
        'average': '#ffc107',   # Yellow
        'bad': '#dc3545'        # Red
    }
    return colors.get(quality, '#6c757d')  # Default gray

def validate_image_dimensions(image_path, min_width=100, min_height=100):
    """
    Validate image dimensions
    
    Args:
        image_path (str): Path to image file
        min_width (int): Minimum allowed width
        min_height (int): Minimum allowed height
    
    Returns:
        tuple: (is_valid, message)
    """
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            width, height = img.size
            
            if width < min_width or height < min_height:
                return False, f"Image too small. Minimum size: {min_width}x{min_height}"
            
            return True, "Image dimensions are valid"
            
    except Exception as e:
        return False, f"Failed to validate image: {str(e)}"

def create_thumbnail(image_path, output_path, size=(150, 150)):
    """
    Create thumbnail of an image
    
    Args:
        image_path (str): Path to source image
        output_path (str): Path to save thumbnail
        size (tuple): Thumbnail size (width, height)
    """
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Create thumbnail
            img.thumbnail(size, Image.Resampling.LANCZOS)
            img.save(output_path, 'JPEG', quality=85)
            
    except Exception as e:
        print(f"❌ Failed to create thumbnail: {str(e)}")
