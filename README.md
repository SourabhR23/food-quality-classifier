# 🍎 Food Quality Classifier

An AI-powered web application for automatically assessing food quality using computer vision and deep learning. The system can classify four different food types (tomato, apple, mango, and potato) into three quality categories: good, average, and bad.

## ✨ Features

- **Multi-Food Classification**: Support for tomato, apple, mango, and potato
- **Quality Assessment**: Three quality levels (good, average, bad)
- **Modern Web Interface**: Responsive design with drag & drop upload
- **Real-time Processing**: Instant quality classification results
- **High Accuracy**: Average 88.75% accuracy across all models
- **Production Ready**: Clean, scalable architecture

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Interface │    │   Flask Backend  │    │  ML Models      │
│   (HTML/CSS/JS) │◄──►│   (Python)       │◄──►│  (TensorFlow)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 📁 Project Structure

```
Final/
├── app.py                 # Main Flask application
├── food_classifier.py     # Core ML classification logic
├── utils.py              # Utility functions
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── models/               # Trained ML models
│   ├── tomato/          # Tomato quality model
│   ├── apple/           # Apple quality model
│   ├── mango/           # Mango quality model
│   └── potato/          # Potato quality model
├── templates/            # HTML templates
│   └── index.html       # Main interface
├── static/               # Static assets
│   └── uploads/         # Uploaded images
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 4GB+ RAM (for model loading)

### Installation

1. **Clone or download the project**
   ```bash
   cd Food_Quality_Classifier
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```



4. **Run the application**
   ```bash
   python app.py
   ```

5. **Open your browser**
   Navigate to: http://localhost:5000

## 🎯 Usage Guide

### Basic Workflow

1. **Select Food Type**: Choose from tomato, apple, mango, or potato
2. **Upload Image**: Drag & drop or click to browse for an image
3. **Classify**: Click "Classify Quality" button
4. **View Results**: See quality assessment and confidence scores

### Supported Image Formats

- **Formats**: JPG, PNG, JPEG, GIF
- **Size Limit**: Maximum 16MB
- **Dimensions**: Minimum 100x100 pixels

### Quality Categories

| Quality | Description | Color Code |
|---------|-------------|------------|
| **Good** | High quality, fresh produce | 🟢 Green |
| **Average** | Acceptable quality, minor defects | 🟡 Yellow |
| **Bad** | Poor quality, spoiled/damaged | 🔴 Red |

## 📊 Model Performance

| Model | Accuracy | Status | Best Use Case |
|-------|----------|--------|---------------|
| 🍅 Tomato | 92% | ✅ Excellent | Fresh produce QC |
| 🍎 Apple | 88% | ✅ Good | Storage monitoring |
| 🥔 Potato | 90% | ✅ Good | Processing plants |
| 🥭 Mango | 85% | ⚠️ Acceptable | Basic screening |

**Overall System Accuracy**: 88.75%

## 🔧 Configuration

### Environment Variables

The application uses default configurations, but you can customize:

```bash
# Flask configuration
export FLASK_ENV=development
export FLASK_DEBUG=1

# Upload settings
export MAX_CONTENT_LENGTH=16777216  # 16MB in bytes
```

### Model Configuration

Models are automatically loaded from the `models/` directory. Each model should contain:
- `saved_model.pb` - TensorFlow SavedModel
- `variables/` - Model variables directory
- `keras_metadata.pb` - Model metadata

## 🧪 Testing

### Manual Testing

1. **Health Check**: Visit `/health` endpoint
2. **Model Info**: Visit `/models` endpoint
3. **Performance**: Visit `/performance` endpoint

### Sample Images

Test the system with various food images:
- Different lighting conditions
- Various angles and orientations
- Mixed quality levels
- Different food varieties

## 🚀 Deployment

### Development

```bash
python app.py
```

### Production

```bash
# Install production dependencies
pip install gunicorn

# Run with Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Docker (Optional)

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

## 🔍 API Endpoints

| Endpoint | Method | Description | Response |
|----------|--------|-------------|----------|
| `/` | GET | Main interface | HTML page |
| `/classify` | POST | Image classification | JSON results |
| `/models` | GET | Model information | JSON metadata |
| `/performance` | GET | Performance metrics | JSON data |
| `/health` | GET | Health check | JSON status |

### Example API Usage

```bash
# Classify an image
curl -X POST -F "image=@food.jpg" -F "food_type=tomato" http://localhost:5000/classify

# Get model information
curl http://localhost:5000/models

# Check system health
curl http://localhost:5000/health
```

## 🛠️ Troubleshooting

### Common Issues

#### 1. **Models Not Loading**
```
Error: Model directory not found
```
**Solution**: Ensure models are in the `models/` directory with proper structure

#### 2. **Memory Issues**
```
Error: Out of memory
```
**Solution**: Close other applications, ensure 4GB+ RAM available

#### 3. **Image Upload Failures**
```
Error: Invalid file type
```
**Solution**: Use supported formats (JPG, PNG, JPEG, GIF)

#### 4. **Classification Errors**
```
Error: Model not found for food type
```
**Solution**: Select a supported food type (tomato, apple, mango, potato)

### Performance Optimization

- **Model Caching**: Models are loaded once and cached
- **Image Optimization**: Efficient preprocessing pipeline
- **Memory Management**: Automatic cleanup of uploaded files

## 📈 Monitoring & Maintenance

### Performance Metrics

- **Response Time**: Monitor classification speed
- **Accuracy**: Track model performance over time
- **Error Rates**: Monitor classification failures
- **Resource Usage**: Track memory and CPU usage

### Regular Maintenance

- **Model Updates**: Retrain models with new data
- **Performance Review**: Monthly accuracy assessment
- **System Updates**: Keep dependencies current
- **Backup**: Regular model and configuration backups

## 🤝 Contributing

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Code Style

- **Python**: Follow PEP 8 guidelines
- **HTML/CSS**: Use consistent indentation
- **JavaScript**: Follow ES6+ standards
- **Documentation**: Add docstrings and comments

## 📚 Documentation

- **Project Report**: `reports/project_report.md`
- **Model Performance**: `reports/model_performance_report.md`
- **API Reference**: See API Endpoints section above

## 📄 License

This project is provided as-is for educational and research purposes.

## 🙏 Acknowledgments

- **TensorFlow**: Machine learning framework
- **Flask**: Web application framework
- **Bootstrap**: Frontend UI components
- **Pillow**: Image processing library

## 📞 Support

For questions or issues:

1. Check the troubleshooting section above
2. Review the project reports
3. Check the API documentation
4. Create an issue in the repository

---

**Version**: 1.0.0  
**Last Updated**: {{ datetime.now().strftime('%Y-%m-%d') }}  
**Status**: Production Ready ✅
