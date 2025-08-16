# 🍎 Food Quality Classifier

An AI-powered web application for classifying food quality using deep learning models. Built with TensorFlow, Flask, and modern web technologies.

## ✨ Features

- **Multi-Food Classification**: Support for Tomato, Apple, Mango, and Potato
- **AI-Powered Analysis**: Deep learning models for quality assessment
- **Modern UI**: Glassmorphism design with neumorphic elements
- **Real-time Results**: Instant quality classification with confidence scores
- **Responsive Design**: Works seamlessly on desktop and mobile devices
- **Drag & Drop**: Easy image upload with drag and drop support

## 🚀 Live Demo

Visit the application: [Food Quality Classifier](https://your-app-url.com)

## 🛠️ Technology Stack

- **Backend**: Python, Flask, TensorFlow 2.13.0
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5
- **AI Models**: EfficientNet-Lite4 architecture
- **Deployment**: Ready for Heroku, Render, or any cloud platform

## 📋 Prerequisites

- Python 3.8+
- TensorFlow 2.13.0
- Flask 2.3.3
- Modern web browser

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/SourabhR23/food-quality-classifier.git
   cd food-quality-classifier
   ```

2. **Create virtual environment**
   ```bash
   python -m venv food_quality
   source food_quality/bin/activate  # On Windows: food_quality\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Open in browser**
   Navigate to `http://localhost:5000`

## 🎯 Usage

1. **Select Food Type**: Choose from Tomato, Apple, Mango, or Potato
2. **Upload Image**: Drag & drop or click to browse for food images
3. **Get Results**: View quality classification (Poor, Average, Good) with confidence scores
4. **Detailed Analysis**: See probability breakdown for each quality level

## 🏗️ Project Structure

```
food-quality-classifier/
├── app.py                 # Main Flask application
├── food_classifier.py     # AI model loading and classification logic
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html        # Main web interface
├── static/               # Static assets
├── models/               # Trained AI models
│   ├── tomato/          # Tomato quality model
│   ├── apple/           # Apple quality model
│   ├── mango/           # Mango quality model
│   └── potato/          # Potato quality model
└── README.md            # This file
```

## 🔧 Configuration

### Environment Variables
- `PORT`: Server port (default: 5000)
- `DEBUG`: Debug mode (default: False)

### Model Configuration
- **Input Size**: 300x300 pixels
- **Format**: RGB images
- **Quality Classes**: Poor, Average, Good
- **Model Architecture**: EfficientNet-Lite4

## 📊 API Endpoints

- `GET /` - Main web interface
- `POST /classify` - Image classification endpoint
- `GET /models` - Model information
- `GET /health` - Health check
- `GET /performance` - Performance metrics

## 🎨 UI Features

- **Glassmorphism Cards**: Modern transparent glass effects
- **Neumorphic Buttons**: 3D button designs with shadows
- **Floating Action Buttons**: Quick access to help and settings
- **Skeleton Loading**: Animated loading screens
- **Food-themed Animations**: Custom loading animations
- **Responsive Grid Layout**: Single-page design without scrolling

## 🔍 Model Performance

- **Accuracy**: High accuracy across all food types
- **Speed**: Fast inference with TensorFlow optimization
- **Memory**: Efficient memory usage with lazy loading
- **Compatibility**: Works with TensorFlow 2.13.0+

## 🚀 Deployment

### Heroku
```bash
heroku create your-app-name
git push heroku master
```

### Render
- Connect your GitHub repository
- Set build command: `pip install -r requirements.txt`
- Set start command: `gunicorn app:app`

### Docker
```bash
docker build -t food-quality-classifier .
docker run -p 5000:5000 food-quality-classifier
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- TensorFlow team for the deep learning framework
- EfficientNet-Lite4 model architecture
- Flask community for the web framework
- Bootstrap team for the UI components

## 📞 Support

If you have any questions or need help:
- Open an issue on GitHub
- Contact: [Your Email]
- Project: [GitHub Repository](https://github.com/SourabhR23/food-quality-classifier)

## 🔄 Changelog

### Version 2.0.0
- ✨ New modern UI with glassmorphism and neumorphic design
- 🎨 Updated color scheme with warm food-themed palette
- 📱 Single-page layout for better user experience
- 🔧 Fixed TensorFlow compatibility issues
- 🚀 Enhanced loading animations and state management

### Version 1.0.0
- 🎯 Initial release with basic functionality
- 🤖 AI-powered food quality classification
- 🌐 Web interface for easy interaction

---

⭐ **Star this repository if you find it helpful!**
