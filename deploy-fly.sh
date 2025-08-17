#!/bin/bash

# Fly.io Deployment Script for Food Quality Classifier
# Make sure you have flyctl installed and are logged in

echo "🚀 Starting Fly.io deployment..."

# Check if flyctl is installed
if ! command -v flyctl &> /dev/null; then
    echo "❌ flyctl is not installed. Please install it first."
    exit 1
fi

# Check if user is logged in
if ! flyctl auth whoami &> /dev/null; then
    echo "🔐 Please log in to Fly.io first:"
    echo "   flyctl auth login"
    exit 1
fi

echo "✅ Fly.io authentication verified"

# Create the app if it doesn't exist
echo "📱 Creating/updating Fly.io app..."
flyctl apps create food-quality-classifier --org personal || true

# Create volume for persistent storage
echo "💾 Creating persistent volume for uploads..."
flyctl volumes create food_quality_data --size 1 --region iad || true

# Deploy the application
echo "🚀 Deploying to Fly.io..."
flyctl deploy

# Check deployment status
echo "🔍 Checking deployment status..."
flyctl status

echo "✅ Deployment complete!"
echo "🌐 Your app is available at: https://food-quality-classifier.fly.dev"
echo "📊 Monitor your app: flyctl dashboard"
echo "📝 View logs: flyctl logs" 