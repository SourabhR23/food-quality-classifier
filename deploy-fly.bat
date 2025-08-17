@echo off
echo 🚀 Starting Fly.io deployment...

REM Check if flyctl is installed
where flyctl >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ flyctl is not installed. Please install it first.
    pause
    exit /b 1
)

REM Check if user is logged in
flyctl auth whoami >nul 2>nul
if %errorlevel% neq 0 (
    echo 🔐 Please log in to Fly.io first:
    echo    flyctl auth login
    pause
    exit /b 1
)

echo ✅ Fly.io authentication verified

REM Create the app if it doesn't exist
echo 📱 Creating/updating Fly.io app...
flyctl apps create food-quality-classifier --org personal

REM Create volume for persistent storage
echo 💾 Creating persistent volume for uploads...
flyctl volumes create food_quality_data --size 1 --region iad

REM Deploy the application
echo 🚀 Deploying to Fly.io...
flyctl deploy

REM Check deployment status
echo 🔍 Checking deployment status...
flyctl status

echo ✅ Deployment complete!
echo 🌐 Your app is available at: https://food-quality-classifier.fly.dev
echo 📊 Monitor your app: flyctl dashboard
echo 📝 View logs: flyctl logs
pause 