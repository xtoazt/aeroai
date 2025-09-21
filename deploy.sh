#!/bin/bash

# Oumi Vercel Deployment Script
echo "🚀 Deploying Oumi to Vercel..."

# Check if Vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo "❌ Vercel CLI not found. Installing..."
    npm install -g vercel
fi

# Check if we're in the right directory
if [ ! -f "vercel.json" ]; then
    echo "❌ vercel.json not found. Please run this script from the project root."
    exit 1
fi

# Deploy to Vercel
echo "📦 Deploying to Vercel..."
vercel --prod

# Check if deployment was successful
if [ $? -eq 0 ]; then
    echo "✅ Deployment successful!"
    echo ""
    echo "🔧 Next steps:"
    echo "1. Set your API keys in Vercel dashboard:"
    echo "   - OPENAI_API_KEY (required)"
    echo "   - ANTHROPIC_API_KEY (optional)"
    echo "   - GEMINI_API_KEY (optional)"
    echo "   - TOGETHER_API_KEY (optional)"
    echo ""
    echo "2. Visit your deployed app and start using the API!"
    echo ""
    echo "📚 For more information, see README_VERCEL.md"
else
    echo "❌ Deployment failed. Check the error messages above."
    exit 1
fi
