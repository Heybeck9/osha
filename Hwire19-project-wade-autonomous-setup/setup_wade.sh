#!/bin/bash

echo "🚀 Setting up WADE Autonomous Repo Refactor System"
echo "=================================================="

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install flask fastapi uvicorn pydantic pytest requests structlog

# Make scripts executable
chmod +x /workspace/wade_refactor_system.py
chmod +x /workspace/openhands_refactor_agent.py
chmod +x /workspace/wade_openhands_integration.py

# Create workspace directories
mkdir -p /workspace/wade_logs
mkdir -p /workspace/wade_backups

echo "✅ WADE setup complete!"
echo ""
echo "🎯 Usage:"
echo "  Demo:        python /workspace/wade_openhands_integration.py demo"
echo "  Interactive: python /workspace/wade_openhands_integration.py interactive"
echo "  Direct:      python /workspace/wade_openhands_integration.py 'your request'"
echo ""
echo "📝 Example requests:"
echo "  'Take this repo and convert it to FastAPI'"
echo "  'Refactor /workspace/demo_repo to use microservices'"
echo "  'Transform this Flask app to async FastAPI with logging'"