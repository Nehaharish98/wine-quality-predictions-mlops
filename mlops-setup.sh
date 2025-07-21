#!/bin/bash

# MLOps Directory Structure Setup for GitHub Codespaces
# This script creates a comprehensive MLOps project structure for wine quality prediction

set -e  # Exit on error

echo "🍷 Starting MLOps Wine Quality Project Setup for GitHub Codespaces..."
echo "=================================="

# Create backup of existing work
BACKUP_DIR="backup_$(date +%Y%m%d_%H%M%S)"
if [ "$(ls -A .)" ]; then
    echo "📦 Creating backup: $BACKUP_DIR"
    mkdir -p "$BACKUP_DIR"
    cp -r . "$BACKUP_DIR/" 2>/dev/null || true
fi

# Create MLOps directory structure
echo "🏗️  Creating MLOps directory structure..."

# Core directories
mkdir -p {src/{wine_quality/{data,features,models,serving,utils},api,tests},data/{raw,interim,processed,external},models,notebooks,configs,scripts,docs}
mkdir -p {monitoring,deployment/{docker,k8s},reports/figures,logs,.github/workflows}
mkdir -p tests/{unit,integration,e2e}

# Create __init__.py files for Python packages
touch src/__init__.py
touch src/wine_quality/__init__.py
touch src/wine_quality/{data,features,models,serving,utils}/__init__.py
touch tests/{unit,integration,e2e}/__init__.py

echo "✅ Directory structure created successfully!"

# Initialize version control systems
echo "🔧 Initializing version control..."
git init
git config --global user.name "MLOps Engineer" || true
git config --global user.email "mlops@example.com" || true

# Initialize DVC
if command -v dvc >/dev/null 2>&1; then
    dvc init --no-scm
    echo "✅ DVC initialized"
else
    echo "⚠️  DVC not found, will install later"
fi

# Install essential MLOps packages
echo "📦 Installing MLOps