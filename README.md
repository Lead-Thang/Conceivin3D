# Conceivo 3D Integration

This project provides integration between the Conceivo AI assistant and 3D modeling tools, enabling natural language creation and manipulation of 3D models.

## Overview

The Conceivo 3D Integration allows users to:
- Generate 3D modeling code from natural language descriptions
- Execute generated code to create 3D models
- Analyze and describe 3D models using natural language
- Extend capabilities to various 3D modeling libraries
- Convert natural language instructions to visual 3D representations

## How to Run the App Locally

### 1. Prerequisites

Before running the app, make sure you have the following installed:
- [Node.js](https://nodejs.org) (v18 or higher)
- [Python](https://www.python.org/downloads/) (v3.8 or higher)
- [PyTorch](https://pytorch.org/get-started/locally/) (with development headers)
- [Hunyuan3D](https://github.com/Tencent/Hunyuan3D) repository cloned in the project root directory

### 2. Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/conceivin3d.git
```

2. Install dependencies:
```bash
cd conceivin3d
npm install
```

3. Install Python dependencies:
```bash
pip install torch transformers pythreejs
```

4. Set up Hunyuan3D:
```bash
# Clone the Hunyuan3D repository next to your project folder:
git clone https://github.com/Tencent/Hunyuan3D.git
```

### 3. Running the App

1. Start the development server:
```bash
npm run dev
```

2. Open [http://localhost:3000](http://localhost:3000) in your browser

### 4. Using the 3D Generation API

You can test the 3D generation API using curl:
```bash
curl -X POST http://localhost:3000/api/generate-3d \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Create a modern chair"}'
```

## Components

1. **conceivo_3d_integration.py**: Core integration module connecting Conceivo with 3D modeling capabilities
2. **conceivo-3d-example.py**: Example demonstrating how to use the integration to create a 3D scene from natural language

## Features

### Natural Language to 3D Code
The integration can interpret natural language descriptions of 3D objects and scenes, translating them into executable code for 3D modeling libraries.

### Interactive Model Analysis
Users can interactively analyze existing 3D models through natural language queries about their properties, structure, and characteristics.

### Multi-library Support
While built with pythreejs for basic functionality, the architecture supports extension to other 3D libraries including Blender, Three.js, and Unity.