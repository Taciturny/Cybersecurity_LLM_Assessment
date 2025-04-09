# Cybersecurity LLM

A fine-tuned language model specialized in cybersecurity knowledge, particularly focusing on NIST guidelines and frameworks. This project includes code for data extraction, training the model and deploying it as a REST API.

## Overview

This project fine-tunes a Qwen 1.5B model on cybersecurity documents from NIST to create a specialized assistant that can answer cybersecurity-related questions. The fine-tuned model is then deployed as a FastAPI service for easy integration with other applications.

## Features

- Fine-tuned language model specialized in cybersecurity knowledge
- Instruction-tuned to provide clear, concise responses about cybersecurity topics
- Low-resource model (Qwen 1.5B) that can run on a single consumer GPU
- REST API for easy integration with web applications and security tools
- Detailed responses with appropriate formatting for cybersecurity concepts

## Project Structure

```
├── trian_llm.ipynb       # Training notebook for fine-tuning the model
├── llm.py                # FastAPI server for model deployment
|── data_extraction.ipynb # Data Extraction notebook
├── qwen-fact-checking/   # Directory containing the fine-tuned model
│   └── final_model/      # Fine-tuned model files
├── requirements.txt      # Project dependencies
└── README.md             # This file
```

## Data Extraction

### Features

Downloads NIST cybersecurity publications including (`data_extraction.ipynb`):

* NIST.SP.800-63-3.pdf (Digital Identity Guidelines)
* NIST.CSWP.29.pdf (Cybersecurity Framework)
* NIST.SP.800-53r5.pdf (Security and Privacy Controls)
* NIST.SP.800-61r2.pdf (Computer Security Incident Handling Guide)
* NIST.SP.800-171r1.pdf (CUI Protection Requirements)
* NIST.SP.800-82r2.pdf (ICS Security)


Extracts text content using PyPDF2

Implements retry logic with exponential backoff

Provides detailed logging of the extraction process

Stores documents with metadata in a dataset format

## Model Training

The model is fine-tuned on NIST cybersecurity documents using a LoRA (Low-Rank Adaptation) approach, which allows efficient fine-tuning with limited computational resources. The training process includes:

1. Data preprocessing of NIST cybersecurity documents
2. Creation of instruction-response pairs
3. Tokenization and dataset preparation
4. Fine-tuning with LoRA on a Qwen 1.5B base model
5. Model evaluation and saving

The training notebook (`train_llm.ipynb`) contains the complete training pipeline.

### Training Details

- **Base Model**: Qwen/Qwen2-1.5B
- **Training Method**: LoRA (Low-Rank Adaptation)
- **LoRA Config**:
  - Rank: 16
  - Alpha: 32
  - Target Modules: Query, Key, Value, Output, Gate, Up, and Down projection layers
- **Training Parameters**:
  - Learning Rate: 2e-4
  - Epochs: 3
  - Batch Size: 4
  - Gradient Accumulation Steps: 4
  - Weight Decay: 0.01
  - Scheduler: Cosine

## API Server

The `llm.py` file contains a FastAPI server that loads the fine-tuned model and exposes it through a REST API. The server includes:

- Model loading with appropriate fallbacks for different loading methods
- Query endpoint for asking cybersecurity questions
- Health check endpoint for monitoring

### API Endpoints

- `GET /`: Returns basic API information
- `POST /query`: Accepts a cybersecurity question and returns the model's response
- `GET /health`: Health check endpoint for API monitoring

### Query Request Format

```json
{
  "question": "What are the main components of the NIST Cybersecurity Framework?",
  "max_tokens": 512,
  "temperature": 0.7,
  "top_p": 0.9
}
```

### Response Format

```json
{
  "response": "The main components of the National Institute of Standards and Technology's (NIST) Cybersecurity Framework include...",
  "model_name": "Fine-tuned Qwen Cybersecurity LLM"
}
```

## Installation

### Prerequisites

- Python 3.9+
- PyTorch 2.0+
- CUDA-compatible GPU (recommended)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Taciturny/Cybersecurity_LLM_Assessment.git
   cd Cybersecurity_LLM_Assessment
   ```

2. Install dependencies:
   ```bash
   pip install fastapi uvicorn transformers torch peft pydantic bitsandbytes
   ```

3. Download or place the fine-tuned model in the appropriate directory:
   ```
   qwen-fact-checking/final_model/
   ```

### Running the API Server

```bash
python llm.py
```

The API will be available at `http://localhost:8000`.

## Usage Examples

### Command Line Query using curl

```bash
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "What is the principle of least privilege in cybersecurity?"}'
```

### Python Client

```python
import requests

url = "http://localhost:8000/query"
payload = {
    "question": "How should organizations respond to cybersecurity incidents according to NIST?",
    "max_tokens": 512,
    "temperature": 0.7
}

response = requests.post(url, json=payload)
print(response.json()["response"])
```

## Sample Outputs

**Question**: What are the main components of the NIST Cybersecurity Framework?

**Response**:
```
The main components of the National Institute of Standards and Technology's (NIST) Cybersecurity Framework include:

1. Policies, which set out the overall direction for a security program.
2. Processes, which describe how to implement policies in practice.
3. Controls, which identify specific actions that should be taken at each level of risk management.
4. Information resources, such as information assets, vulnerabilities, and tools used during the assessment process.
5. Evaluation metrics, including cost-benefit analysis, effectiveness measures, and compliance reporting requirements.
```

## Future Work/ Improvements

- [ ] Increase the training dataset with more NIST and industry cybersecurity documents
- [ ] Add support for retrieving information from specific documents
- [ ] Implement RAG (Retrieval-Augmented Generation) for more accurate responses
- [ ] Add more cybersecurity-focused evaluation metrics
- [ ] Create a simple web UI for interacting with the model
