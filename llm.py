import os
import json
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Define the API app
app = FastAPI(
    title="Cybersecurity LLM API",
    description="REST API for querying fine-tuned cybersecurity language model",
    version="1.0.0"
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Define request and response models
class QueryRequest(BaseModel):
    question: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9

class QueryResponse(BaseModel):
    response: str
    model_name: str

# Global variables for model and tokenizer
model = None
tokenizer = None
inference_pipeline = None

# Helper function to extract assistant response
def extract_assistant_response(response):
    """Extract only the assistant's response, stopping at any new user message"""
    parts = response.split("<|assistant|>\n")
    if len(parts) < 2:
        return response

    assistant_text = parts[1]

    # Stop at the next user tag if it exists
    if "<|user|>" in assistant_text:
        assistant_text = assistant_text.split("<|user|>")[0].strip()

    return assistant_text

@app.on_event("startup")
async def load_model():
    """Load the fine-tuned model during app startup"""
    global model, tokenizer, inference_pipeline

    try:
        # Find the model path using relative or absolute path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        adapter_path = os.path.join(script_dir, "qwen-fact-checking", "qwen-fact-checking", "final_model")

        # If the path doesn't exist, try to find the extracted folder
        if not os.path.exists(adapter_path):
            possible_paths = [
                os.path.join(script_dir, "qwen-fact-checking", "final_model"),
                os.path.join(script_dir, "final_model"),
                os.path.join(os.path.dirname(script_dir), "qwen-fact-checking", "final_model"),
                "qwen-fact-checking/final_model"
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    adapter_path = path
                    break

            if not os.path.exists(adapter_path):
                raise FileNotFoundError(f"Model path not found. Tried paths: {possible_paths}")

        print(f"Loading adapter from: {adapter_path}")

        # Check if adapter files exist
        files = os.listdir(adapter_path)
        print(f"Model files exist: {os.path.exists(adapter_path)}")
        print(f"Contents: {files}")

        # Manually read the adapter config to find the base model
        adapter_config_path = os.path.join(adapter_path, "adapter_config.json")

        if os.path.exists(adapter_config_path):
            with open(adapter_config_path, 'r') as f:
                config_data = json.load(f)

            # Extract base model path from config
            base_model_id = config_data.get('base_model_name_or_path')
            if not base_model_id:
                print("Base model not found in adapter config, using default")
                base_model_id = "Qwen/Qwen2-1.5B"
        else:
            base_model_id = "Qwen/Qwen2-1.5B"

        print(f"Using base model: {base_model_id}")

        # Try first to load the model directly with AutoModelForCausalLM
        try:
            print("Attempting to load model directly...")
            model = AutoModelForCausalLM.from_pretrained(
                adapter_path,
                device_map="auto",
                torch_dtype=torch.float16
            )
            print("Direct model loading succeeded")
        except Exception as direct_error:
            print(f"Direct model loading failed: {str(direct_error)}")
            print("Falling back to loading base model with adapter...")

            # Try to import and use PEFT
            try:
                from peft import PeftModel

                # Load the base model
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_id,
                    device_map="auto",
                    torch_dtype=torch.float16
                )

                # Load the adapter onto the base model
                model = PeftModel.from_pretrained(
                    base_model,
                    adapter_path,
                    is_trainable=False  # Set to inference mode
                )

                print("PEFT model loading succeeded")

            except Exception as peft_error:
                print(f"PEFT loading failed: {str(peft_error)}")
                print("Trying with modified PEFT loading approach...")

                try:
                    # Check if this is a Qwen model
                    if "qwen" in base_model_id.lower():
                        # For Qwen models, sometimes direct merging works better
                        from peft import AutoPeftModelForCausalLM

                        model = AutoPeftModelForCausalLM.from_pretrained(
                            adapter_path,
                            device_map="auto",
                            torch_dtype=torch.float16
                        )

                        print("AutoPeftModel loading succeeded")
                    else:
                        raise Exception("Not a Qwen model, cannot use AutoPeftModel fallback")

                except Exception as auto_peft_error:
                    print(f"AutoPeftModel loading failed: {str(auto_peft_error)}")
                    raise Exception(f"All loading methods failed. Please check model compatibility.")

        # Load tokenizer from adapter path
        print(f"Loading tokenizer from adapter path")
        tokenizer = AutoTokenizer.from_pretrained(adapter_path)

        # Create the pipeline
        inference_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="auto",
            torch_dtype=torch.float16
        )

        print("Model loaded successfully")

    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise e

@app.get("/")
async def root():
    """Root endpoint that returns API info"""
    return {"message": "Cybersecurity LLM API", "status": "active"}

@app.post("/query", response_model=QueryResponse)
async def query_model(request: QueryRequest):
    """Endpoint to query the model with a cybersecurity question"""
    if not model or not tokenizer or not inference_pipeline:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Format the prompt
        prompt = f"<|user|>\nAnswer the following cybersecurity question clearly and concisely. Use numbered lists where appropriate and be direct.\n\nQuestion: {request.question}\n<|assistant|>\n"

        # Generate response
        response = inference_pipeline(
            prompt,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            do_sample=True,
            repetition_penalty=1.2
        )[0]['generated_text']

        # Extract just the assistant's response
        assistant_response = extract_assistant_response(response)

        return QueryResponse(
            response=assistant_response,
            model_name="Fine-tuned Qwen Cybersecurity LLM"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model and tokenizer and inference_pipeline:
        return {"status": "healthy", "model_loaded": True}
    return {"status": "unhealthy", "model_loaded": False}

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


