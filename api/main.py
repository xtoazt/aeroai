"""
FastAPI web application for Oumi platform.
This provides REST API endpoints for key Oumi functionality suitable for Vercel deployment.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import tempfile
import os
from pathlib import Path

# Import lightweight Oumi components
from oumi_lite import (
    infer, evaluate, judge_dataset,
    InferenceConfig, ModelParams, GenerationParams, InferenceEngineType,
    Conversation, Message, Role
)
import os

app = FastAPI(
    title="Oumi API",
    description="REST API for Oumi ML platform - inference, evaluation, and model operations",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API requests/responses
class InferenceRequest(BaseModel):
    model_name: str
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    engine: str = "OPENAI"  # Default to OpenAI for Vercel compatibility

class InferenceResponse(BaseModel):
    response: str
    model_used: str
    tokens_generated: int

class EvaluationRequest(BaseModel):
    model_name: str
    tasks: List[str]
    num_samples: int = 10

class EvaluationResponse(BaseModel):
    results: Dict[str, Any]
    tasks_evaluated: List[str]

class JudgeRequest(BaseModel):
    judge_model: str
    dataset: List[Dict[str, str]]
    prompt_template: str

class JudgeResponse(BaseModel):
    judgments: List[Dict[str, Any]]
    total_judged: int

class HealthResponse(BaseModel):
    status: str
    version: str
    available_engines: List[str]

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Welcome to Oumi API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        available_engines=["OPENAI", "ANTHROPIC", "GEMINI", "TOGETHER"]
    )

@app.post("/inference", response_model=InferenceResponse)
async def run_inference(request: InferenceRequest):
    """
    Run inference on a model with the given prompt.
    
    This endpoint supports various inference engines including:
    - OpenAI API
    - Anthropic API  
    - Google Gemini API
    - Together AI API
    """
    try:
        # Create inference configuration
        config = InferenceConfig(
            model=ModelParams(
                model_name=request.model_name,
                model_max_length=2048,
                torch_dtype_str="bfloat16"
            ),
            generation=GenerationParams(
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p
            ),
            engine=InferenceEngineType(request.engine)
        )
        
        # Run inference
        results = await infer(config, inputs=[request.prompt])
        
        if results and len(results) > 0:
            response_text = results[0].messages[-1].content
            return InferenceResponse(
                response=response_text,
                model_used=request.model_name,
                tokens_generated=len(response_text.split())  # Rough estimate
            )
        else:
            raise HTTPException(status_code=500, detail="No response generated")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@app.post("/evaluate", response_model=EvaluationResponse)
async def run_evaluation(request: EvaluationRequest):
    """
    Evaluate a model on specified tasks.
    
    Note: This is a simplified evaluation suitable for Vercel's constraints.
    For full evaluation capabilities, use the local Oumi CLI.
    """
    try:
        # Create evaluation configuration
        config = InferenceConfig(
            model=ModelParams(
                model_name=request.model_name,
                model_max_length=2048,
                torch_dtype_str="bfloat16"
            ),
            generation=GenerationParams(
                batch_size=1,
                max_new_tokens=512,
                temperature=0.7
            ),
            engine=InferenceEngineType.OPENAI  # Default to OpenAI for evaluation
        )
        
        # Run evaluation
        results = await evaluate(config, request.tasks, request.num_samples)
        
        return EvaluationResponse(
            results=results[0] if results else {},
            tasks_evaluated=request.tasks
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

@app.post("/judge", response_model=JudgeResponse)
async def run_judgment(request: JudgeRequest):
    """
    Judge a dataset using a model.
    
    This endpoint allows you to evaluate the quality of responses
    using an LLM judge.
    """
    try:
        # Create judge configuration
        config = InferenceConfig(
            model=ModelParams(
                model_name=request.judge_model,
                model_max_length=2048
            ),
            generation=GenerationParams(
                max_new_tokens=512,
                temperature=0.1
            ),
            engine=InferenceEngineType.OPENAI
        )
        
        # Run judgment
        judgments = await judge_dataset(config, request.dataset, request.prompt_template)
        
        return JudgeResponse(
            judgments=judgments,
            total_judged=len(judgments)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Judgment failed: {str(e)}")

@app.get("/models", response_model=Dict[str, List[str]])
async def list_available_models():
    """List available models by category."""
    return {
        "openai": [
            "gpt-4o",
            "gpt-4o-mini", 
            "gpt-3.5-turbo"
        ],
        "anthropic": [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022"
        ],
        "google": [
            "gemini-1.5-pro",
            "gemini-1.5-flash"
        ],
        "together": [
            "meta-llama/Llama-3.1-8B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct"
        ]
    }

@app.get("/tasks", response_model=Dict[str, List[str]])
async def list_available_tasks():
    """List available evaluation tasks."""
    return {
        "reasoning": [
            "arc_challenge",
            "arc_easy",
            "hellaswag"
        ],
        "knowledge": [
            "mmlu_college_computer_science",
            "mmlu_college_mathematics",
            "truthfulqa_mc2"
        ],
        "math": [
            "gsm8k",
            "math"
        ],
        "coding": [
            "humaneval",
            "mbpp"
        ]
    }

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a file for processing."""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        return {
            "filename": file.filename,
            "size": len(content),
            "temp_path": tmp_file_path,
            "message": "File uploaded successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found", "available_endpoints": [
            "/", "/health", "/inference", "/evaluate", "/judge", "/models", "/tasks", "/upload"
        ]}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "message": str(exc)}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
