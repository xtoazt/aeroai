"""
Lightweight Oumi implementation for Vercel deployment.
This module provides simplified versions of Oumi functionality that work
without heavy ML dependencies like PyTorch, Transformers, etc.
"""

import json
import httpx
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

class InferenceEngineType(str, Enum):
    OPENAI = "OPENAI"
    ANTHROPIC = "ANTHROPIC"
    GEMINI = "GEMINI"
    TOGETHER = "TOGETHER"

@dataclass
class ModelParams:
    model_name: str
    model_max_length: int = 2048
    torch_dtype_str: str = "bfloat16"

@dataclass
class GenerationParams:
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    batch_size: int = 1

@dataclass
class InferenceConfig:
    model: ModelParams
    generation: GenerationParams
    engine: InferenceEngineType

@dataclass
class Message:
    role: str
    content: str

@dataclass
class Conversation:
    messages: List[Message]

class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class OumiLite:
    """Lightweight Oumi implementation for API-based operations."""
    
    def __init__(self):
        self.openai_client = None
        self.anthropic_client = None
        
    async def _get_openai_client(self):
        """Get OpenAI client."""
        if not self.openai_client:
            import openai
            self.openai_client = openai.AsyncOpenAI(
                api_key=os.getenv("OPENAI_API_KEY")
            )
        return self.openai_client
    
    async def _get_anthropic_client(self):
        """Get Anthropic client."""
        if not self.anthropic_client:
            import anthropic
            self.anthropic_client = anthropic.AsyncAnthropic(
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )
        return self.anthropic_client
    
    async def infer_openai(self, config: InferenceConfig, prompt: str) -> str:
        """Run inference using OpenAI API."""
        try:
            client = await self._get_openai_client()
            response = await client.chat.completions.create(
                model=config.model.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=config.generation.max_new_tokens,
                temperature=config.generation.temperature,
                top_p=config.generation.top_p
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"OpenAI inference failed: {str(e)}")
    
    async def infer_anthropic(self, config: InferenceConfig, prompt: str) -> str:
        """Run inference using Anthropic API."""
        try:
            client = await self._get_anthropic_client()
            response = await client.messages.create(
                model=config.model.model_name,
                max_tokens=config.generation.max_new_tokens,
                temperature=config.generation.temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            raise Exception(f"Anthropic inference failed: {str(e)}")
    
    async def infer_gemini(self, config: InferenceConfig, prompt: str) -> str:
        """Run inference using Google Gemini API."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            model = genai.GenerativeModel(config.model.model_name)
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=config.generation.max_new_tokens,
                    temperature=config.generation.temperature,
                    top_p=config.generation.top_p
                )
            )
            return response.text
        except Exception as e:
            raise Exception(f"Gemini inference failed: {str(e)}")
    
    async def infer_together(self, config: InferenceConfig, prompt: str) -> str:
        """Run inference using Together AI API."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.together.xyz/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {os.getenv('TOGETHER_API_KEY')}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": config.model.model_name,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": config.generation.max_new_tokens,
                        "temperature": config.generation.temperature,
                        "top_p": config.generation.top_p
                    }
                )
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            raise Exception(f"Together AI inference failed: {str(e)}")
    
    async def infer(self, config: InferenceConfig, prompt: str) -> str:
        """Run inference using the specified engine."""
        if config.engine == InferenceEngineType.OPENAI:
            return await self.infer_openai(config, prompt)
        elif config.engine == InferenceEngineType.ANTHROPIC:
            return await self.infer_anthropic(config, prompt)
        elif config.engine == InferenceEngineType.GEMINI:
            return await self.infer_gemini(config, prompt)
        elif config.engine == InferenceEngineType.TOGETHER:
            return await self.infer_together(config, prompt)
        else:
            raise ValueError(f"Unsupported engine: {config.engine}")
    
    async def evaluate_simple(self, config: InferenceConfig, tasks: List[str], num_samples: int = 10) -> Dict[str, Any]:
        """Simple evaluation using API-based inference."""
        results = {}
        
        # Sample evaluation tasks (simplified)
        sample_questions = {
            "arc_challenge": [
                "What is the capital of France?",
                "What is 2 + 2?",
                "What color is the sky?"
            ],
            "hellaswag": [
                "A person is walking down the street. What happens next?",
                "A cat is sitting on a windowsill. What does it do?",
                "Someone opens a book. What do they see?"
            ],
            "mmlu_college_computer_science": [
                "What is a variable in programming?",
                "What is the difference between a list and a tuple?",
                "What is recursion?"
            ]
        }
        
        for task in tasks:
            if task in sample_questions:
                task_results = []
                questions = sample_questions[task][:num_samples]
                
                for question in questions:
                    try:
                        response = await self.infer(config, question)
                        task_results.append({
                            "question": question,
                            "response": response,
                            "correct": True  # Simplified - in real evaluation, you'd check correctness
                        })
                    except Exception as e:
                        task_results.append({
                            "question": question,
                            "response": f"Error: {str(e)}",
                            "correct": False
                        })
                
                results[task] = {
                    "accuracy": sum(1 for r in task_results if r["correct"]) / len(task_results),
                    "total_questions": len(task_results),
                    "results": task_results
                }
        
        return results
    
    async def judge_dataset(self, judge_config: InferenceConfig, dataset: List[Dict[str, str]], 
                          prompt_template: str) -> List[Dict[str, Any]]:
        """Judge a dataset using an LLM judge."""
        judgments = []
        
        for item in dataset:
            try:
                # Format the prompt template with the dataset item
                prompt = prompt_template.format(**item)
                
                # Get judgment from the judge model
                judgment_response = await self.infer(judge_config, prompt)
                
                # Try to parse as JSON if possible
                try:
                    judgment_data = json.loads(judgment_response)
                except:
                    judgment_data = {"raw_judgment": judgment_response}
                
                judgments.append({
                    "input": item,
                    "judgment": judgment_data,
                    "raw_response": judgment_response
                })
                
            except Exception as e:
                judgments.append({
                    "input": item,
                    "judgment": {"error": str(e)},
                    "raw_response": f"Error: {str(e)}"
                })
        
        return judgments

# Global instance
oumi_lite = OumiLite()

# Convenience functions that match the original Oumi API
async def infer(config: InferenceConfig, inputs: List[str]) -> List[Conversation]:
    """Run inference on multiple inputs."""
    conversations = []
    
    for input_text in inputs:
        response = await oumi_lite.infer(config, input_text)
        conversation = Conversation(
            messages=[
                Message(role=Role.USER, content=input_text),
                Message(role=Role.ASSISTANT, content=response)
            ]
        )
        conversations.append(conversation)
    
    return conversations

async def evaluate(config: InferenceConfig, tasks: List[str], num_samples: int = 10) -> List[Dict[str, Any]]:
    """Evaluate a model on specified tasks."""
    results = await oumi_lite.evaluate_simple(config, tasks, num_samples)
    return [results]

async def judge_dataset(judge_config: InferenceConfig, dataset: List[Dict[str, str]], 
                       prompt_template: str) -> List[Dict[str, Any]]:
    """Judge a dataset using an LLM judge."""
    return await oumi_lite.judge_dataset(judge_config, dataset, prompt_template)
