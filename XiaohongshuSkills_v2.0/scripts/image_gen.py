#!/usr/bin/env python3
"""
AI Image Generation using SiliconFlow API
Supports FLUX.1-schnell model for fast image generation
"""

import json
import os
import sys
import argparse
import requests
import time
from pathlib import Path

CONFIG_DIR = Path(__file__).parent.parent / "config"
DEFAULT_API_CONFIG = CONFIG_DIR / "api_config.json"

# Default model settings
DEFAULT_MODEL = "black-forest-labs/FLUX.1-schnell"
API_BASE_URL = "https://api.siliconflow.cn/v1"


def load_api_config():
    """Load API configuration from config file"""
    if DEFAULT_API_CONFIG.exists():
        with open(DEFAULT_API_CONFIG, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def generate_image(prompt: str, api_key: str = None, model: str = None, output_path: str = None) -> str:
    """
    Generate image using SiliconFlow API
    
    Args:
        prompt: Image description
        api_key: SiliconFlow API key
        model: Model name (default: FLUX.1-schnell)
        output_path: Where to save the image
    
    Returns:
        Path to generated image
    """
    config = load_api_config()
    
    if not api_key:
        api_key = config.get("siliconflow_api_key")
    if not model:
        model = config.get("default_model", DEFAULT_MODEL)
    
    if not api_key:
        print("Error: No API key provided. Please set siliconflow_api_key in config/api_config.json")
        sys.exit(1)
    
    # Prepare request
    url = f"{API_BASE_URL}/images/generations"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "prompt": prompt,
        "image_size": "1024x1024",
        "num_inference_steps": 20,
        "guidance_scale": 7.5
    }
    
    print(f"[image_gen] Generating image with prompt: {prompt}")
    print(f"[image_gen] Using model: {model}")
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        
        result = response.json()
        
        if "data" in result and len(result["data"]) > 0:
            image_url = result["data"][0].get("url")
            
            if not image_url:
                print("Error: No image URL in response")
                sys.exit(1)
            
            # Download image
            print(f"[image_gen] Downloading image from: {image_url}")
            image_response = requests.get(image_url, timeout=60)
            image_response.raise_for_status()
            
            # Save image
            if not output_path:
                timestamp = int(time.time())
                output_path = f"tmp/generated_{timestamp}.jpg"
            
            # Ensure tmp directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'wb') as f:
                f.write(image_response.content)
            
            print(f"[image_gen] Image saved to: {output_path}")
            return output_path
        else:
            print(f"Error: Unexpected response format: {result}")
            sys.exit(1)
            
    except requests.exceptions.RequestException as e:
        print(f"Error: API request failed: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="AI Image Generation using SiliconFlow")
    parser.add_argument("--prompt", "-p", required=True, help="Image description/prompt")
    parser.add_argument("--api-key", help="SiliconFlow API key (or use config file)")
    parser.add_argument("--model", "-m", help="Model name")
    parser.add_argument("--output", "-o", help="Output image path")
    
    args = parser.parse_args()
    
    generate_image(
        prompt=args.prompt,
        api_key=args.api_key,
        model=args.model,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
