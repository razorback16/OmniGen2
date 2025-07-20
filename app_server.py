import dotenv

dotenv.load_dotenv(override=True)

import argparse
import base64
import io
import os
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional

from PIL import Image
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import numpy as np
import torch
import vtracer

from accelerate import Accelerator

from model_manager import ModelManager
from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="OmniGen2 OpenAI-compatible API Server")
    parser.add_argument(
        "--model_path",
        type=str,
        default="OmniGen2/OmniGen2",
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--transformer_path",
        type=str,
        default=None,
        help="Path to transformer checkpoint.",
    )
    parser.add_argument(
        "--transformer_lora_path",
        type=str,
        default=None,
        help="Path to transformer LoRA checkpoint.",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="euler",
        choices=["euler", "dpmsolver"],
        help="Scheduler to use.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default='bf16',
        choices=['fp32', 'fp16', 'bf16'],
        help="Data type for model weights."
    )
    parser.add_argument(
        "--enable_model_cpu_offload",
        action="store_true",
        help="Enable model CPU offload."
    )
    parser.add_argument(
        "--enable_sequential_cpu_offload",
        action="store_true",
        help="Enable sequential CPU offload."
    )
    parser.add_argument(
        "--enable_group_offload",
        action="store_true",
        help="Enable group offload."
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the server to."
    )
    return parser.parse_args()


# Global variable to store parsed arguments
ARGS = parse_args()

# ========== GENERATION CONFIGURATION ==========
# Configuration for different generation modes and quality levels

GENERATE_MODE_CONFIG = {
    "text_guidance_scale": 4.0,
    "image_guidance_scale": 2.0,
    "cfg_range_start": 0.0,
    "cfg_range_end": 1.0,
    "num_inference_steps": {
        "low": 10,
        "medium": 20,
        "high": 50
    }
}

EDIT_MODE_CONFIG = {
    "text_guidance_scale": 5.0,
    "image_guidance_scale": 1.2,  # Keep same as generate mode
    "cfg_range_start": 0.0,
    "cfg_range_end": 0.6,
    "num_inference_steps": {
        "low": 10,
        "medium": 20,
        "high": 50
    }
}

def get_mode_config(mode: str = "generate") -> dict:
    """Get configuration for the specified mode (generate or edit)."""
    if mode.lower() == "edit":
        return EDIT_MODE_CONFIG
    return GENERATE_MODE_CONFIG

def get_quality_steps(quality: str, mode: str = "generate") -> int:
    """Map quality level to number of inference steps based on mode."""
    config = get_mode_config(mode)
    quality = quality.lower()
    if quality == "auto":
        quality = "medium"  # auto defaults to medium
    return config["num_inference_steps"].get(quality, config["num_inference_steps"]["medium"])

def get_guidance_scales(mode: str = "generate") -> tuple[float, float]:
    """Get text and image guidance scales for the specified mode."""
    config = get_mode_config(mode)
    return config["text_guidance_scale"], config["image_guidance_scale"]

def get_cfg_range(mode: str = "generate") -> tuple[float, float]:
    """Get CFG range for the specified mode."""
    config = get_mode_config(mode)
    return config["cfg_range_start"], config["cfg_range_end"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    args = argparse.Namespace(
        model_path=ARGS.model_path,
        transformer_path=ARGS.transformer_path,
        transformer_lora_path=ARGS.transformer_lora_path,
        scheduler=ARGS.scheduler,
        num_inference_step=20,
        seed=0,
        height=1024,
        width=1024,
        max_input_image_pixels=1048576,
        dtype=ARGS.dtype,
        text_guidance_scale=4.0,
        image_guidance_scale=2.0,
        cfg_range_start=0.0,
        cfg_range_end=1.0,
        negative_prompt="(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar",
        enable_model_cpu_offload=ARGS.enable_model_cpu_offload,
        enable_sequential_cpu_offload=ARGS.enable_sequential_cpu_offload,
        enable_group_offload=ARGS.enable_group_offload,
    )
    
    # Create model manager (no models loaded yet)
    print("Initializing model manager...")
    app.state.model_manager = ModelManager(ARGS, idle_timeout_seconds=3600)  # 1 hour timeout
    app.state.args = args
    print("Server ready. Models will be loaded on first request.")
    
    yield
    
    # Shutdown
    print("Shutting down model manager...")
    if hasattr(app.state, 'model_manager'):
        app.state.model_manager.shutdown()
    app.state.model_manager = None
    app.state.args = None


app = FastAPI(lifespan=lifespan)

# Create static directory for serving SVG files
STATIC_DIR = "static/svgs"
os.makedirs(STATIC_DIR, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


class GenerationRequest(BaseModel):
    prompt: str
    n: int = 1
    size: str = "auto"
    quality: str = "auto"
    response_format: str = "b64_json"




def get_size_dimensions(size: str) -> tuple[int, int]:
    """Parse size parameter and return width, height."""
    if size.lower() == "auto":
        return 1024, 1024  # auto defaults to 1024x1024
    
    try:
        width, height = map(int, size.split('x'))
        return width, height
    except ValueError:
        return 1024, 1024  # fallback to default


def run_pipeline(model_manager: ModelManager,
                args: argparse.Namespace, 
                instruction: str, 
                negative_prompt: str, 
                input_images: Optional[List[Image.Image]],
                quality: str = "auto",
                mode: str = "generate") -> List[Image.Image]:
    """Run the image generation pipeline with mode-specific configuration."""
    # Get pipeline and accelerator (loads if necessary)
    pipeline, accelerator = model_manager.get_pipeline()
    
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
    
    # Get mode-specific configuration
    num_inference_steps = get_quality_steps(quality, mode)
    text_guidance_scale, image_guidance_scale = get_guidance_scales(mode)
    cfg_range_start, cfg_range_end = get_cfg_range(mode)

    results = pipeline(
        prompt=instruction,
        input_images=input_images,
        width=args.width,
        height=args.height,
        num_inference_steps=num_inference_steps,
        max_sequence_length=1024,
        text_guidance_scale=text_guidance_scale,
        image_guidance_scale=image_guidance_scale,
        cfg_range=(cfg_range_start, cfg_range_end),
        negative_prompt=negative_prompt,
        num_images_per_prompt=args.num_images_per_prompt,
        generator=generator,
        output_type="pil",
    )
    return results.images


def encode_images_to_base64(images: List[Image.Image]) -> List[dict]:
    """Encode PIL images to base64 format."""
    images_b64 = []
    for image in images:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        images_b64.append({"b64_json": img_str})
    return images_b64


@app.post("/v1/images/generations")
async def generate_image(request: GenerationRequest):
    """Generate images from text prompt (OpenAI-compatible endpoint)."""
    try:
        args = app.state.args
        
        # Handle size parameter (including auto mode)
        args.width, args.height = get_size_dimensions(request.size)
        args.num_images_per_prompt = request.n
        
        results = run_pipeline(app.state.model_manager, args, request.prompt, args.negative_prompt, None, request.quality, "generate")
        return {"data": encode_images_to_base64(results)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")


@app.post("/v1/images/edits")
async def edit_image(
    request: Request,
    prompt: str = Form(...),
    mask: UploadFile = File(None),
    n: int = Form(1),
    size: str = Form("auto"),
    quality: str = Form("auto"),
    response_format: str = Form("b64_json"),
):
    """Edit images based on prompt and input images (OpenAI-compatible endpoint)."""
    try:
        args = app.state.args
        
        # Handle size parameter (including auto mode)
        args.width, args.height = get_size_dimensions(size)
        args.num_images_per_prompt = n

        # Get form data to handle both 'image' and 'image[]' field names
        form = await request.form()
        image_files = []
        
        # Check for 'image' field(s)
        if 'image' in form:
            image_value = form.getlist('image')
            for img in image_value:
                if hasattr(img, 'read'):  # It's an UploadFile
                    image_files.append(img)
        
        # Check for 'image[]' field(s) if no 'image' found
        if not image_files and 'image[]' in form:
            image_value = form.getlist('image[]')
            for img in image_value:
                if hasattr(img, 'read'):  # It's an UploadFile
                    image_files.append(img)
        
        if not image_files:
            raise HTTPException(status_code=400, detail="No image files provided. Use 'image' or 'image[]' field name.")

        # Process input images
        input_images = []
        for img_file in image_files:
            img_content = await img_file.read()
            input_images.append(Image.open(io.BytesIO(img_content)).convert("RGB"))

        # Process mask if provided
        if mask:
            if len(input_images) > 1:
                raise HTTPException(status_code=400, detail="Mask is only supported for single image inpainting.")
            
            mask_content = await mask.read()
            mask_image = Image.open(io.BytesIO(mask_content)).convert("L")
            
            # Ensure mask is the same size as the image
            if mask_image.size != input_images[0].size:
                mask_image = mask_image.resize(input_images[0].size)
            
            # Apply the mask to the image by setting the unmasked area to black
            input_image_np = np.array(input_images[0])
            mask_np = np.array(mask_image)
            input_image_np[mask_np == 0] = 0  # Black out the unmasked area
            input_images[0] = Image.fromarray(input_image_np)

        results = run_pipeline(app.state.model_manager, args, prompt, args.negative_prompt, input_images, quality, "edit")
        return {"data": encode_images_to_base64(results)}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image editing failed: {str(e)}")


@app.post("/v1/images/vectorize")
async def vectorize_image(
    request: Request,
    file: UploadFile = File(...),
    response_format: str = Form("url"),
):
    """Vectorize raster image to SVG format."""
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and validate image
        img_content = await file.read()
        
        # Check file size (5MB limit)
        if len(img_content) > 5 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Image must be less than 5MB")
        
        try:
            image = Image.open(io.BytesIO(img_content))
            
            # Check resolution and dimensions
            width, height = image.size
            if width * height > 16 * 1024 * 1024:  # 16 MP
                raise HTTPException(status_code=400, detail="Image resolution must be less than 16 MP")
            
            if max(width, height) > 4096:
                raise HTTPException(status_code=400, detail="Image max dimension must be less than 4096 pixels")
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
        
        # Save image to temporary bytes buffer for vtracer
        temp_buffer = io.BytesIO()
        image.save(temp_buffer, format='PNG')
        temp_buffer.seek(0)
        
        # Use vtracer to convert to SVG
        try:
            svg_str = vtracer.convert_raw_image_to_svg(
                temp_buffer.getvalue(),
                img_format='png',
                colormode='color',  # Use color mode for better results
                hierarchical='stacked',  # Use stacked hierarchical clustering
                mode='spline',  # Use spline mode for smooth curves
                filter_speckle=4,  # Filter small speckles
                color_precision=6,  # Good balance of quality and file size
                layer_difference=16,  # Reasonable layer difference
                corner_threshold=60,  # Corner detection threshold
                length_threshold=4.0,  # Length threshold for segments
                splice_threshold=45,  # Splice threshold for splines
                path_precision=8  # Path precision
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Vectorization failed: {str(e)}")
        
        # Return response based on format
        if response_format == "b64_json":
            svg_b64 = base64.b64encode(svg_str.encode('utf-8')).decode('utf-8')
            return {"image": {"b64_json": svg_b64}}
        else:  # URL format - save SVG file and return URL
            # Generate unique filename
            svg_filename = f"{uuid.uuid4().hex}.svg"
            svg_filepath = os.path.join(STATIC_DIR, svg_filename)
            
            # Save SVG file
            with open(svg_filepath, 'w', encoding='utf-8') as f:
                f.write(svg_str)
            
            # Construct URL using environment variable or fallback to request URL
            base_url = os.getenv('BASE_URL')
            if not base_url:
                # Fallback to constructing from request
                host = request.headers.get('host', f'{request.url.hostname}:{request.url.port}')
                base_url = f"{request.url.scheme}://{host}"
            
            # Remove trailing slash if present
            base_url = base_url.rstrip('/')
            svg_url = f"{base_url}/static/svgs/{svg_filename}"
            
            return {"image": {"url": svg_url}}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vectorization failed: {str(e)}")


@app.post("/v1/models/load")
async def load_models():
    """Manually load models to GPU."""
    try:
        loaded = app.state.model_manager.load_models()
        if loaded:
            return {"message": "Models loaded successfully", "status": "loaded"}
        else:
            return {"message": "Models already loaded", "status": "already_loaded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load models: {str(e)}")


@app.post("/v1/models/unload")
async def unload_models():
    """Manually unload models from GPU."""
    try:
        unloaded = app.state.model_manager.unload_models()
        if unloaded:
            return {"message": "Models unloaded successfully", "status": "unloaded"}
        else:
            return {"message": "Models already unloaded", "status": "already_unloaded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to unload models: {str(e)}")


@app.get("/v1/models/status")
async def get_model_status():
    """Get current status of model loading."""
    try:
        status = app.state.model_manager.get_status()
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model status: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "OmniGen2 OpenAI-compatible API Server",
        "endpoints": {
            "generate": "/v1/images/generations",
            "edit": "/v1/images/edits",
            "vectorize": "/v1/images/vectorize",
            "models": {
                "load": "/v1/models/load",
                "unload": "/v1/models/unload",
                "status": "/v1/models/status"
            },
            "health": "/health"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=ARGS.host, port=ARGS.port)
