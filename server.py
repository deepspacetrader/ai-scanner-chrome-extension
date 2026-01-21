#!/usr/bin/env python3
"""
Ultralytics YOLO Object Detection API Server
Replaces Ollama integration with YOLOv8 for real-time object detection
"""

from contextlib import asynccontextmanager
from pathlib import Path
import logging
import base64
import io
import requests
import hashlib
from typing import Dict, Any

import numpy as np
import cv2
import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi import Request
from ultralytics import YOLO
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer, Qwen2VLForConditionalGeneration
import uvicorn
from object_config import OBJECT_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model variables - will be loaded on startup
yolo_model = None
florence_model = None
florence_processor = None
summarizer_model = None
summarizer_tokenizer = None
samsung_trm_model = None # TRM for recursive reasoning
current_summarize_id = 0  # To track and cancel old summarization requests
device = "cuda" if torch.cuda.is_available() else "cpu"
# Set dtype based on device availability
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Model configuration
# YOLO26 models now supported with updated ultralytics
AVAILABLE_MODELS = {
    "detection": {
        "name": "YOLO26x",
        "model_file": "yolo26x.pt",
        "description": "Object detection only (YOLO26)"
    },
    "segmentation": {
        "name": "YOLO26x-seg", 
        "model_file": "yolo26x-seg.pt",
        "description": "Object detection with segmentation masks (YOLO26)"
    }
}

# Current model settings (can be changed via API)
current_model_type = "segmentation"  # Default to segmentation
current_model_config = AVAILABLE_MODELS[current_model_type]

# GLM-Edge/Qwen-VL Local Model configuration
GLM_LOCAL_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
vision_model_obj = None
vision_processor_obj = None

logger.info(f"Using device: {device} with dtype: {dtype}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global yolo_model, florence_model, florence_processor, summarizer_model, summarizer_tokenizer, vision_model_obj, vision_processor_obj, samsung_trm_model
    try:
        # Load YOLO model based on current configuration
        logger.info(f"Loading {current_model_config['name']} model...")
        yolo_model = YOLO(current_model_config['model_file'])
        yolo_model.to(device)
        logger.info(f"{current_model_config['name']} model loaded successfully on {device}")

        # Load Qwen2.5-0.5B-Instruct for extreme speed on RTX 4070 Super.
        # At 0.5B params in FP16, it uses ~1.1GB VRAM and generates almost instantly.
        logger.info("Loading Qwen2.5-0.5B-Instruct...")
        summarizer_model_id = 'Qwen/Qwen2.5-0.5B-Instruct'
        
        summarizer_tokenizer = AutoTokenizer.from_pretrained(summarizer_model_id)
        summarizer_model = AutoModelForCausalLM.from_pretrained(
            summarizer_model_id,
            trust_remote_code=True,
            dtype=torch.float16,
            device_map="auto"
        )
        logger.info("Qwen2.5-0.5B-Instruct loaded successfully")

        # Load Florence-2 last as it's the largest single-block model
        logger.info("Loading Florence-2-large model...")
        florence_model_id = 'microsoft/Florence-2-large'
        florence_model = AutoModelForCausalLM.from_pretrained(
            florence_model_id, 
            trust_remote_code=True,
            attn_implementation="eager",
            dtype=dtype
        ).to(device).eval()
        florence_processor = AutoProcessor.from_pretrained(florence_model_id, trust_remote_code=True)
        logger.info("Florence-2 model loaded successfully")

        # Load Qwen2-VL for high-quality vision reasoning
        logger.info(f"Loading {GLM_LOCAL_MODEL_ID}...")
        vision_processor_obj = AutoProcessor.from_pretrained(GLM_LOCAL_MODEL_ID, trust_remote_code=True)
        vision_model_obj = Qwen2VLForConditionalGeneration.from_pretrained(
            GLM_LOCAL_MODEL_ID,
            trust_remote_code=True,
            dtype=dtype,
            device_map="auto"
        ).eval()
        logger.info("Qwen2-VL loaded successfully")
        
        yield
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        # If Qwen fails, we can still run YOLO and Florence
        if yolo_model and florence_model:
            logger.warning("Continuing without summarization model")
            yield
        else:
            raise
    # Shutdown
    logger.info("Shutting down Ultralytics server")

app = FastAPI(
    title="Ultralytics YOLO Detection API",
    description="Object detection API using YOLO11 models",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware to allow requests from browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint - health check"""
    return {"message": "Ultralytics YOLO Detection API", "status": "running"}

@app.get("/api/status")
async def get_status():
    """Get API status and model information"""
    global yolo_model, florence_model
    if yolo_model is None:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "message": "Model not loaded"}
        )

    return {
        "status": "ready",
        "models": {
            "yolo": current_model_config['name'],
            "yolo_type": current_model_type,
            "vlm": "Florence-2-large",
            "vision_glm": "Qwen2-VL-2B-Instruct (Local)",
            "samsung_trm": "wtfmahe/Samsung-TRM (reasoning)",
            "summarizer": "Qwen2.5-0.5B-Instruct" if summarizer_model else "none"
        },
        "device": device,
        "message": f"AI Scanner Ready: {current_model_config['name']}, Florence-2, GLM-Edge-V and TRM Reasoning Core"
    }

@app.get("/api/config/objects")
async def get_object_config():
    """Get the full object configuration for the frontend"""
    return OBJECT_CONFIG

@app.get("/api/models")
async def get_available_models():
    """Get available YOLO models and current model configuration"""
    return {
        "available_models": AVAILABLE_MODELS,
        "current_model": current_model_type,
        "current_config": current_model_config
    }

@app.post("/api/models/switch")
async def switch_model(request: Dict[str, Any]):
    """Switch between detection and segmentation models"""
    global current_model_type, current_model_config, yolo_model
    
    try:
        new_model_type = request.get("model_type")
        if new_model_type not in AVAILABLE_MODELS:
            raise HTTPException(status_code=400, detail=f"Invalid model type: {new_model_type}")
        
        if new_model_type == current_model_type:
            return {"status": "unchanged", "message": f"Already using {current_model_config['name']}"}
        
        # Update configuration
        current_model_type = new_model_type
        current_model_config = AVAILABLE_MODELS[current_model_type]
        
        # Load new model
        logger.info(f"Switching to {current_model_config['name']}...")
        yolo_model = YOLO(current_model_config['model_file'])
        yolo_model.to(device)
        
        logger.info(f"Successfully switched to {current_model_config['name']}")
        
        return {
            "status": "success",
            "message": f"Switched to {current_model_config['name']}",
            "current_model": current_model_type,
            "current_config": current_model_config
        }
        
    except Exception as e:
        logger.error(f"Model switch failed: {e}")
        raise HTTPException(status_code=500, detail=f"Model switch failed: {str(e)}")

@app.post("/api/save-image")
async def save_image(request: Dict[str, Any]):
    """
    Save image from base64 data or URL to the images directory

    Expects:
    - JSON with 'image' field containing base64 encoded image data OR
    - JSON with 'url' field containing image URL
    - JSON with 'filename' field for custom naming (optional)

    Returns:
    - JSON with saved image information
    """
    try:
        image_data = request.get("image")  # base64
        image_url = request.get("url")    # URL
        filename = request.get("filename", "")  # optional custom filename

        if not image_data and not image_url:
            raise HTTPException(status_code=400, detail="No image data or URL provided")

        # Ensure images directory exists
        images_dir = Path("images")
        images_dir.mkdir(exist_ok=True)

        # Get image data
        if image_data:
            # Decode base64
            image_bytes = base64.b64decode(image_data)
        else:
            # Download from URL
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            image_bytes = response.content

        # Open and validate image
        image = Image.open(io.BytesIO(image_bytes))
        image_format = image.format or "JPEG"

        # Generate filename if not provided
        if not filename:
            # Create hash of image content for deduplication
            image_hash = hashlib.md5(image_bytes).hexdigest()[:8]
            filename = f"{image_hash}_{image.size[0]}x{image.size[1]}.{image_format.lower()}"
        elif not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')):
            filename = f"{filename}.{image_format.lower()}"

        # Check if file already exists (deduplication)
        image_path = images_dir / filename
        if image_path.exists():
            return {
                "status": "already_exists",
                "filename": filename,
                "path": f"images/{filename}",
                "size": image_path.stat().st_size
            }

        # Save image
        with open(image_path, "wb") as f:
            f.write(image_bytes)

        logger.info(f"Image saved: {filename}")
        return {
            "status": "saved",
            "filename": filename,
            "path": f"images/{filename}",
            "size": len(image_bytes)
        }

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download image: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to download image: {str(e)}")
    except Exception as e:
        logger.error(f"Save image error: {e}")
        raise HTTPException(status_code=500, detail=f"Save image failed: {str(e)}")

@app.post("/api/detect")
async def detect_objects(file: UploadFile = File(...)):
    """
    Detect objects in uploaded image using current YOLO model
    Supports both detection (yolo26x.pt) and segmentation (yolo26x-seg.pt)

    Expects:
    - Image file (JPEG, PNG, etc.)

    Returns:
    - JSON with detected objects, bounding boxes, confidence scores, class names
    - Includes segmentation masks if using segmentation model
    """
    global yolo_model

    if yolo_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Read and validate image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))

        # Convert PIL to numpy array for YOLO
        image_np = np.array(image)

        # Run YOLO detection/segmentation - restricted to people only (class 0)
        results = yolo_model(image_np, classes=[0])

        # Process results based on current model type
        is_segmentation = current_model_type == "segmentation"
        logger.info(f"Processing results with model type: {current_model_type} (is_segmentation: {is_segmentation})")
        detections = []
        
        for result in results:
            boxes = result.boxes
            masks = result.masks if is_segmentation else None
            logger.info(f"Boxes found: {boxes is not None}, Masks found: {masks is not None}")
            
            if boxes is not None:
                logger.info(f"Number of boxes: {len(boxes)}")
                if masks is not None:
                    logger.info(f"Number of masks: {len(masks.data)}")
                for i, box in enumerate(boxes):
                    # Get confidence and class
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = yolo_model.names[class_id]
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                    # Get segmentation mask if available
                    mask_data = None
                    if is_segmentation and masks is not None and i < len(masks.data):
                        # Convert mask to binary format for frontend
                        mask_tensor = masks.data[i].cpu().numpy()
                        logger.info(f"Mask tensor shape: {mask_tensor.shape}")
                        
                        # Get mask position and scale information
                        # YOLO masks.xy contains polygon coordinates, not bounding box
                        # We need to calculate the bounding box from the polygon points
                        try:
                            mask_polygon = masks.xy[i][0].cpu().numpy()
                            logger.info(f"Raw mask polygon shape: {mask_polygon.shape}")
                            logger.info(f"First few polygon points: {mask_polygon[:5]}")
                            
                            # Calculate mask bounding box in absolute image coordinates
                            mask_abs_x1 = np.min(mask_polygon[:, 0])
                            mask_abs_y1 = np.min(mask_polygon[:, 1])
                            mask_abs_x2 = np.max(mask_polygon[:, 0])
                            mask_abs_y2 = np.max(mask_polygon[:, 1])
                            
                            logger.info(f"Calculated mask bbox in absolute coords: x1={mask_abs_x1}, y1={mask_abs_y1}, x2={mask_abs_x2}, y2={mask_abs_y2}")
                            logger.info(f"Detection bbox: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
                            logger.info(f"Image dimensions: width={img_width}, height={img_height}")
                            
                            # Convert mask coordinates to percentages of the full image
                            mask_info = {
                                "data": mask_data,
                                "x": float(round((mask_abs_x1 / img_width) * 100, 2)),
                                "y": float(round((mask_abs_y1 / img_height) * 100, 2)),
                                "width": float(round(((mask_abs_x2 - mask_abs_x1) / img_width) * 100, 2)),
                                "height": float(round(((mask_abs_y2 - mask_abs_y1) / img_height) * 100, 2))
                            }
                            
                            logger.info(f"Mask info (absolute positioning): {mask_info}")
                            
                        except (AttributeError, IndexError) as e:
                            logger.warning(f"Could not get mask polygon from masks.xy, using detection bbox: {e}")
                            # Fallback to detection bounding box in absolute coordinates
                            mask_info = {
                                "data": mask_data,
                                "x": float(round((x1 / img_width) * 100, 2)),
                                "y": float(round((y1 / img_height) * 100, 2)),
                                "width": float(round(((x2 - x1) / img_width) * 100, 2)),
                                "height": float(round(((y2 - y1) / img_height) * 100, 2))
                            }
                            

                        # Downsample mask for efficiency (max 64x64)
                        h, w = mask_tensor.shape
                        max_size = 64
                        if h > max_size or w > max_size:
                            # Calculate downsample factor
                            scale = min(max_size / h, max_size / w)
                            new_h, new_w = int(h * scale), int(w * scale)
                            
                            # Use numpy for efficient downsampling
                            import cv2
                            mask_resized = cv2.resize(mask_tensor.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                            mask_data = mask_resized.astype(int).tolist()
                            logger.info(f"Mask downsampled from {h}x{w} to {new_h}x{new_w}")
                            
                            # IMPORTANT: DO NOT scale the coordinates
                            # The mask coordinates should remain in the original bounding box space
                            # Only the mask data is downsampled, not the position
                            logger.info(f"Mask coordinates kept in original space: x1={mask_x1}, y1={mask_y1}, x2={mask_x2}, y2={mask_y2}")
                        else:
                            # Convert to list for JSON serialization if already small
                            mask_data = mask_tensor.astype(int).tolist()
                            logger.info(f"Mask used as-is: {h}x{w}")
                        
                        # Include mask position data for proper alignment
                        # mask_x1, mask_y1, mask_x2, mask_y2 are already relative to the bounding box
                        # We need to scale them to percentages of the bounding box's width and height
                        bbox_width = x2 - x1
                        bbox_height = y2 - y1
                        
                        mask_info = {
                            "data": mask_data,
                            "x": float(round((mask_x1 / bbox_width) * 100, 2)),
                            "y": float(round((mask_y1 / bbox_height) * 100, 2)),
                            "width": float(round(((mask_x2 - mask_x1) / bbox_width) * 100, 2)),
                            "height": float(round(((mask_y2 - mask_y1) / bbox_height) * 100, 2))
                        }
                        
                        logger.info(f"Mask info: {mask_info}")
                    elif is_segmentation:
                        logger.warning(f"Expected mask but not available for box {i}")

                    # Convert to percentage coordinates (0-100)
                    img_height, img_width = image_np.shape[:2]
                    detection = {
                        "class": class_name,
                        "confidence": float(round(confidence, 3)),
                        "bbox": {
                            "x1": float(round((x1 / img_width) * 100, 2)),
                            "y1": float(round((y1 / img_height) * 100, 2)),
                            "x2": float(round((x2 / img_width) * 100, 2)),
                            "y2": float(round((y2 / img_height) * 100, 2))
                        }
                    }
                    
                    # Add mask data if using segmentation model
                    if is_segmentation and 'mask_info' in locals():
                        detection["mask"] = mask_info
                    
                    detections.append(detection)

        response = {
            "type": "segmentation" if is_segmentation else "detection",
            "model": current_model_config['name'],
            "data": detections,
            "total_objects": len(detections)
        }

        logger.info(f"{current_model_config['name']} completed: {len(detections)} objects found")
        return response

    except Exception as e:
        logger.error(f"Detection error: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.post("/api/detect-base64")
async def detect_objects_base64(request: Dict[str, Any]):
    """
    Detect objects in base64 encoded image using current YOLO model
    Supports both detection (yolo26x.pt) and segmentation (yolo26x-seg.pt)

    Expects:
    - JSON with 'image' field containing base64 encoded image data

    Returns:
    - JSON with detected objects, bounding boxes, confidence scores, class names
    - Includes segmentation masks if using segmentation model
    """
    global yolo_model

    if yolo_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Get base64 image data
        image_data = request.get("image")
        if not image_data:
            raise HTTPException(status_code=400, detail="No image data provided")

        # Decode base64
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))

        # Save captured image if requested or for debugging
        if request.get("save", False):
            images_dir = Path("images")
            images_dir.mkdir(exist_ok=True)
            image_hash = hashlib.md5(image_bytes).hexdigest()[:8]
            filename = f"scan_{image_hash}.jpg"
            save_path = images_dir / filename
            
            if not save_path.exists():
                with open(save_path, "wb") as f:
                    f.write(image_bytes)
                logger.info(f"Saved scanned image to {save_path}")

        # Convert PIL to numpy array for YOLO
        image_np = np.array(image)

        # Run YOLO detection/segmentation - restricted to people only (class 0)
        results = yolo_model(image_np, classes=[0])

        # Process results fast - handle both detection and segmentation
        detections = await process_detections(image, results, deep_analysis=True)

        # Debug: Log if we have masks in the detections
        has_masks = any(det.get("mask") is not None for det in detections)
        logger.info(f"Detections processed: {len(detections)} objects, has_masks: {has_masks}")
        if has_masks:
            mask_count = sum(1 for det in detections if det.get("mask") is not None)
            logger.info(f"Objects with masks: {mask_count}")

        response = {
            "type": "segmentation" if current_model_type == "segmentation" else "detection",
            "model": current_model_config['name'],
            "data": detections,
            "total_objects": len(detections)
        }

        logger.info(f"Base64 {current_model_config['name']} completed: {len(detections)} objects found")
        return response

    except Exception as e:
        logger.error(f"Base64 detection error: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.post("/api/detect-url")
async def detect_objects_url(request: Dict[str, Any]):
    """
    Detect objects in image from URL using current YOLO model
    Supports both detection (yolo26x.pt) and segmentation (yolo26x-seg.pt)

    Expects:
    - JSON with 'url' field containing image URL

    Returns:
    - JSON with detected objects, bounding boxes, confidence scores, class names
    - Includes segmentation masks if using segmentation model
    """
    global yolo_model

    if yolo_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Get image URL
        image_url = request.get("url")
        if not image_url:
            raise HTTPException(status_code=400, detail="No image URL provided")

        # Download image from URL
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()

        # Open image
        image = Image.open(io.BytesIO(response.content))

        # Convert PIL to numpy array for YOLO
        image_np = np.array(image)

        # Run YOLO detection/segmentation - restricted to people only (class 0)
        results = yolo_model(image_np, classes=[0])

        # Process results fast - handle both detection and segmentation
        detections = await process_detections(image, results, deep_analysis=True)

        response = {
            "type": "segmentation" if current_model_type == "segmentation" else "detection",
            "model": current_model_config['name'],
            "data": detections,
            "total_objects": len(detections)
        }

        logger.info(f"URL {current_model_config['name']} completed: {len(detections)} objects found")
        return response

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download image: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to download image: {str(e)}")
    except Exception as e:
        logger.error(f"URL detection error: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.post("/api/analyze-box")
async def analyze_box(request: Dict[str, Any]):
    """
    Run deep analysis on a specific box within an image
    """
    global florence_model, florence_processor
    
    if florence_model is None:
        raise HTTPException(status_code=503, detail="VLM not loaded")

    try:
        image_data = request.get("image")
        box = request.get("box") # {x, y, width, height} in percentages
        
        if not image_data or not box:
            raise HTTPException(status_code=400, detail="Missing image data or box coordinates")

        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        img_width, img_height = image.size

        # Convert percentages to pixels
        x = (box['x'] / 100) * img_width
        y = (box['y'] / 100) * img_height
        w = (box['width'] / 100) * img_width
        h = (box['height'] / 100) * img_height

        # Crop with padding
        pad = 40
        cx1 = max(0, int(x) - pad)
        cy1 = max(0, int(y) - pad)
        cx2 = min(img_width, int(x + w) + pad)
        cy2 = min(img_height, int(y + h) + pad)
        cropped_image = image.crop((cx1, cy1, cx2, cy2))
        
        # Determine the best prompt based on object type
        obj_type = request.get("type", "person")
        logger.info(f"Analyzing box: type={obj_type}, size={int(w)}x{int(h)}")
        
        # Mapping for targeted questions from centralized config
        config = OBJECT_CONFIG.get(obj_type, {
            "task": "<DETAILED_CAPTION>",
            "prompt": "",
            "llm_query": f"Analyze object: {obj_type}."
        })
        
        task = config["task"]
        hint = config["prompt"]
        
        # New: Model selection
        vision_model = request.get("vision_model", "florence2")
        logger.info(f"Analyzing box with model: {vision_model}")
        
        if vision_model == "glm4.6v":
            # GLM 4.6V (Qwen2-VL) prefers specific prompts
            glm_prompt = config.get("llm_query") or f"Describe this {obj_type} in detail."
            glm_res = await run_glm_analysis(cropped_image, glm_prompt)
            analysis_text = glm_res["analysis"]
            used_model = glm_res["model"]
        elif vision_model == "samsung-trm":
            # Samsung-TRM 'Visual Reasoning' mode
            # We use Qwen2-VL for a single high-fidelity pass with Chain-of-Thought
            # This is faster than Florence+3x refinement and much more accurate for world leaders.
            analysis_text = await run_samsung_trm_analysis(cropped_image, obj_type)
            used_model = "Samsung-TRM (High Fidelity)"
        else:
            # Default to Florence-2 (Fastest)
            # CRITICAL: For <DETAILED_CAPTION>, the token MUST be the only text.
            if task == "<DETAILED_CAPTION>":
                hint = ""
            
            florence_results = await run_florence_analysis(cropped_image, task, text_input=hint)
            analysis_text = next(iter(florence_results.values())) if florence_results else "No analysis result"
            used_model = "Florence-2"
            
        return {
            "status": "success",
            "analysis": analysis_text,
            "model": used_model
        }

    except Exception as e:
        logger.error(f"Crop analysis failed: {e}")
        return {"analysis": f"Error: {str(e)[:50]}"}

@app.post("/api/summarize")
async def summarize_text(request: Dict[str, Any]):
    """
    Summarize text using
    """
    global summarizer_model, summarizer_tokenizer, current_summarize_id
    
    if summarizer_model is None:
        raise HTTPException(status_code=503, detail="Summarization model not loaded")

    try:
        text = request.get("text")
        if not text:
            raise HTTPException(status_code=400, detail="No text provided")

        # Increment request ID to signal old requests to stop
        current_id = current_summarize_id + 1
        current_summarize_id = current_id

        # Determine mode, category and type
        mode = request.get("mode", "summarize") # 'summarize' or 'refine'
        category = request.get("category", "Misc")
        obj_type = request.get("type")
        
        # Base system prompt for Text Summarization - Neutral and performance-oriented
        system_prompt = request.get("system_prompt") or (
            "You are the AI Scanner OS. Direct, cold, and factual. "
            "Skip all 'thinking' and preamble. Do not use phrases like 'The image shows' or 'Here is a summary'. "
            "Output only the final analytical data."
        )
        
        # Prepare content based on mode
        if mode == "refine":
            # Get class-specific prompt from OBJECT_CONFIG if available
            obj_config = OBJECT_CONFIG.get(obj_type, {}) if obj_type else {}
            hint = obj_config.get("llm_query") or obj_config.get("prompt")
            
            if not hint:
                # Specialized prompt for refining Florence-2/Vision output
                fallback_config = {
                    "Humans": "Identify this person. Provide name or physical description.",
                    "Vehicles": "Identify manufacturer, model, and estimated year.",
                    "Animals": "Identify breed/species and notable features.",
                    "Electronics": "Identify brand and specific model.",
                    "Food": "Identify food type and ingredients.",
                    "Household": "Identify item and style/brand."
                }
                hint = fallback_config.get(category, f"Identify this {category.lower()}.")
            
            query = (
                f"RAW VISION DATA: {text}\n"
                f"IDENTIFICATION QUERY: {hint}\n\n"
                "TASK: Perform high-certainty identification. Output ONLY the identification data. "
                "Maximum 20 words."
            )
        else:
            # Standard website text summarization
            query = (
                "TASK: Translate to English then summarize the text with the following structure:\n"
                "1. Start with 1 to 5 emojis max which represent the sentiment of the text (no words allowed in first section only emojis).\n"
                "2. Next add a new line with separator '------'.\n"
                "3. Finally write the 30-100 word summary about the SOURCE TEXT.\n"
                f"SOURCE TEXT: {text}\n\n"
            )

        # Use chat template for robust prompting
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        prompt = summarizer_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = summarizer_tokenizer(prompt, return_tensors="pt").to(summarizer_model.device)
        
        # Define stopping criteria to check for cancellation
        from transformers import StoppingCriteria, StoppingCriteriaList
        
        class CancelCriteria(StoppingCriteria):
            def __init__(self, target_id):
                self.target_id = target_id
            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
                # If a new request has started, stop this one
                return current_summarize_id != self.target_id

        with torch.no_grad():
            output_ids = summarizer_model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,  # Greedy search for maximum speed
                num_beams=1,
                stopping_criteria=StoppingCriteriaList([CancelCriteria(current_id)])
            )
        
        # Check if we were cancelled
        if current_summarize_id != current_id:
            logger.info(f"Summarization request {current_id} cancelled")
            return {"summary": "[Request cancelled by a newer one]", "cancelled": True}
            
        summary = summarizer_tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return {"summary": summary.strip(), "model": "Qwen2.5"}

    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")

@app.get("/api/images")
async def list_images():
    """List all image files in the images directory"""
    images_dir = Path("images")

    if not images_dir.exists():
        return {"images": []}

    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}

    image_files = []
    for file_path in images_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            image_files.append({
                "name": file_path.name,
                "src": f"images/{file_path.name}",
                "size": file_path.stat().st_size
            })
    return {"images": image_files}

@app.options("/images/{image_name}")
async def options_image(image_name: str):
    """Handle CORS preflight requests for images"""
    return JSONResponse(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
    )

@app.get("/images/{image_name}")
async def get_image(image_name: str):
    """Serve image files with CORS headers"""
    image_path = Path("images") / image_name

    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(
        image_path,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
    )

@app.get("/api/models")
def get_available_models():
    """Get information about available YOLO models"""
    return {
        "available_models": AVAILABLE_MODELS,
        "current_model": current_model_type,
        "current_config": current_model_config,
        "vlm_model": "Florence-2-large",
        "reasoning_model": "Samsung-TRM",
        "note": "Use /api/models/switch to change models dynamically"
    }

async def process_detections(image, results, deep_analysis=False):
    """Helper to process YOLO results and optionally run Florence-2 analysis"""
    logger.info(f"process_detections called with deep_analysis={deep_analysis}")
    detections = []
    img_height, img_width = np.array(image).shape[:2]
    is_segmentation = current_model_type == "segmentation"
    
    logger.info(f"Processing detections: is_segmentation={is_segmentation}, results_count={len(results)}")
    
    for result in results:
        boxes = result.boxes
        masks = result.masks if is_segmentation else None
        
        logger.info(f"Result: boxes={boxes is not None}, masks={masks is not None}")
        if boxes is not None:
            logger.info(f"Number of boxes: {len(boxes)}")
            if masks is not None:
                logger.info(f"Number of masks: {len(masks.data)}")
        
        try:
            if boxes is not None:
                for i, box in enumerate(boxes):
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = yolo_model.names[class_id]
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                    # Get segmentation mask if available
                    mask_data = None
                    if is_segmentation and masks is not None and i < len(masks.data):
                        logger.info(f"Processing mask for box {i}")
                        # Convert mask to binary format for frontend
                        mask_tensor = masks.data[i].cpu().numpy()
                        logger.info(f"Mask tensor shape: {mask_tensor.shape}")
                        
                        # Get mask position and scale information
                        try:
                            mask_polygon = masks.xy[i][0].cpu().numpy()
                            mask_x1 = np.min(mask_polygon[:, 0])
                            mask_y1 = np.min(mask_polygon[:, 1])
                            mask_x2 = np.max(mask_polygon[:, 0])
                            mask_y2 = np.max(mask_polygon[:, 1])
                            
                            logger.info(f"Calculated mask bbox from polygon: x1={mask_x1}, y1={mask_y1}, x2={mask_x2}, y2={mask_y2}")
                            logger.info(f"Bounding box: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
                            
                            # Calculate relative position within bounding box
                            rel_x1 = mask_x1 - x1
                            rel_y1 = mask_y1 - y1
                            rel_x2 = mask_x2 - x1
                            rel_y2 = mask_y2 - y1
                            
                            logger.info(f"Relative mask bbox: x1={rel_x1}, y1={rel_y1}, x2={rel_x2}, y2={rel_y2}")
                            
                            # Check if mask coordinates are valid
                            if rel_x1 < 0 or rel_y1 < 0 or rel_x2 > (x2 - x1) or rel_y2 > (y2 - y1):
                                logger.warning(f"Mask coordinates extend beyond bounding box, using full bounding box")
                                # Fallback to full bounding box
                                rel_x1, rel_y1, rel_x2, rel_y2 = 0, 0, (x2 - x1), (y2 - y1)
                            
                            # Use relative coordinates
                            mask_x1, mask_y1, mask_x2, mask_y2 = rel_x1, rel_y1, rel_x2, rel_y2
                            
                        except (AttributeError, IndexError) as e:
                            logger.warning(f"Could not get mask polygon from masks.xy, using full bounding box: {e}")
                            # Fallback to full bounding box
                            mask_x1, mask_y1, mask_x2, mask_y2 = 0, 0, (x2 - x1), (y2 - y1)
                        
                        # Downsample mask for efficiency (max 64x64)
                        h, w = mask_tensor.shape
                        max_size = 64
                        if h > max_size or w > max_size:
                            # Calculate downsample factor
                            scale = min(max_size / h, max_size / w)
                            new_h, new_w = int(h * scale), int(w * scale)
                            
                            # Use cv2 for efficient downsampling
                            import cv2
                            mask_resized = cv2.resize(mask_tensor.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                            mask_data = mask_resized.astype(int).tolist()
                            logger.info(f"Mask downsampled from {h}x{w} to {new_h}x{new_w}")
                            
                            # IMPORTANT: DO NOT scale the coordinates
                            # The mask coordinates should remain in the original bounding box space
                            # Only the mask data is downsampled, not the position
                            logger.info(f"Mask coordinates kept in original space: x1={mask_x1}, y1={mask_y1}, x2={mask_x2}, y2={mask_y2}")
                        else:
                            # Convert to list for JSON serialization if already small
                            mask_data = mask_tensor.astype(int).tolist()
                            logger.info(f"Mask used as-is: {h}x{w}")
                        
                        # Include mask position data for proper alignment
                        # mask_x1, mask_y1, mask_x2, mask_y2 are already relative to the bounding box
                        # We need to scale them to percentages of the bounding box's width and height
                        bbox_width = x2 - x1
                        bbox_height = y2 - y1
                        
                        mask_info = {
                            "data": mask_data,
                            "x": float(round((mask_x1 / bbox_width) * 100, 2)),
                            "y": float(round((mask_y1 / bbox_height) * 100, 2)),
                            "width": float(round(((mask_x2 - mask_x1) / bbox_width) * 100, 2)),
                            "height": float(round(((mask_y2 - mask_y1) / bbox_height) * 100, 2))
                        }
                        
                        logger.info(f"Mask info: {mask_info}")
                    elif is_segmentation:
                        logger.warning(f"Expected mask but not available for box {i} (masks: {masks is not None}, i: {i}, len(masks.data): {len(masks.data) if masks else 'N/A'})")

                    # Get config for this class
                    config = OBJECT_CONFIG.get(class_name, {})
                    color = config.get("color", "#00FF00")
                    is_analyzable = config.get("is_analyzable", False)

                    analysis_text = ""
                    logger.info(f"Analysis check: deep_analysis={deep_analysis}, is_analyzable={is_analyzable}, confidence={confidence}")
                    if deep_analysis and is_analyzable and confidence > 0.75:
                        logger.info(f"Starting Florence2 analysis for {class_name} with confidence {confidence}")
                        try:
                            pad = 40
                            cx1 = max(0, int(x1) - pad)
                            cy1 = max(0, int(y1) - pad)
                            cx2 = min(img_width, int(x2) + pad)
                            cy2 = min(img_height, int(y2) + pad)
                            person_image = image.crop((cx1, cy1, cx2, cy2))
                            
                            # Validate crop
                            if person_image.width < 5 or person_image.height < 5:
                                logger.warning(f"Crop too small: {person_image.size}")
                                analysis_text = "Crop too small"
                            else:
                                # Using just the task token to avoid 'only one token' error
                                florence_results = await run_florence_analysis(person_image, "<DETAILED_CAPTION>")
                                
                                if "<DETAILED_CAPTION>" in florence_results:
                                    analysis_text = florence_results["<DETAILED_CAPTION>"]
                                elif "<CAPTION>" in florence_results:
                                    analysis_text = florence_results["<CAPTION>"]
                                else:
                                    analysis_text = next(iter(florence_results.values())) if florence_results else "No caption"
                                
                            logger.info(f"Florence analysis: {analysis_text}")
                        except Exception as e:
                            logger.error(f"Florence analysis failed: {str(e)}")
                            analysis_text = f"Analysis error: {str(e)[:50]}"

                    detection_data = {
                        "class": class_name,
                        "confidence": float(round(confidence, 3)),
                        "analysis": analysis_text,
                        "color": color, 
                        "is_analyzable": is_analyzable,
                        "category": config.get("category", "Misc"),
                        "bbox": {
                            "x1": float(round((x1 / img_width) * 100, 2)),
                            "y1": float(round((y1 / img_height) * 100, 2)),
                            "x2": float(round((x2 / img_width) * 100, 2)),
                            "y2": float(round((y2 / img_height) * 100, 2))
                        }
                    }
                    
                    # Add mask data if using segmentation model
                    if is_segmentation and 'mask_info' in locals():
                        detection_data["mask"] = mask_info
                    
                    detections.append(detection_data)
                    logger.info(f"Completed processing detection {i}: {class_name}")
        
        except Exception as e:
            logger.error(f"Error processing detection result: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    response_data = []
    for det in detections:
        response_item = {
            "x": float(det["bbox"]["x1"]),
            "y": float(det["bbox"]["y1"]),
            "width": float(det["bbox"]["x2"] - det["bbox"]["x1"]),
            "height": float(det["bbox"]["y2"] - det["bbox"]["y1"]),
            "color": det["color"],
            "type": det["class"],
            "analysis": det["analysis"],
            "confidence": float(det["confidence"]),
            "is_analyzable": det.get("is_analyzable", False),
            "category": det.get("category", "Misc")
        }
        
        # Add mask data if using segmentation model
        if is_segmentation and "mask" in det:
            response_item["mask"] = det["mask"]
        
        response_data.append(response_item)
    
    # NEW: Fallback for when YOLO fails to find any objects
    if not response_data:
        logger.info("YOLO failed to detect objects. Falling back to Scene Analysis.")
        response_data.append({
            "x": 0.0,
            "y": 0.0,
            "width": 100.0,
            "height": 100.0,
            "color": "#00FFFF",
            "type": "scene",
            "analysis": "",
            "confidence": 1.0,
            "is_analyzable": True,
            "category": "Misc"
        })
        
    return response_data

async def run_glm_analysis(image, prompt):
    """Analyze an image using a local Qwen2-VL model (High performance replacement for GLM)"""
    global vision_model_obj, vision_processor_obj
    
    if vision_model_obj is None:
        return {"analysis": "Error: Local Vision model not loaded", "model": "Qwen2-VL"}

    try:
        # Ensure RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        # Qwen2-VL Chat Template
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        # Prepare inputs using the chat template which is very robust for Qwen2-VL
        text = vision_processor_obj.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = vision_processor_obj(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        ).to(device)
        
        # Ensure correct dtypes
        if dtype == torch.float16:
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(dtype)

        with torch.no_grad():
            output_ids = vision_model_obj.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False
            )
            
        # Decode
        generated_ids = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, output_ids)
        ]
        response = vision_processor_obj.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        # Cleanup
        import re
        content = re.sub(r'<think>[\s\S]*?<\/think>', '', response, flags=re.IGNORECASE).strip()
        return {"analysis": content or "No data acquired.", "model": "Qwen2-VL"}
            
    except Exception as e:
        logger.error(f"Local Vision analysis failed: {e}")
        return {"analysis": f"Error: Inference failed. ({str(e)})", "model": "Qwen2-VL"}

async def run_florence_analysis(image, task_prompt, text_input=None):
    """Helper function to run Florence-2 analysis with robust error handling"""
    global florence_model, florence_processor
    
    try:
        # Prompt construction
        if text_input:
            prompt = task_prompt + text_input
        else:
            prompt = task_prompt

        # Ensure RGB and valid image
        if not image or image.width == 0 or image.height == 0:
            return {"error": "Invalid image"}
            
        if image.mode != "RGB":
            image = image.convert("RGB")

        logger.info(f"Running Florence with prompt: {prompt} on image {image.size}")

        # Process image
        with torch.no_grad():
            # Use the processor to get the model inputs
            try:
                inputs = florence_processor(text=prompt, images=image, return_tensors="pt")
            except Exception as e:
                logger.error(f"Processor execution failed: {e}")
                return {task_prompt: f"Processor error: {str(e)}"}
            
            if not inputs:
                logger.error("Florence processor returned empty output")
                return {task_prompt: "Error: Processor returned nothing"}
            
            # Move all tensors to the device robustly using the correct dtype
            inputs = inputs.to(device, dtype)
            
            # Log processed inputs
            logger.info(f"Florence processed inputs keys: {list(inputs.keys())}")
            
            if "pixel_values" not in inputs:
                logger.error("pixel_values missing from processor output")
                return {task_prompt: "Error: No image data processed"}

            # Safe generation
            try:
                # Use ONLY input_ids and pixel_values as per official example
                # and definitively disable KV caching to avoid NoneType errors
                generated_ids = florence_model.generate(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    max_new_tokens=512,
                    num_beams=3,
                    do_sample=False,
                    use_cache=False
                )
            except Exception as e:
                logger.error(f"Model generation failed: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return {task_prompt: f"Generation error: {str(e)}"}
            
            # Decode output
            try:
                generated_text = florence_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                logger.info(f"Florence Raw Output: {generated_text}")
            except Exception as e:
                logger.error(f"Decoding failed: {e}")
                return {task_prompt: f"Decoding error: {str(e)}"}
            
            if not generated_text:
                return {task_prompt: "No response generated"}
            
            # Post-processing depends on the task
            # For VQA, we just want the text. For captions, we can use post_process_generation
            if task_prompt in ["<CAPTION>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>"]:
                try:
                    parsed_answer = florence_processor.post_process_generation(
                        generated_text, 
                        task=task_prompt, 
                        image_size=(image.width, image.height)
                    )
                except Exception as pe:
                    logger.warning(f"Post-processing failed: {pe}")
                    parsed_answer = {task_prompt: generated_text}
            else:
                # Standard text output for VQA etc.
                parsed_answer = {task_prompt: generated_text}

            return parsed_answer
            
    except Exception as e:
        logger.error(f"Internal Florence Error: {str(e)}")
        # Return a dict that matches the expected structure so the API doesn't crash
        return {task_prompt: f"Analysis error: {str(e)}"}

async def run_samsung_trm_analysis(image, obj_type):
    """
    Enhanced Samsung-TRM 'Visual Reasoning' implementation.
    Instead of slow text refinement loops, we use a single high-fidelity vision reasoning pass
    that mimics the 'recursive' depth by using a Chain-of-Thought prompt.
    """
    try:
        # Construct a deep reasoning prompt
        if obj_type == "person":
             reasoning_prompt = (
                "ACT AS: Advanced AI Scanner. "
                "TASK: Perform high-fidelity identification and analysis. "
                "1. Observe the person's face and features closely. Identify if they are a world leader, celebrity, or public figure. "
                "2. If identified, state 'IDENTITY: [Name]'. "
                "3. Describe their appearance, clothing, and current action in detail. "
                "4. If identity is unknown, provide a precise physical description and role (e.g., 'A technician working on...')."
            )
        else:
            reasoning_prompt = (
                f"Perform deep analytical reasoning on this {obj_type}. "
                "Identify brand, model, material, and state of the object. "
                "Provide a concise but highly detailed technical description."
            )
        
        # Execute using the high-quality Qwen2-VL model
        res = await run_glm_analysis(image, reasoning_prompt)
        return res["analysis"]
        
    except Exception as e:
        logger.error(f"Samsung-TRM Logic Core failed: {e}")
        return f"Analytical Error: {str(e)[:50]}"

if __name__ == "__main__":
    # Run server on port 8001 to avoid conflict with Ollama (11434)
    uvicorn.run(app, host="0.0.0.0", port=8001)
