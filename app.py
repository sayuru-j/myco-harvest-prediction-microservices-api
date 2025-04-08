import io
import os
import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from PIL import Image

app = FastAPI(title="Pink Oyster Mushroom Harvest Prediction API")

# Class names for the pink oyster mushroom harvest stages
CLASS_NAMES = [
    'Pink Oyster- 2-3 days to harvest', 
    'Pink Oyster- 4-5 days to harvest', 
    'Pink Oyster- 6-7 days to harvest', 
    'Pink Oyster- Ready to harvest'
]

# Get model path from environment variable or use default
MODEL_PATH = os.environ.get("MODEL_PATH", "pink_oyster_harvest_model.onnx")

# Define global session variable
session = None
input_name = None
output_names = None

class PredictionResult(BaseModel):
    class_counts: dict
    recommendation: str
    confidence_scores: dict

def load_model():
    """Load the ONNX model into a global session"""
    global session, input_name, output_names
    
    try:
        print(f"Loading model from {MODEL_PATH}")
        session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
        
        # Get input and output details
        model_inputs = session.get_inputs()
        input_name = model_inputs[0].name
        output_names = [output.name for output in session.get_outputs()]
        
        print(f"Model loaded successfully. Input name: {input_name}")
        print(f"Output names: {output_names}")
        
        return session
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

@app.on_event("startup")
async def startup_event():
    """Initialize the model when the application starts"""
    global session, input_name, output_names
    session = load_model()
    
    if session is None:
        print("WARNING: Model failed to load. API will return errors for predictions.")
    else:
        # Print detailed model information
        print("\nModel Inputs:")
        for input in session.get_inputs():
            print(f"  - {input.name}: {input.shape} ({input.type})")
        
        print("\nModel Outputs:")
        for output in session.get_outputs():
            print(f"  - {output.name}: {output.shape} ({output.type})")
        
        # Print model metadata if available
        metadata = session.get_modelmeta()
        if metadata:
            print("\nModel Metadata:")
            print(f"  - Producer: {metadata.producer_name}")
            print(f"  - Graph Name: {metadata.graph_name}")
            print(f"  - Description: {metadata.description}")

def preprocess_image(image_bytes):
    """Preprocess image for ONNX model inference"""
    # Convert to PIL Image
    image = Image.open(io.BytesIO(image_bytes))
    
    # Resize to the expected input shape (640x640 for YOLOv8)
    input_width, input_height = 640, 640
    image = image.resize((input_width, input_height))
    
    # Convert to numpy array and normalize
    img_array = np.array(image) / 255.0
    
    # Ensure the image is in the right format (RGB)
    if len(img_array.shape) == 2:  # Grayscale
        img_array = np.stack([img_array] * 3, axis=2)
    elif img_array.shape[2] == 4:  # RGBA
        img_array = img_array[:, :, :3]
    
    # Transpose to match ONNX input format (batch, channels, height, width)
    img_array = img_array.transpose(2, 0, 1)
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    
    return img_array

def process_yolo_output(outputs, image_size, confidence_threshold=0.3):
    """Process YOLOv8 ONNX model output to get bounding boxes and class predictions"""
    # Add debug logging
    print(f"Model outputs: {len(outputs)} tensors")
    for i, output in enumerate(outputs):
        print(f"Output {i} shape: {output.shape}")
    
    # Check if we have any outputs
    if not outputs or len(outputs) == 0:
        print("Warning: Model returned no outputs")
        return [], [], []
    
    # Get the predictions tensor (YOLOv8 format)
    predictions = outputs[0]  # Shape: [1, 8, 8400]
    print(f"Using predictions with shape: {predictions.shape}")
    
    # Initialize results
    boxes = []
    scores = []
    class_ids = []
    
    try:
        # For YOLOv8, the output is typically [batch, num_predictions, num_boxes * (num_classes + 4)]
        # where the first 4 values are [x, y, w, h] and the rest are class scores
        
        # Transpose from [batch, num_values, num_boxes] to [batch, num_boxes, num_values]
        # This makes it easier to process each detection
        transposed = np.transpose(predictions[0], (1, 0))  # Now shape: [8400, 8]
        
        # Number of classes (total values - 4 box coordinates)
        num_classes = transposed.shape[1] - 4
        
        # Process each potential detection
        for detection in transposed:
            # First 4 values are bounding box coordinates
            box = detection[:4]
            
            # Remaining values are class scores
            class_scores = detection[4:]
            
            # Get highest class score and its index
            class_id = np.argmax(class_scores)
            score = class_scores[class_id]
            
            # Filter by confidence threshold
            if score >= confidence_threshold:
                boxes.append(box.tolist())
                scores.append(float(score))
                class_ids.append(int(class_id))
    
    except Exception as e:
        print(f"Error processing model output: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print(f"Detected {len(boxes)} objects above threshold {confidence_threshold}")
    return boxes, scores, class_ids

def draw_predictions(image_bytes, boxes, scores, class_ids):
    """Draw predictions on the image"""
    # Check if we have any detections
    if not boxes or len(boxes) == 0:
        # Return original image if no detections
        return Image.open(io.BytesIO(image_bytes))
    
    # Convert to PIL Image and then to OpenCV format
    image = Image.open(io.BytesIO(image_bytes))
    image_np = np.array(image)
    
    # Convert to BGR for OpenCV if image is RGB
    if len(image_np.shape) == 3 and image_np.shape[2] == 3:
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    else:
        image_cv = image_np  # Handle grayscale or other formats
    
    # Get image dimensions
    height, width = image_cv.shape[:2]
    
    # Draw each detection
    for i in range(len(boxes)):
        if i >= len(class_ids) or i >= len(scores):
            print(f"Warning: Index mismatch - boxes:{len(boxes)}, scores:{len(scores)}, class_ids:{len(class_ids)}")
            continue
            
        box = boxes[i]
        score = scores[i]
        class_id = class_ids[i]
        
        # Ensure class_id is valid
        if class_id < 0 or class_id >= len(CLASS_NAMES):
            print(f"Warning: Invalid class_id {class_id}, max allowed is {len(CLASS_NAMES)-1}")
            continue
        
        # Convert normalized coordinates to pixel coordinates if needed
        # Assuming box format is [x_center, y_center, width, height]
        x, y, w, h = box
        
        # If coordinates are normalized (between 0 and 1)
        if 0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1:
            x = int(x * width)
            y = int(y * height)
            w = int(w * width)
            h = int(h * height)
        
        # Calculate box corners
        x1 = max(0, int(x - w/2))
        y1 = max(0, int(y - h/2))
        x2 = min(width, int(x + w/2))
        y2 = min(height, int(y + h/2))
        
        # Draw box
        color = (0, 255, 0)  # Green
        cv2.rectangle(image_cv, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        try:
            label = f"{CLASS_NAMES[class_id]}: {score:.2f}"
        except IndexError:
            label = f"Class {class_id}: {score:.2f}"
            
        # Calculate text size to position the background rectangle
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
        )
        
        # Draw background rectangle for text
        cv2.rectangle(
            image_cv,
            (x1, y1 - label_height - 10),
            (x1 + label_width, y1),
            color,
            -1  # Filled rectangle
        )
        
        # Draw text
        cv2.putText(
            image_cv,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),  # Black text
            2
        )
    
    # Convert back to RGB for PIL
    image_result = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image_result)


@app.post("/predict", response_model=PredictionResult)
async def predict(file: UploadFile = File(...)):
    """
    Predict harvest time for pink oyster mushrooms in the uploaded image
    """
    global session, input_name, output_names
    
    # Check if model is loaded
    if session is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please try again later or contact the administrator."
        )
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail=f"File '{file.filename}' is not an image. Only image files are supported."
        )
    
    # Read image file
    try:
        contents = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read file: {str(e)}"
        )
    
    try:
        # Preprocess image
        input_tensor = preprocess_image(contents)
        
        # Run inference
        outputs = session.run(output_names, {input_name: input_tensor})
        
        # Process outputs
        boxes, scores, class_ids = process_yolo_output(
            outputs, 
            (input_tensor.shape[2], input_tensor.shape[3]),
            confidence_threshold=0.3
        )
        
        # Count detections by class
        class_counts = {i: 0 for i in range(len(CLASS_NAMES))}
        confidence_scores = {i: [] for i in range(len(CLASS_NAMES))}
        
        for cls_id, score in zip(class_ids, scores):
            class_counts[cls_id] += 1
            confidence_scores[cls_id].append(score)
        
        # Calculate average confidence per class
        avg_confidence = {}
        for cls_id, scores_list in confidence_scores.items():
            if scores_list:
                avg_confidence[cls_id] = sum(scores_list) / len(scores_list)
            else:
                avg_confidence[cls_id] = 0
        
        # Determine overall recommendation
        if sum(class_counts.values()) > 0:
            max_class = max(class_counts.items(), key=lambda x: x[1])
            recommendation = CLASS_NAMES[max_class[0]]
        else:
            recommendation = "No mushrooms detected"
        
        return PredictionResult(
            class_counts=class_counts,
            recommendation=recommendation,
            confidence_scores=avg_confidence
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/image")
async def predict_with_image(file: UploadFile = File(...)):
    """
    Predict harvest time and return annotated image
    """
    global session, input_name, output_names
    
    # Check if model is loaded
    if session is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please try again later or contact the administrator."
        )
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail=f"File '{file.filename}' is not an image. Only image files are supported."
        )
    
    # Read image file
    try:
        contents = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read file: {str(e)}"
        )
    
    try:
        # Preprocess and run inference
        input_tensor = preprocess_image(contents)
        outputs = session.run(output_names, {input_name: input_tensor})
        
        # Process outputs
        boxes, scores, class_ids = process_yolo_output(
            outputs, 
            (input_tensor.shape[2], input_tensor.shape[3]),
            confidence_threshold=0.3
        )
        
        # Draw predictions on image
        annotated_image = draw_predictions(contents, boxes, scores, class_ids)
        
        # Convert PIL Image to bytes
        img_byte_arr = io.BytesIO()
        annotated_image.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        
        return StreamingResponse(img_byte_arr, media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing error: {str(e)}")

@app.get("/")
def read_root():
    """
    Health check endpoint
    """
    return {
        "message": "MycoMentor Harvest Prediction API is running",
        "model_loaded": session is not None,
        "classes": CLASS_NAMES
    }

# If running as script
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)