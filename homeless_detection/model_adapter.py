import torch
import os
import logging
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

# Set up logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelAdapter:
    """Adapter for the homeless detection model"""
    
    def __init__(self):
        self.model = None
        self.device = None
        # Updated transform to match notebook configuration
        self.transform = transforms.Compose([
            transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Fixed class mapping matching the notebook
        self.class_map = {
            1: "Homeless_People",
            2: "Homeless_Encampments",
            3: "Homeless_Cart",
            4: "Homeless_Bike"
        }
        
        # Color mapping for visualization
        self.color_map = {
            1: (255, 0, 0),    # Red for people
            2: (0, 255, 0),    # Green for encampments
            3: (0, 0, 255),    # Blue for carts
            4: (255, 255, 0)   # Yellow for bikes
        }
    
    def load_model(self, model_path, device=None):
        """Load model from path"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
            
        logger.info(f"Loading model from: {model_path}")
        
        # Set device with better error handling
        try:
            if device is None:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = device
            logger.info(f"Using device: {self.device}")
            
            # Create model matching notebook configuration
            logger.info("Creating model with FasterRCNN ResNet50 FPN V2")
            self.model = fasterrcnn_resnet50_fpn_v2(pretrained=True)
            
            # Replace the classifier head
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(self.class_map) + 1)
            
            # Load weights with better error handling
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                logger.info("Model weights loaded successfully")
            except Exception as e:
                logger.error(f"Error loading model weights: {str(e)}")
                raise
            
            # Set model to evaluation mode
            self.model.to(self.device)
            self.model.eval()
            logger.info("Model setup completed successfully")
            
            return self.model, self.device, {}
            
        except Exception as e:
            logger.error(f"Error in model setup: {str(e)}")
            raise
    
    def predict(self, image, confidence_threshold=0.3):
        """Run prediction on an image"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        try:
            # Convert string path to PIL Image
            if isinstance(image, str):
                if not os.path.exists(image):
                    raise FileNotFoundError(f"Image file not found: {image}")
                image = Image.open(image).convert("RGB")
            elif not isinstance(image, Image.Image):
                raise ValueError("Input must be either a file path or a PIL Image")
            
            # Apply transforms
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                predictions = self.model(img_tensor)
                prediction = predictions[0]
                
                # Extract predictions
                pred_boxes = prediction["boxes"].cpu().numpy()
                pred_labels = prediction["labels"].cpu().numpy()
                pred_scores = prediction["scores"].cpu().numpy()
                
                # Filter by confidence
                mask = pred_scores >= confidence_threshold
                pred_boxes = pred_boxes[mask]
                pred_labels = pred_labels[mask]
                pred_scores = pred_scores[mask]
                
                logger.debug(f"Found {len(pred_boxes)} predictions above threshold {confidence_threshold}")
                return pred_boxes, pred_labels, pred_scores
                
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            # Return empty arrays instead of raising to prevent pipeline breaks
            return np.array([]), np.array([]), np.array([])
    
    def get_class_map(self):
        """Get class mapping"""
        return self.class_map
    
    def get_color_map(self):
        """Get color mapping"""
        return self.color_map

def get_model_adapter():
    """Get model adapter instance"""
    return ModelAdapter()

def draw_predictions(image, boxes, labels, scores, score_thresh=0.5):
    """Draw predictions on image"""
    from PIL import ImageDraw, ImageFont
    
    # Get a new model adapter instance for color mapping
    adapter = get_model_adapter()
    color_map = adapter.get_color_map()
    class_map = adapter.get_class_map()
    
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except:
        font = ImageFont.load_default()
    
    for i, box in enumerate(boxes):
        if scores[i] < score_thresh:
            continue
            
        x1, y1, x2, y2 = map(int, box)
        cls_id = int(labels[i])
        color = color_map.get(cls_id, (255, 255, 255))
        
        # Draw box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        
        # Draw label with class name
        class_name = class_map.get(cls_id, f"Class {cls_id}")
        text = f"{class_name}: {scores[i]:.2f}"
        text_size = draw.textbbox((0, 0), text, font=font)[2:4]
        draw.rectangle([x1, y1 - text_size[1], x1 + text_size[0], y1], fill=color)
        draw.text((x1, y1 - text_size[1]), text, fill=(0, 0, 0) if cls_id in [2, 4] else (255, 255, 255), font=font)
    
    return image 