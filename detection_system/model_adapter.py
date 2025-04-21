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
    """Adapter for the encampment detection model"""
    
    def __init__(self):
        self.model = None
        self.device = None
        # Updated transform pipeline for better small object detection
        self.transform = transforms.Compose([
            transforms.Resize((800, 800), interpolation=transforms.InterpolationMode.BILINEAR),  # Increased resolution
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Class-specific confidence thresholds
        self.confidence_thresholds = {
            1: 0.3,  # Lower threshold for people
            2: 0.5,  # Higher threshold for encampments
            3: 0.4,  # Medium threshold for carts
            4: 0.3   # Lower threshold for bikes
        }
        
        # Fixed class mapping matching the notebook
        self.class_map = {
            1: "People",
            2: "Encampments",
            3: "Cart",
            4: "Bike"
        }
        
        # Color mapping for visualization
        self.color_map = {
            1: (255, 0, 0),    # Red for people
            2: (0, 255, 0),    # Green for encampments
            3: (0, 0, 255),    # Blue for carts
            4: (255, 255, 0)   # Yellow for bikes
        }
    
    def load_model(self, model_path, device=None):
        """Load model from path with optimized configuration"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
            
        logger.info(f"Loading model from: {model_path}")
        
        try:
            if device is None:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = device
            logger.info(f"Using device: {self.device}")
            
            # Create model with optimized configuration
            self.model = fasterrcnn_resnet50_fpn_v2(
                pretrained=True,
                box_detections_per_img=100,  # Increased from default
                rpn_pre_nms_top_n_train=2000,  # Increased for better small object detection
                rpn_pre_nms_top_n_test=1000,
                rpn_post_nms_top_n_train=2000,
                rpn_post_nms_top_n_test=1000,
                rpn_nms_thresh=0.7,  # Adjusted NMS threshold
                box_nms_thresh=0.3,  # Lower NMS threshold for better detection
                box_score_thresh=0.05  # Lower initial threshold to catch more potential objects
            )
            
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
    
    def predict(self, image, confidence_threshold=None):
        """Run prediction on an image with class-specific thresholds"""
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
            
            # Store original size for scaling
            original_size = image.size
            
            # Apply transforms
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Run inference with higher IoU threshold for NMS
            with torch.no_grad():
                predictions = self.model(img_tensor)
                prediction = predictions[0]
                
                # Extract predictions
                pred_boxes = prediction["boxes"].cpu().numpy()
                pred_labels = prediction["labels"].cpu().numpy()
                pred_scores = prediction["scores"].cpu().numpy()
                
                # Apply class-specific confidence thresholds
                keep_indices = []
                for i, (label, score) in enumerate(zip(pred_labels, pred_scores)):
                    threshold = self.confidence_thresholds.get(label, 0.5)
                    if score >= threshold:
                        keep_indices.append(i)
                
                # Filter predictions
                pred_boxes = pred_boxes[keep_indices]
                pred_labels = pred_labels[keep_indices]
                pred_scores = pred_scores[keep_indices]
                
                # Scale boxes back to original image size
                if len(pred_boxes) > 0:
                    scale_x = original_size[0] / 800
                    scale_y = original_size[1] / 800
                    pred_boxes[:, [0, 2]] *= scale_x
                    pred_boxes[:, [1, 3]] *= scale_y
                
                logger.debug(f"Found {len(pred_boxes)} predictions")
                return pred_boxes, pred_labels, pred_scores
                
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
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

def create_custom_fasterrcnn_with_bn(num_classes=5, pretrained_backbone=True):
    """
    Create a custom FasterRCNN model with batch normalization.
    This is used by the model loading mechanism.
    
    Args:
        num_classes: Number of classes (including background)
        pretrained_backbone: Whether to use pretrained backbone
        
    Returns:
        The FasterRCNN model
    """
    import torchvision
    from torchvision.models.detection.faster_rcnn import FasterRCNN
    from torchvision.models.detection.rpn import AnchorGenerator
    
    # Use ResNet-50 with FPN as the backbone
    backbone = torchvision.models.resnet50(pretrained=pretrained_backbone)
    
    # FasterRCNN needs to know the number of output channels in the backbone
    backbone_out_channels = backbone.fc.in_features
    
    # Create the anchor generator
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )
    
    # Create RoI pooler
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2
    )
    
    # Create the FasterRCNN model
    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        min_size=800,
        max_size=1333,
        rpn_pre_nms_top_n_train=2000,
        rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_train=2000,
        rpn_post_nms_top_n_test=1000,
        rpn_nms_thresh=0.7,
        box_nms_thresh=0.3,
        box_detections_per_img=100,
        box_score_thresh=0.05
    )
    
    return model

def draw_predictions(image, boxes, labels, scores, score_thresh=0.5):
    """Draw predictions on image with improved visibility"""
    from PIL import ImageDraw, ImageFont
    
    # Get a new model adapter instance for color mapping
    adapter = get_model_adapter()
    color_map = adapter.get_color_map()
    class_map = adapter.get_class_map()
    
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", 16)  # Increased font size
    except:
        font = ImageFont.load_default()
    
    # Draw boxes with thicker lines and improved visibility
    for i, box in enumerate(boxes):
        if scores[i] < score_thresh:
            continue
            
        x1, y1, x2, y2 = map(int, box)
        cls_id = int(labels[i])
        color = color_map.get(cls_id, (255, 255, 255))
        
        # Draw box with thicker outline
        for thickness in range(3):  # Draw multiple rectangles for thickness
            draw.rectangle([x1-thickness, y1-thickness, x2+thickness, y2+thickness], 
                         outline=color)
        
        # Draw label with improved visibility
        class_name = class_map.get(cls_id, f"Class {cls_id}")
        text = f"{class_name}: {scores[i]:.2f}"
        text_size = draw.textbbox((0, 0), text, font=font)[2:4]
        
        # Draw label background
        draw.rectangle([x1, y1 - text_size[1] - 4, x1 + text_size[0], y1], 
                      fill=color)
        
        # Draw text with contrasting color
        text_color = (0, 0, 0) if cls_id in [2, 4] else (255, 255, 255)
        draw.text((x1, y1 - text_size[1] - 2), text, fill=text_color, font=font)
    
    return image 