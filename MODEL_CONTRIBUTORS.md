# Guide for Model Contributors

This document provides guidelines for contributing new detection models to the VisionAid.

## Model Requirements

The detection system is designed to work with PyTorch-based object detection models. Currently, the system supports:

1. Faster R-CNN with ResNet50 FPN backbone
2. Faster R-CNN with ResNet50 FPN V2 backbone
3. Mask R-CNN (if mask predictions are included)

When contributing a new model, please ensure:

1. Your model uses the same class labels:
   - Class 1:  People
   - Class 2:  Encampments
   - Class 3:  Carts
   - Class 4:  Bikes

2. Your model produces predictions in the standard PyTorch detection format:
   ```python
   {
     "boxes": tensor([[x1, y1, x2, y2], ...]),  # Coordinates in (x1, y1, x2, y2) format
     "labels": tensor([class_id, ...]),         # Integer class IDs
     "scores": tensor([confidence, ...])        # Confidence scores between 0 and 1
   }
   ```

3. Your model weights are saved in a `.pth` file using `torch.save(model.state_dict(), filename)`.

## Adding Your Model

### Step 1: Export Your Model

Export your trained model using PyTorch's standard saving mechanism:

```python
torch.save(model.state_dict(), "your_model_name.pth")
```

### Step 2: Place Model in the Models Directory

Copy your model file to the `models/` directory in the project root.

### Step 3: Add Model Configuration

Add your model to the `models_config.json` file with the following information:

```json
{
  "models": {
    "your_model_name.pth": {
      "name": "Your Model Display Name",
      "description": "Brief description of your model",
      "type": "fasterrcnn_resnet50_fpn_v2",  # Model architecture type
      "num_classes": 5,  # Including background class
      "class_map": {
        "1": "Homeless_People",
        "2": "Homeless_Encampments",
        "3": "Homeless_Cart",
        "4": "Homeless_Bike"
      },
      "input_size": [640, 640],  # Expected input dimensions
      "preprocessing": "normalize",  # "default" or "normalize"
      "normalize_mean": [0.485, 0.456, 0.406],  # Only needed if preprocessing="normalize"
      "normalize_std": [0.229, 0.224, 0.225],   # Only needed if preprocessing="normalize"
      "confidence_thresholds": {
        "1": 0.5,  # Default confidence threshold for each class
        "2": 0.5,
        "3": 0.5,
        "4": 0.5
      }
    }
  }
}
```

### Step 4: Test Your Model

Run the application with your new model to verify it works correctly:

```bash
streamlit run app.py
```

Select your model from the dropdown in the sidebar and run a detection to confirm everything works.

## Supporting New Model Architectures

If your model uses an architecture not currently supported by the system, you'll need to:

1. Update the `model_adapter.py` file to add support for your architecture:

   ```python
   def _create_model_instance(self, model_type, num_classes):
       # Add your model type here
       if model_type == "your_model_type":
           return your_model_creation_function(weights=None, num_classes=num_classes)
       # ... existing model types ...
   ```

2. Ensure any special preprocessing needed for your model is handled in the `_setup_transforms` method.

3. If your model produces outputs in a different format, you may need to add a post-processing step in the `predict` method of the `ModelAdapter` class.

## Model Performance Considerations

1. **Inference Speed**: The system runs on users' machines which may not have GPUs. Consider optimizing your model for CPU inference if possible.

2. **Model Size**: Keep model size reasonable (preferably under 200MB) to ensure quick downloads and loading.

3. **Memory Usage**: Be mindful of peak memory usage during inference.

4. **Accuracy vs. Speed Tradeoffs**: Document any tradeoffs you've made between accuracy and inference speed.

## Documentation

When contributing a new model, please provide:

1. A brief description of the model architecture and any special features
2. Training details (dataset size, augmentations used, epochs, etc.)
3. Performance metrics (mAP, per-class accuracy, etc.)
4. Any known limitations or specific scenarios where the model performs particularly well or poorly

This information helps users select the appropriate model for their specific needs.

## Troubleshooting

If you encounter issues with your model:

1. **Model Loading Errors**: Ensure your model architecture matches what's specified in the config file.
2. **Memory Issues**: Try reducing batch size or model complexity.
3. **Cuda Errors**: Make sure your model can run on CPU if needed.
4. **Different Results**: Check for preprocessing differences between training and inference.

For assistance, please reach out to the project maintainers. 