import numpy as np
import tensorflow as tf
import cv2

def get_last_conv_layer_name(model):
    """
    Finds the last convolutional layer in a Keras model.
    If the model has nested layers (like MobileNetV2 base), it looks inside.
    """
    # Check if the model has a base_model (transfer learning setup)
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model):
            for inner_layer in reversed(layer.layers):
                if isinstance(inner_layer, tf.keras.layers.Conv2D):
                    return inner_layer.name, layer.name
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name, None
    raise ValueError("Could not find a convolutional layer.")

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, base_model_name=None, pred_index=None):
    """
    Generates the Grad-CAM heatmap for a given image and model.
    """
    # If the model uses a nested base model, we extract it
    if base_model_name:
        base_model = model.get_layer(base_model_name)
        # We need a model that maps from the input of the base model to its last conv layer
        last_conv_layer = base_model.get_layer(last_conv_layer_name)
        last_conv_layer_model = tf.keras.Model(base_model.inputs, last_conv_layer.output)
        
        # We also need a model that maps from the last conv layer to the final predictions
        # This includes the layers in the main model AFTER the base model
        classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
        x = classifier_input
        # Find index of base_model in main model
        start_idx = model.layers.index(base_model) + 1
        for layer in model.layers[start_idx:]:
            x = layer(x)
        classifier_model = tf.keras.Model(classifier_input, x)
    else:
        # Standard flattened model
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
        )

    with tf.GradientTape() as tape:
        if base_model_name:
            last_conv_layer_output = last_conv_layer_model(img_array)
            tape.watch(last_conv_layer_output)
            preds = classifier_model(last_conv_layer_output)
        else:
            last_conv_layer_output, preds = grad_model(img_array)
            
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Gradient of the output neuron w.r.t. the last conv layer feature map
    if base_model_name:
        grads = tape.gradient(class_channel, last_conv_layer_output)
    else:
        grads = tape.gradient(class_channel, last_conv_layer_output)

    # Average gradients spatially
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    # Multiply each channel by "how important it is" with regard to the class
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(img_array, heatmap, alpha=0.4):
    """
    Overlays the heatmap onto the original image.
    img_array: numpy array of original image (0-255 RGB)
    """
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # OpenCV uses BGR, convert back to RGB
    jet = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)

    # Resize heatmap to match original image size
    jet = cv2.resize(jet, (img_array.shape[1], img_array.shape[0]))

    # Superimpose the heatmap on original image
    superimposed_img = jet * alpha + img_array
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

    return superimposed_img
