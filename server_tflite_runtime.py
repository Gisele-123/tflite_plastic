import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter  # Updated import

# Path to your TFLite model
MODEL_PATH = 'model.tflite'

# Initialize the TFLite interpreter
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Debug: Print input and output details to confirm shape and dtype
print("Input details:", input_details)
print("Output details:", output_details)

# Get input size and type
input_shape = input_details[0]['shape']
height = input_shape[1]
width = input_shape[2]
channels = input_shape[3] if len(input_shape) > 3 else 1  # Number of channels expected
input_dtype = input_details[0]['dtype']

print(f"Model expects input shape: {input_shape}, dtype: {input_dtype}")

# Extract quantization parameters for input
input_scale, input_zero_point = input_details[0]['quantization']
print(f"Input quantization scale: {input_scale}, zero point: {input_zero_point}")

# Extract quantization parameters for output
output_scale, output_zero_point = output_details[0]['quantization']
print(f"Output quantization scale: {output_scale}, zero point: {output_zero_point}")

# Initialize video capture (0 for default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Preprocess the frame
    # Resize to model's expected size
    resized_frame = cv2.resize(frame, (width, height))

    if channels == 1:
        # Convert BGR to Grayscale
        gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        # Expand dimensions to [1, height, width, 1]
        input_data = np.expand_dims(gray_frame, axis=(0, -1))
    else:
        # Convert BGR to RGB if model expects 3 channels
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        # Expand dimensions to [1, height, width, 3]
        input_data = np.expand_dims(rgb_frame, axis=0)

    # Handle quantization
    if input_dtype == np.int8:
        # Normalize the image to [0, 1]
        normalized_frame = input_data.astype(np.float32) / 255.0
        # Apply quantization formula: quantized = real / scale + zero_point
        quantized_frame = normalized_frame / input_scale + input_zero_point
        # Round and cast to int8
        quantized_frame = np.round(quantized_frame).astype(np.int8)
        input_data = quantized_frame
    else:
        # If the model expects float inputs, normalize accordingly
        input_data = input_data.astype(input_dtype) / 255.0

    # Debug: Print input data shape and dtype
    print(f"Input data shape: {input_data.shape}, dtype: {input_data.dtype}")

    try:
        # Set the tensor to the input data
        interpreter.set_tensor(input_details[0]['index'], input_data)
    except ValueError as e:
        print("Error setting tensor:", e)
        break

    # Run inference
    interpreter.invoke()

    # Get the output
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Dequantize the output if necessary
    if output_scale != 0:
        real_output = output_scale * (output_data.astype(np.float32) - output_zero_point)
    else:
        real_output = output_data

    # Debug: Print output data
    print(f"Output data (quantized): {output_data}")
    print(f"Output data (real): {real_output}")

    # Determine predicted class
    if real_output.ndim == 2:
        # Multi-class classification
        predicted_class = np.argmax(real_output, axis=1)[0]
        confidence = real_output[0][predicted_class]
    elif real_output.ndim == 1:
        # Binary classification (if model outputs single value)
        predicted_class = int(real_output[0] > 0)  # Example threshold
        confidence = real_output[0]
    else:
        # Handle other output shapes if necessary
        predicted_class = 0
        confidence = real_output[0]

    # Define class labels (modify as per your model)
    class_labels = ['Plastic', 'Paper', 'Gemstone']  # Ensure this matches model's output classes

    # Verify that predicted_class is within the range of class_labels
    if predicted_class < len(class_labels):
        label = class_labels[predicted_class]
    else:
        label = 'Unknown'

    # Display the label and confidence on the frame
    cv2.putText(frame, f"{label}: {confidence*100:.2f}%", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow('Plastic Recognition', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
