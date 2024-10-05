import cv2
import numpy as np
import tensorflow as tf

# Path to your TFLite model
MODEL_PATH = 'model.tflite'

# Initialize the TFLite interpreter
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Debug: Print input details to confirm shape and dtype
print("Input details:", input_details)

# Get input size and type
input_shape = input_details[0]['shape']
height = input_shape[1]
width = input_shape[2]
channels = input_shape[3] if len(input_shape) > 3 else 1  # Number of channels expected
input_dtype = input_details[0]['dtype']

print(f"Model expects input shape: {input_shape}, dtype: {input_dtype}")

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

    # Normalize if required by your model (example: [0,1])
    if input_dtype == np.float32:
        input_data = input_data.astype(np.float32) / 255.0
    else:
        # If the model expects integer inputs (e.g., uint8), adjust accordingly
        input_data = input_data.astype(input_dtype)

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

    # Assuming the model outputs probabilities for classes
    # Modify this part based on your model's output structure
    predicted_class = np.argmax(output_data, axis=1)[0]
    confidence = output_data[0][predicted_class]

    # Define class labels (modify as per your model)
    class_labels = ['Plastic', 'Paper', 'Gemstone']
    label = class_labels[predicted_class]

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
