import cv2
import numpy as np
import tensorflow as tf
import serial
import time
import os
import serial.tools.list_ports

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


MODEL_PATH = 'new_tflite_model.tflite'


def list_available_ports():
    ports = serial.tools.list_ports.comports()
    available_ports = [port.device for port in ports]
    return available_ports

available_ports = list_available_ports()
print("Available COM Ports:", available_ports)


if not available_ports:
    print("No COM ports found. Please connect your Arduino and try again.")
    exit()

print("Please select the COM port your Arduino is connected to:")
for idx, port in enumerate(available_ports):
    print(f"{idx + 1}: {port}")

selected = int(input("Enter the number corresponding to the COM port: ")) - 1

if selected not in range(len(available_ports)):
    print("Invalid selection. Exiting.")
    exit()

SERIAL_PORT = available_ports[selected]
BAUD_RATE = 9600     

COMMAND_COOLDOWN = 6  

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()


input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


print("Input details:", input_details)


input_shape = input_details[0]['shape']
height = input_shape[1]
width = input_shape[2]
channels = input_shape[3] if len(input_shape) > 3 else 1  
input_dtype = input_details[0]['dtype']

print(f"Model expects input shape: {input_shape}, dtype: {input_dtype}")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    print(f"Connected to Arduino on port {SERIAL_PORT} at {BAUD_RATE} baud.")
except serial.SerialException as e:
    print(f"Error: Could not open serial port {SERIAL_PORT}: {e}")
    cap.release()
    cv2.destroyAllWindows()
    exit()

last_command_time = 0 

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    resized_frame = cv2.resize(frame, (width, height))

    if channels == 1:
        gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        input_data = np.expand_dims(gray_frame, axis=(0, -1))
    else:
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(rgb_frame, axis=0)

    if input_dtype == np.float32:
        input_data = input_data.astype(np.float32) / 255.0
    else:
        input_data = input_data.astype(input_dtype)

    print(f"Input data shape: {input_data.shape}, dtype: {input_data.dtype}")

    try:
        interpreter.set_tensor(input_details[0]['index'], input_data)
    except ValueError as e:
        print("Error setting tensor:", e)
        break

    interpreter.invoke()

    
    output_data = interpreter.get_tensor(output_details[0]['index'])

    if output_details[0]['dtype'] == np.int8:
        output_data = output_data.astype(np.float32)


    confidence_scores = tf.nn.softmax(output_data[0]).numpy() 

   
    class_labels = ['Plastic', 'Non-plastic']

    plastic_confidence = confidence_scores[0]
    non_plastic_confidence = confidence_scores[1]

   
    for i, label in enumerate(class_labels):
        confidence = confidence_scores[i]
        cv2.putText(frame, f"{label}: {confidence:.2f}",
                    (10, 30 + i*40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Plastic Recognition', frame)

    current_time = time.time()

    if (plastic_confidence > non_plastic_confidence) and (current_time - last_command_time > COMMAND_COOLDOWN):
        try:
            ser.write(b'1') 
            print("Sent '1' to Arduino to activate servo.")
            last_command_time = current_time
        except serial.SerialException as e:
            print(f"Error sending data to Arduino: {e}")


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
ser.close()
print("Serial connection closed.")
