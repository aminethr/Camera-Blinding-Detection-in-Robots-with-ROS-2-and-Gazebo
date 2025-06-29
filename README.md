## 🔦 Hybrid CNN-Based Camera Blinding Detection for Robots

A hybrid deep learning model designed to detect camera blinding caused by strong light sources, such as flashlights or laser interference. The system is ideal for robotic platforms including UAVs, UGVs, and surveillance agents in both simulation and real-world environments.
📸 Dataset Collection

The dataset was generated in a Gazebo simulation using a robot equipped with a light-sensitive camera. The robot was placed in various positions and angles to simulate real-world lighting scenarios, including potential blinding angles.

    The resulting images are grayscale and reflect how light impacts visibility.

Each image is labeled as:

    1: Blinded (light obstructs vision)

    0: Not Blinded

## 🏷️ Auto-Labeling Logic

A script was used to automatically generate labels based on:

    Mean brightness ≥ 130

    Saturated pixels ≥ 3%

    OR a large bright blob (area > 2000 px) detected in the image

This outputs a labels.csv file like:

filename,label,percentage
frame_0001.png,0,1.85
frame_0002.png,1,12.63

    ⚠️ You may need to adjust thresholds based on camera type or image resolution.

## 🧪 Preprocessing

Each image is:

    Converted to grayscale

    Resized to 128x128

    Normalized

    Analyzed to compute:

        Mean brightness

        Saturation percentage

These handcrafted features are combined with CNN output for classification.
## 🧠 Model Architecture

A hybrid neural network:

    CNN Branch:

        3 convolutional layers

        Extracts deep spatial features

    Handcrafted Features:

        Mean brightness

        Saturated pixel ratio

    Fusion:

        CNN output + handcrafted features → Fully connected layers → Binary output

Summary:

    Input: 128×128 grayscale image + 2 features

    Output: Sigmoid → prediction ∈ [0, 1]

## 🏋️ Training

Training is handled in model_training.py. The script:

    Loads the dataset and labels

    Trains the hybrid CNN model

    Saves the final model weights

python model_training.py

Example (inside script):

if __name__ == "__main__":
    train()
    predict("/path/to/test/image.png")

✅ After training, the model is saved as:

hybrid_blinding_model.pth

    🧠 These weights are then loaded inside the ROS 2 node for real-time use.

## 🤖 ROS 2 Node — blinding_detector

This ROS 2 node uses the trained weights (.pth file) to perform live inference on incoming camera images.

    Subscribes to:
    /drone2_camera1_sensor/image_raw (or any camera topic)

    Outputs:

        ⚠️ BLINDED if prediction ≥ 0.5

        ✅ NOT BLINDED otherwise

ros2 run <your_package_name> blinding_detector

    The node is written using rclpy and cv_bridge to handle image streams.

## 📌 Note:

If you change the image topic or model path, edit it in blinding_detector_node.py:

self.model.load_state_dict(torch.load("path/to/hybrid_blinding_model.pth", ...))


## ✅ Sample ROS Output
![Blinding Detection Demo](https://github.com/user-attachments/assets/fb43d1f8-0986-45c1-b01f-616fd401c062)


[INFO] [blinding_detector]: ✅ NOT BLINDED
[WARN] [blinding_detector]: ⚠️ BLINDED

## 🛠️ Dependencies

    Python 3.8+

    PyTorch

    OpenCV

    NumPy

    ROS 2 (Humble or newer)

    cv_bridge (ROS 2 image conversions)
    

## 💡 Use Cases & Notes

    Supports UAVs, ground robots, and stationary surveillance cameras

    Can be extended to alert systems, adaptive behaviors, or swarm logic

    Easily upgradable with additional image features or multi-class outputs

