import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# Define the path to the saved model
model_path = r'C:\Users\parth sawant\Desktop\digit recognition\best_model_digit.pt'

# Load the YOLO model
model = YOLO(model_path)

# Define class labels (0-9 for digits)
CLASS_LABELS = [str(i) for i in range(10)]  # Digits 0-9

def predict_image(image_path):
    # Predict on the uploaded image
    results = model.predict(source=image_path)
    output = []
    for result in results:
        boxes = result.boxes  # Bounding box outputs
        output.append({
            "boxes": boxes.xyxy.tolist(),  # Get bounding boxes in xyxy format
            "labels": boxes.cls.tolist(),  # Class labels (index of detected classes)
            "confidences": boxes.conf.tolist(),  # Confidence scores
        })
    return output

def annotate_image(image, boxes, labels):
    """Annotate the image with bounding boxes and class labels."""
    draw = ImageDraw.Draw(image)

    # Use a larger font for better visibility
    font_size = max(15, image.size[0] // 30)  # Adjust font size based on image size
    font = ImageFont.truetype("arial.ttf", font_size)

    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box
        label_text = CLASS_LABELS[int(label)]  # Convert label index to digit
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1 - font_size), label_text, fill="red", font=font)  # Display the label above the bounding box

    return image

def main():
    st.title("Digit Recognition System")
    st.write("Upload an image containing digits, and the app will detect and highlight the digits.")

    # File uploader for the image
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Open the uploaded image
        image = Image.open(uploaded_file).convert("RGB")

        # Button to trigger prediction
        if st.button("Detect Digits"):
            # Save the uploaded image to a temporary file
            temp_file_path = r'C:\Users\parth sawant\Desktop\temp_digit_image.jpg'
            image.save(temp_file_path)

            # Perform prediction
            results = predict_image(temp_file_path)

            # Process the results and annotate the image
            for result in results:
                annotated_image = annotate_image(image.copy(), result["boxes"], result["labels"])

            # Display the annotated image above the button
            st.image(annotated_image, caption="Annotated Image with Detected Digits", use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown("Developed with ❤️ using Streamlit and YOLO.")

if __name__ == '__main__':
    main()
