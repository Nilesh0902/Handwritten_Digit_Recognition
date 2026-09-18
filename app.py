import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from streamlit_drawable_canvas import st_canvas


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="🔢",
    layout="centered"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

.main {
    padding-top: 1rem;
}

.block-container {
    max-width: 900px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Main Header */
.main-title {
    text-align: center;
    font-size: 38px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 17px;
    opacity: 0.75;
    margin-bottom: 30px;
}

/* Info Cards */
.info-card {
    border: 1px solid rgba(128, 128, 128, 0.35);
    border-radius: 12px;
    padding: 18px;
    text-align: center;
    min-height: 100px;
}

.info-title {
    font-size: 14px;
    opacity: 0.7;
    margin-bottom: 7px;
}

.info-value {
    font-size: 22px;
    font-weight: 700;
}

/* Drawing Area */
.drawing-title {
    font-size: 20px;
    font-weight: 600;
    margin-bottom: 8px;
}

/* Prediction Box */
.prediction-box {
    border: 2px solid rgba(128, 128, 128, 0.45);
    border-radius: 15px;
    padding: 25px;
    text-align: center;
    margin-top: 10px;
    margin-bottom: 25px;
}

.prediction-label {
    font-size: 17px;
    font-weight: 600;
    margin-bottom: 8px;
}

.prediction-number {
    font-size: 70px;
    font-weight: 800;
    line-height: 1.1;
    margin: 5px 0 10px 0;
}

.confidence {
    font-size: 17px;
}

/* Digit Cards */
.digit-card {
    border: 1px solid rgba(128, 128, 128, 0.35);
    border-radius: 12px;
    padding: 15px;
    text-align: center;
    margin-bottom: 10px;
}

.digit-number {
    font-size: 38px;
    font-weight: 700;
}

.digit-confidence {
    font-size: 14px;
    opacity: 0.8;
}

/* Footer */
.footer {
    text-align: center;
    opacity: 0.6;
    font-size: 13px;
    margin-top: 40px;
    padding-top: 20px;
    border-top: 1px solid rgba(128, 128, 128, 0.25);
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🔢 Handwritten Digit Recognition</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">CNN-powered handwritten digit recognition using the MNIST dataset</div>',
    unsafe_allow_html=True
)


# ============================================================
# MODEL INFORMATION
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        '<div class="info-card"><div class="info-title">Dataset</div><div class="info-value">MNIST</div></div>',
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        '<div class="info-card"><div class="info-title">Model</div><div class="info-value">CNN</div></div>',
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        '<div class="info-card"><div class="info-title">Test Accuracy</div><div class="info-value">99.28%</div></div>',
        unsafe_allow_html=True
    )


st.write("")


# ============================================================
# LOAD MODEL ARCHITECTURE
# ============================================================

@st.cache_resource
def load_model():

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),

        tf.keras.layers.Conv2D(
            32,
            kernel_size=(3, 3),
            activation="relu"
        ),

        tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2)
        ),

        tf.keras.layers.Conv2D(
            64,
            kernel_size=(3, 3),
            activation="relu"
        ),

        tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2)
        ),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(
            128,
            activation="relu"
        ),

        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(
            10,
            activation="softmax"
        )
    ])

    model.load_weights("digit_cnn.weights.h5")

    return model


try:
    model = load_model()
except Exception as e:
    st.error("Unable to load the trained model.")
    st.code(str(e))
    st.stop()


# ============================================================
# INSTRUCTIONS
# ============================================================

st.markdown("### ✍️ Draw Your Digit")

st.info(
    "Draw up to 3 digits using the mouse. "
    "Keep separate digits clearly spaced from each other."
)

st.markdown(
    '<div class="drawing-title">Drawing Canvas</div>',
    unsafe_allow_html=True
)


# ============================================================
# DRAWING CANVAS
# ============================================================

canvas_result = st_canvas(
    fill_color="rgba(0, 0, 0, 0)",
    stroke_width=10,
    stroke_color="#FFFFFF",
    background_color="#000000",
    width=500,
    height=280,
    drawing_mode="freedraw",
    key="digit_canvas",
    update_streamlit=True,
    return_image_data=True
)


# ============================================================
# BUTTONS
# ============================================================

col1, col2 = st.columns(2)

with col1:
    predict_button = st.button(
        "🔍 Predict",
        use_container_width=True,
        type="primary"
    )

with col2:
    clear_button = st.button(
        "🗑️ Clear",
        use_container_width=True
    )


# ============================================================
# CLEAR BUTTON
# ============================================================

if clear_button:
    st.session_state.pop("prediction_result", None)
    st.rerun()


# ============================================================
# DIGIT DETECTION / SEGMENTATION
# ============================================================

def detect_digits(image_array):
    """
    Detect individual handwritten digits by finding
    continuous non-empty regions along the horizontal axis.
    """

    if image_array is None:
        return []

    # Convert RGBA/RGB image to grayscale
    if image_array.ndim == 3:

        if image_array.shape[2] >= 3:
            gray = image_array[:, :, :3].mean(axis=2)
        else:
            gray = image_array[:, :, 0]

    else:
        gray = image_array

    # Threshold
    binary = gray > 20

    # Find columns containing white pixels
    column_has_pixels = np.any(binary, axis=0)

    digit_regions = []

    start = None

    for i, has_pixel in enumerate(column_has_pixels):

        if has_pixel and start is None:
            start = i

        elif not has_pixel and start is not None:

            end = i

            # Ignore very small regions/noise
            if end - start >= 5:
                digit_regions.append((start, end))

            start = None

    # Handle region reaching the end
    if start is not None:

        end = len(column_has_pixels)

        if end - start >= 5:
            digit_regions.append((start, end))

    # Merge regions that are very close together
    merged_regions = []

    for region in digit_regions:

        if not merged_regions:
            merged_regions.append(list(region))

        else:

            previous = merged_regions[-1]

            gap = region[0] - previous[1]

            if gap <= 5:
                previous[1] = region[1]

            else:
                merged_regions.append(list(region))

    # Maximum 3 digits
    merged_regions = merged_regions[:3]

    return merged_regions


# ============================================================
# DIGIT PREPROCESSING
# ============================================================

def preprocess_digit(digit_image):
    """
    Convert a cropped digit into MNIST-like 28x28 format.
    """

    # Convert to grayscale
    if digit_image.ndim == 3:

        gray = digit_image[:, :, :3].mean(axis=2)

    else:
        gray = digit_image

    # Convert to binary
    binary = gray > 20

    # Find non-zero pixels
    rows = np.where(np.any(binary, axis=1))[0]
    cols = np.where(np.any(binary, axis=0))[0]

    if len(rows) == 0 or len(cols) == 0:
        return None

    # Crop to bounding box
    cropped = gray[
        rows.min():rows.max() + 1,
        cols.min():cols.max() + 1
    ]

    # Convert to PIL
    cropped_image = Image.fromarray(
        cropped.astype(np.uint8)
    )

    # Preserve aspect ratio
    width, height = cropped_image.size

    max_dimension = max(width, height)

    scale = 20 / max_dimension

    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    cropped_image = cropped_image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )

    # Create 28x28 canvas
    final_image = Image.new(
        "L",
        (28, 28),
        0
    )

    # Center digit
    x_offset = (28 - new_width) // 2
    y_offset = (28 - new_height) // 2

    final_image.paste(
        cropped_image,
        (x_offset, y_offset)
    )

    # Normalize
    image_array = np.array(final_image).astype("float32") / 255.0

    # Add channel dimension
    image_array = image_array[..., np.newaxis]

    # Add batch dimension
    image_array = image_array[np.newaxis, ...]

    return image_array


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    if canvas_result.image_data is None:

        st.warning("Please draw at least one digit first.")

    else:

        image_data = canvas_result.image_data

        # Detect digit regions
        digit_regions = detect_digits(image_data)

        if len(digit_regions) == 0:

            st.warning(
                "No digit was detected. Please draw a clearer digit."
            )

        else:

            predictions = []
            processed_images = []

            for start_x, end_x in digit_regions:

                # Crop digit region
                digit_crop = image_data[:, start_x:end_x, :]

                # Preprocess
                processed_digit = preprocess_digit(
                    digit_crop
                )

                if processed_digit is None:
                    continue

                # Predict
                probabilities = model.predict(
                    processed_digit,
                    verbose=0
                )[0]

                predicted_digit = int(
                    np.argmax(probabilities)
                )

                confidence = float(
                    np.max(probabilities)
                )

                predictions.append({
                    "digit": predicted_digit,
                    "confidence": confidence,
                    "probabilities": probabilities
                })

                processed_images.append(
                    processed_digit[0, :, :, 0]
                )

            # ====================================================
            # STORE RESULTS
            # ====================================================

            if len(predictions) > 0:

                predicted_number = "".join(
                    str(item["digit"])
                    for item in predictions
                )

                average_confidence = np.mean([
                    item["confidence"]
                    for item in predictions
                ])

                st.session_state["prediction_result"] = {
                    "predicted_number": predicted_number,
                    "average_confidence": average_confidence,
                    "predictions": predictions,
                    "processed_images": processed_images
                }


# ============================================================
# DISPLAY PREDICTION RESULT
# ============================================================

if "prediction_result" in st.session_state:

    result = st.session_state["prediction_result"]

    predicted_number = result["predicted_number"]
    average_confidence = result["average_confidence"]
    predictions = result["predictions"]
    processed_images = result["processed_images"]

    st.markdown("---")

    st.markdown(
        "### 🎯 Prediction Result"
    )

    # IMPORTANT:
    # HTML starts at column 1 to prevent Streamlit
    # from interpreting it as a code block.

    prediction_html = f"""<div class="prediction-box">
<div class="prediction-label">Recognized Number</div>
<div class="prediction-number">{predicted_number}</div>
<div class="confidence">Average Confidence: <strong>{average_confidence * 100:.2f}%</strong></div>
</div>"""

    st.markdown(
        prediction_html,
        unsafe_allow_html=True
    )


    # ========================================================
    # INDIVIDUAL DIGIT PREDICTIONS
    # ========================================================

    st.markdown(
        "### 🔍 Individual Digit Predictions"
    )

    digit_columns = st.columns(len(predictions))

    for index, prediction in enumerate(predictions):

        with digit_columns[index]:

            digit = prediction["digit"]
            confidence = prediction["confidence"]

            digit_html = f"""<div class="digit-card">
<div class="info-title">Digit {index + 1}</div>
<div class="digit-number">{digit}</div>
<div class="digit-confidence">{confidence * 100:.2f}% confidence</div>
</div>"""

            st.markdown(
                digit_html,
                unsafe_allow_html=True
            )


    # ========================================================
    # PROCESSED IMAGES
    # ========================================================

    with st.expander("🖼️ View Processed Digit Images"):

        image_columns = st.columns(len(processed_images))

        for index, processed_image in enumerate(processed_images):

            with image_columns[index]:

                st.image(
                    processed_image,
                    caption=f"Digit {index + 1} - 28×28",
                    width=150
                )


    # ========================================================
    # PROBABILITY DISTRIBUTION
    # ========================================================

    with st.expander("📊 View Prediction Probabilities"):

        for index, prediction in enumerate(predictions):

            st.markdown(
                f"**Digit {index + 1}: {prediction['digit']}**"
            )

            probabilities = prediction["probabilities"]

            for digit_value, probability in enumerate(probabilities):

                st.progress(
                    float(probability),
                    text=f"{digit_value}: {probability * 100:.2f}%"
                )


# ============================================================
# ABOUT MODEL
# ============================================================

st.markdown("---")

with st.expander("ℹ️ About This Model"):

    st.markdown(
        """
### CNN Architecture

This project uses a Convolutional Neural Network trained on the MNIST dataset.

**Architecture:**

- Input: 28 × 28 × 1
- Conv2D: 32 filters, 3 × 3, ReLU
- MaxPooling: 2 × 2
- Conv2D: 64 filters, 3 × 3, ReLU
- MaxPooling: 2 × 2
- Flatten
- Dense: 128 neurons, ReLU
- Dropout: 0.5
- Output: 10 neurons, Softmax

### Training

- Dataset: MNIST
- Training images: 60,000
- Test images: 10,000
- Epochs: 10
- Batch size: 128
- Optimizer: Adam
- Loss function: Sparse Categorical Crossentropy
- Test Accuracy: **99.28%**

### Application

The application allows users to draw **1 to 3 handwritten digits**.
The system segments the digits, preprocesses them into MNIST-like
28 × 28 images, and uses the trained CNN to predict each digit.
"""
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
<div class="footer">
Handwritten Digit Recognition using CNN<br>
MSc Data Science Mini Project
</div>
""",
    unsafe_allow_html=True
)
