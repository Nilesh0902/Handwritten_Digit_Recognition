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
    layout="centered",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main page */
    .main {
        padding-top: 1rem;
    }

    /* Header */
    .main-title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
    }

    /* Information cards */
    .info-card {
        border: 1px solid rgba(128, 128, 128, 0.3);
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        min-height: 90px;
    }

    .info-title {
        font-size: 0.85rem;
        margin-bottom: 5px;
    }

    .info-value {
        font-size: 1.25rem;
        font-weight: 700;
    }

    /* Prediction */
    .prediction-box {
        border: 2px solid rgba(128, 128, 128, 0.4);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        margin-top: 15px;
    }

    .prediction-label {
        font-size: 1rem;
        margin-bottom: 5px;
    }

    .prediction-number {
        font-size: 3.5rem;
        font-weight: 800;
        margin: 0;
    }

    .confidence {
        font-size: 1rem;
        margin-top: 5px;
    }

    /* Section headings */
    .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }

    /* Footer */
    .footer {
        text-align: center;
        font-size: 0.85rem;
        margin-top: 2rem;
        padding-top: 1rem;
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
    '<div class="subtitle">'
    'CNN-Based Deep Learning Application'
    '</div>',
    unsafe_allow_html=True
)

st.write(
    "Draw one, two, or three handwritten digits and let the trained "
    "Convolutional Neural Network recognize them."
)


# ============================================================
# MODEL INFORMATION
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class="info-card">
            <div class="info-title">Dataset</div>
            <div class="info-value">MNIST</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div class="info-card">
            <div class="info-title">Model</div>
            <div class="info-value">CNN</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        """
        <div class="info-card">
            <div class="info-title">Test Accuracy</div>
            <div class="info-value">99.28%</div>
        </div>
        """,
        unsafe_allow_html=True
    )


st.divider()


# ============================================================
# LOAD TRAINED MODEL
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
    st.error("Unable to load the trained CNN model.")
    st.exception(e)
    st.stop()


# ============================================================
# DRAWING INSTRUCTIONS
# ============================================================

st.markdown(
    '<div class="section-title">✏️ Draw Your Digit(s)</div>',
    unsafe_allow_html=True
)

st.info(
    "Draw up to 3 digits. Leave a small gap between digits for better "
    "segmentation and prediction."
)

st.markdown(
    """
    **How to use:**
    1. Draw your digit(s) using the mouse.
    2. Leave a small space between different digits.
    3. Draw a maximum of 3 digits.
    4. Click **Predict Digit(s)**.
    """
)


# ============================================================
# CANVAS CONTROL
# ============================================================

if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None


# ============================================================
# DRAWING CANVAS
# ============================================================

canvas_result = st_canvas(
    fill_color="white",
    stroke_width=10,
    stroke_color="white",
    background_color="black",
    width=500,
    height=280,
    drawing_mode="freedraw",
    key=f"digit_canvas_{st.session_state.canvas_key}",
    return_image_data=True
)


# ============================================================
# BUTTONS
# ============================================================

button_col1, button_col2 = st.columns(2)

with button_col1:

    predict_clicked = st.button(
        "🔮 Predict Digit(s)",
        use_container_width=True,
        type="primary"
    )

with button_col2:

    clear_clicked = st.button(
        "🗑️ Clear Canvas",
        use_container_width=True
    )


# ============================================================
# CLEAR CANVAS
# ============================================================

if clear_clicked:

    st.session_state.canvas_key += 1
    st.session_state.prediction_result = None
    st.rerun()


# ============================================================
# DIGIT DETECTION
# ============================================================

def detect_digits(image):

    """
    Detect individual digits by identifying continuous
    horizontal regions containing white pixels.
    """

    # Convert RGBA canvas image to grayscale
    if image.shape[-1] == 4:
        gray = image[:, :, :3].mean(axis=2)
    else:
        gray = image.mean(axis=2)

    # Threshold
    binary = gray > 20

    # Find columns containing digit pixels
    column_has_pixels = binary.any(axis=0)

    regions = []

    start = None

    for i, has_pixels in enumerate(column_has_pixels):

        if has_pixels and start is None:
            start = i

        elif not has_pixels and start is not None:

            end = i

            if end - start >= 5:
                regions.append((start, end))

            start = None

    # Handle region reaching the end
    if start is not None:

        end = len(column_has_pixels)

        if end - start >= 5:
            regions.append((start, end))

    # Maximum of 3 digits
    if len(regions) > 3:
        regions = regions[:3]

    return gray, regions


# ============================================================
# DIGIT PREPROCESSING
# ============================================================

def preprocess_digit(gray, region):

    """
    Crop the digit, resize while maintaining aspect ratio,
    center it on a 28x28 canvas and normalize it.
    """

    x_start, x_end = region

    digit = gray[:, x_start:x_end]

    # Find actual bounding box
    binary = digit > 20

    rows = np.where(binary.any(axis=1))[0]
    cols = np.where(binary.any(axis=0))[0]

    if len(rows) == 0 or len(cols) == 0:
        return None

    y_min, y_max = rows.min(), rows.max() + 1
    x_min, x_max = cols.min(), cols.max() + 1

    digit = digit[y_min:y_max, x_min:x_max]

    # Convert to PIL image
    digit_image = Image.fromarray(
        digit.astype(np.uint8)
    )

    # Preserve aspect ratio
    width, height = digit_image.size

    max_size = 20

    scale = min(
        max_size / width,
        max_size / height
    )

    new_width = max(
        1,
        int(width * scale)
    )

    new_height = max(
        1,
        int(height * scale)
    )

    digit_image = digit_image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )

    # Create 28x28 black canvas
    final_image = Image.new(
        "L",
        (28, 28),
        0
    )

    # Center digit
    paste_x = (28 - new_width) // 2
    paste_y = (28 - new_height) // 2

    final_image.paste(
        digit_image,
        (paste_x, paste_y)
    )

    # Normalize
    processed = np.array(
        final_image
    ).astype("float32") / 255.0

    # Add channel dimension
    processed = processed[..., np.newaxis]

    return processed


# ============================================================
# PREDICTION
# ============================================================

if predict_clicked:

    if (
        canvas_result.image_data is None
        or not np.any(canvas_result.image_data[:, :, :3] > 20)
    ):

        st.warning(
            "Please draw at least one digit before clicking Predict."
        )

    else:

        gray, regions = detect_digits(
            canvas_result.image_data
        )

        if len(regions) == 0:

            st.warning(
                "No clear digit was detected. Please draw the digit again."
            )

        else:

            predictions = []
            processed_images = []

            for region in regions:

                processed = preprocess_digit(
                    gray,
                    region
                )

                if processed is not None:

                    prediction = model.predict(
                        processed[np.newaxis, ...],
                        verbose=0
                    )[0]

                    predicted_digit = int(
                        np.argmax(prediction)
                    )

                    confidence = float(
                        np.max(prediction)
                    )

                    predictions.append(
                        {
                            "digit": predicted_digit,
                            "confidence": confidence,
                            "probabilities": prediction
                        }
                    )

                    processed_images.append(
                        processed.squeeze()
                    )


            if len(predictions) == 0:

                st.warning(
                    "Unable to process the drawn digit."
                )

            else:

                st.session_state.prediction_result = (
                    predictions,
                    processed_images
                )


# ============================================================
# DISPLAY PREDICTION
# ============================================================

if st.session_state.prediction_result is not None:

    predictions, processed_images = (
        st.session_state.prediction_result
    )

    st.divider()

    st.markdown(
        '<div class="section-title">🎯 Prediction Result</div>',
        unsafe_allow_html=True
    )

    # Combine predicted digits
    predicted_number = "".join(
        str(item["digit"])
        for item in predictions
    )

    # Average confidence
    average_confidence = np.mean(
        [
            item["confidence"]
            for item in predictions
        ]
    )

    st.markdown(
        f"""
        <div class="prediction-box">

            <div class="prediction-label">
                Recognized Number
            </div>

            <div class="prediction-number">
                {predicted_number}
            </div>

            <div class="confidence">
                Average Confidence:
                <strong>
                    {average_confidence * 100:.2f}%
                </strong>
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    # Individual predictions
    st.markdown("### 🔍 Individual Digit Predictions")

    columns = st.columns(
        len(predictions)
    )

    for i, (column, item) in enumerate(
        zip(columns, predictions)
    ):

        with column:

            st.metric(
                label=f"Digit {i + 1}",
                value=str(item["digit"]),
                delta=f"{item['confidence'] * 100:.2f}% confidence"
            )

    # Processed images
    with st.expander("🖼️ View Processed Digit Images"):

        image_columns = st.columns(
            len(processed_images)
        )

        for i, (column, image) in enumerate(
            zip(image_columns, processed_images)
        ):

            with column:

                st.image(
                    image,
                    caption=f"Digit {i + 1} → {predictions[i]['digit']}",
                    width=120
                )

    # Probability information
    with st.expander("📊 View Prediction Probabilities"):

        for i, item in enumerate(predictions):

            st.write(
                f"**Digit {i + 1} — Predicted: "
                f"{item['digit']}**"
            )

            probabilities = item["probabilities"]

            for digit, probability in enumerate(
                probabilities
            ):

                st.progress(
                    float(probability),
                    text=f"{digit}: {probability * 100:.2f}%"
                )


# ============================================================
# ABOUT MODEL
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">🧠 About the Model</div>',
    unsafe_allow_html=True
)

st.write(
    "This application uses a Convolutional Neural Network (CNN) "
    "trained on the MNIST handwritten digit dataset."
)

about_col1, about_col2 = st.columns(2)

with about_col1:

    st.markdown(
        """
        **Model Details**

        - Dataset: MNIST
        - Input: 28 × 28 grayscale image
        - Classes: 10
        - Architecture: CNN
        - Optimizer: Adam
        """
    )

with about_col2:

    st.markdown(
        """
        **Performance**

        - Training Accuracy: 99.16%
        - Validation Accuracy: 99.17%
        - Test Accuracy: 99.28%
        - Test Loss: 0.02295
        - Maximum Digits: 3
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Handwritten Digit Recognition using CNN<br>
        Deep Learning and Neural Networks Mini Project
    </div>
    """,
    unsafe_allow_html=True
)
