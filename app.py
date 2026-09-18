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
# CNN MODEL
# ============================================================

@st.cache_resource
def create_model():

    model = tf.keras.Sequential([

        # Input
        tf.keras.layers.Input(
            shape=(28, 28, 1)
        ),

        # Convolution Layer 1
        tf.keras.layers.Conv2D(
            32,
            kernel_size=(3, 3),
            activation="relu"
        ),

        # Pooling Layer 1
        tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2)
        ),

        # Convolution Layer 2
        tf.keras.layers.Conv2D(
            64,
            kernel_size=(3, 3),
            activation="relu"
        ),

        # Pooling Layer 2
        tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2)
        ),

        # Flatten
        tf.keras.layers.Flatten(),

        # Dense Layer
        tf.keras.layers.Dense(
            128,
            activation="relu"
        ),

        # Dropout
        tf.keras.layers.Dropout(
            0.5
        ),

        # Output Layer
        tf.keras.layers.Dense(
            10,
            activation="softmax"
        )
    ])

    # Load trained weights
    model.load_weights(
        "digit_cnn.weights.h5"
    )

    return model


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model = create_model()

except Exception as e:

    st.error(
        "❌ Unable to load the trained CNN model."
    )

    st.exception(e)

    st.stop()


# ============================================================
# PREPROCESS SINGLE DIGIT
# ============================================================

def preprocess_digit(digit_image):

    """
    Converts one detected digit into the
    28 x 28 format expected by the CNN.
    """

    # Convert to grayscale
    digit_image = digit_image.convert("L")

    image_array = np.array(
        digit_image
    )

    # --------------------------------------------------------
    # Find digit pixels
    # --------------------------------------------------------

    mask = image_array > 20

    if not np.any(mask):
        return None

    # --------------------------------------------------------
    # Find bounding box
    # --------------------------------------------------------

    rows = np.where(
        mask.any(axis=1)
    )[0]

    cols = np.where(
        mask.any(axis=0)
    )[0]

    top = rows[0]
    bottom = rows[-1]

    left = cols[0]
    right = cols[-1]

    # Crop
    digit = image_array[
        top:bottom + 1,
        left:right + 1
    ]

    digit_image = Image.fromarray(
        digit.astype("uint8")
    )

    # --------------------------------------------------------
    # Resize while preserving aspect ratio
    # --------------------------------------------------------

    width, height = digit_image.size

    max_size = 20

    if width > height:

        new_width = max_size

        new_height = max(
            1,
            int(
                height *
                max_size /
                width
            )
        )

    else:

        new_height = max_size

        new_width = max(
            1,
            int(
                width *
                max_size /
                height
            )
        )

    digit_image = digit_image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )

    # --------------------------------------------------------
    # Create 28 x 28 canvas
    # --------------------------------------------------------

    final_image = Image.new(
        "L",
        (28, 28),
        0
    )

    # --------------------------------------------------------
    # Center digit
    # --------------------------------------------------------

    x = (
        28 - new_width
    ) // 2

    y = (
        28 - new_height
    ) // 2

    final_image.paste(
        digit_image,
        (x, y)
    )

    # --------------------------------------------------------
    # Convert to NumPy
    # --------------------------------------------------------

    final_array = np.array(
        final_image
    ).astype("float32")

    # Normalize
    final_array = (
        final_array / 255.0
    )

    # Add channel
    final_array = np.expand_dims(
        final_array,
        axis=-1
    )

    # Add batch
    final_array = np.expand_dims(
        final_array,
        axis=0
    )

    return final_array


# ============================================================
# DETECT INDIVIDUAL DIGITS
# ============================================================

def detect_digits(image):

    """
    Detect separate handwritten digits from
    a drawing containing up to 3 digits.

    Digits should have some horizontal space
    between them.
    """

    # Convert to grayscale
    image = image.convert("L")

    image_array = np.array(
        image
    )

    # --------------------------------------------------------
    # Threshold
    # --------------------------------------------------------

    binary = image_array > 20

    # --------------------------------------------------------
    # Find columns containing digit pixels
    # --------------------------------------------------------

    column_has_pixels = binary.any(
        axis=0
    )

    # --------------------------------------------------------
    # Find continuous horizontal regions
    # --------------------------------------------------------

    regions = []

    in_region = False

    start = 0

    for i, has_pixel in enumerate(
        column_has_pixels
    ):

        if has_pixel and not in_region:

            start = i

            in_region = True

        elif not has_pixel and in_region:

            end = i - 1

            regions.append(
                (start, end)
            )

            in_region = False

    # Handle region reaching the edge
    if in_region:

        regions.append(
            (
                start,
                len(column_has_pixels) - 1
            )
        )

    # --------------------------------------------------------
    # Filter very small regions
    # --------------------------------------------------------

    filtered_regions = []

    for start, end in regions:

        width = end - start + 1

        if width >= 5:

            filtered_regions.append(
                (start, end)
            )

    # --------------------------------------------------------
    # Limit to maximum 3 digits
    # --------------------------------------------------------

    if len(filtered_regions) > 3:

        return filtered_regions[:3]

    return filtered_regions


# ============================================================
# APPLICATION TITLE
# ============================================================

st.title(
    "🔢 Handwritten Digit Recognition"
)

st.write(
    "Draw up to 3 handwritten digits and "
    "let the CNN recognize them individually."
)

st.divider()


# ============================================================
# MODEL INFORMATION
# ============================================================

st.subheader(
    "📊 Model Performance"
)

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Test Accuracy",
        "99.28%"
    )

with col2:

    st.metric(
        "Test Images",
        "10,000"
    )

with col3:

    st.metric(
        "Parameters",
        "225,034"
    )


st.divider()


# ============================================================
# DRAWING AREA
# ============================================================

st.subheader(
    "✏️ Draw 1–3 Digits"
)

st.info(
    "Draw digits from left to right and leave "
    "a small gap between each digit."
)


canvas_result = st_canvas(

    background_color="black",

    stroke_color="white",

    stroke_width=10,

    width=500,

    height=280,

    drawing_mode="freedraw",

    return_image_data=True,

    key="digit_canvas"

)


st.write("")


# ============================================================
# PREDICTION BUTTON
# ============================================================

predict_button = st.button(
    "🔮 Predict Number",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    # --------------------------------------------------------
    # Check drawing
    # --------------------------------------------------------

    if canvas_result.image_data is None:

        st.warning(
            "⚠️ Please draw at least one digit."
        )

        st.stop()


    # --------------------------------------------------------
    # Convert canvas to PIL
    # --------------------------------------------------------

    original_image = Image.fromarray(
        canvas_result.image_data.astype(
            "uint8"
        )
    )


    # --------------------------------------------------------
    # Detect digits
    # --------------------------------------------------------

    regions = detect_digits(
        original_image
    )


    # --------------------------------------------------------
    # Check number of digits
    # --------------------------------------------------------

    if len(regions) == 0:

        st.warning(
            "⚠️ No digit detected. "
            "Please draw a digit."
        )

        st.stop()


    if len(regions) > 3:

        st.warning(
            "⚠️ Please draw a maximum of 3 digits."
        )

        st.stop()


    # ========================================================
    # PROCESS EACH DIGIT
    # ========================================================

    predictions_list = []

    confidence_list = []

    processed_images = []


    for start, end in regions:

        # ----------------------------------------------------
        # Crop vertical region
        # ----------------------------------------------------

        cropped = original_image.crop(
            (
                start,
                0,
                end + 1,
                original_image.height
            )
        )


        # ----------------------------------------------------
        # Preprocess
        # ----------------------------------------------------

        processed = preprocess_digit(
            cropped
        )


        if processed is None:

            continue


        # Save processed image
        processed_images.append(
            processed[0].squeeze()
        )


        # ----------------------------------------------------
        # CNN prediction
        # ----------------------------------------------------

        prediction = model.predict(
            processed,
            verbose=0
        )


        probabilities = prediction[0]


        # ----------------------------------------------------
        # Predicted digit
        # ----------------------------------------------------

        predicted_digit = int(
            np.argmax(probabilities)
        )


        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        confidence = float(
            np.max(probabilities)
        )


        predictions_list.append(
            predicted_digit
        )

        confidence_list.append(
            confidence
        )


    # ========================================================
    # FINAL RESULT
    # ========================================================

    if len(predictions_list) == 0:

        st.error(
            "❌ Unable to process the drawn digits."
        )

        st.stop()


    # Combine digits

    predicted_number = "".join(
        str(digit)
        for digit in predictions_list
    )


    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    st.divider()

    st.subheader(
        "🎯 Prediction Result"
    )


    st.success(
        f"Predicted Number: {predicted_number}"
    )


    # ========================================================
    # INDIVIDUAL DIGIT RESULTS
    # ========================================================

    st.subheader(
        "🔍 Individual Digit Predictions"
    )


    columns = st.columns(
        len(predictions_list)
    )


    for i, column in enumerate(
        columns
    ):

        with column:

            st.metric(
                f"Digit {i + 1}",
                predictions_list[i]
            )

            st.write(
                f"Confidence: "
                f"{confidence_list[i] * 100:.2f}%"
            )


    # ========================================================
    # PROCESSED DIGITS
    # ========================================================

    st.divider()

    st.subheader(
        "🧠 Images Given to the CNN"
    )


    processed_columns = st.columns(
        len(processed_images)
    )


    for i, column in enumerate(
        processed_columns
    ):

        with column:

            st.image(
                processed_images[i],
                width=120
            )

            st.caption(
                f"Digit {i + 1}"
            )


    # ========================================================
    # PROBABILITY BAR
    # ========================================================

    st.divider()

    st.subheader(
        "📈 Digit Confidence"
    )


    for i, confidence in enumerate(
        confidence_list
    ):

        st.write(
            f"Digit {i + 1} "
            f"({predictions_list[i]}): "
            f"{confidence * 100:.2f}%"
        )

        st.progress(
            confidence
        )


# ============================================================
# PROJECT INFORMATION
# ============================================================

st.divider()

with st.expander(
    "ℹ️ About the Project"
):

    st.write(
        """
        This project implements handwritten digit
        recognition using a Convolutional Neural Network.

        The CNN was trained using the MNIST dataset,
        containing 60,000 training images and 10,000
        testing images.

        Each MNIST image is a 28 × 28 grayscale image.

        The CNN architecture consists of:

        • Conv2D layer with 32 filters
        • MaxPooling layer
        • Conv2D layer with 64 filters
        • MaxPooling layer
        • Flatten layer
        • Dense layer with 128 neurons
        • Dropout regularization
        • Softmax output layer with 10 classes

        The trained model achieved a test accuracy of 99.28%.

        The application extends single-digit recognition
        to support recognition of up to three separately
        drawn digits.
        """
    )