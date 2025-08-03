import streamlit as st
import cv2
import numpy as np
import pytesseract
import io
import base64
from PIL import Image, ImageEnhance, ImageFilter
from scipy import ndimage
from skimage import morphology, exposure, restoration, filters
import streamlit.components.v1 as components
from streamlit_drawable_canvas import st_canvas

# Configure page
st.set_page_config(
    page_title="Enhanced Image OCR Extractor",
    page_icon="🔧",
    layout="wide"
)

def advanced_image_enhancement(img, method='auto_adaptive'):
    """Apply advanced enhancement methods to generate improved image"""
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    if method == 'number_optimized':
        # Specialized enhancement for number recognition
        # 1. Strong denoising
        denoised = cv2.fastNlMeansDenoising(gray, h=12)

        # 2. High contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(6,6))
        enhanced = clahe.apply(denoised)

        # 3. Morphological operations to clean digits
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        cleaned = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)

        # 4. Edge-preserving smoothing
        smoothed = cv2.bilateralFilter(cleaned, 9, 80, 80)

        # 5. Adaptive thresholding for better digit separation
        binary = cv2.adaptiveThreshold(smoothed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # 6. Slight dilation to thicken thin digits
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        result = cv2.dilate(binary, kernel_dilate, iterations=1)

    elif method == 'measurement_enhanced':
        # For measurements like "12.51m", "3.4kg", etc.
        # 1. Denoise
        denoised = cv2.fastNlMeansDenoising(gray, h=10)

        # 2. Enhance contrast specifically for small text
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4,4))
        enhanced = clahe.apply(denoised)

        # 3. Sharpen to make decimal points clearer
        kernel_sharp = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
        sharpened = cv2.filter2D(enhanced, -1, kernel_sharp)

        # 4. Morphological opening to separate connected characters
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        opened = cv2.morphologyEx(sharpened, cv2.MORPH_OPEN, kernel)

        result = opened

    elif method == 'digit_sharpening':
        # For digital/printed numbers with maximum sharpness
        # 1. Light denoising
        denoised = cv2.fastNlMeansDenoising(gray, h=8)

        # 2. Unsharp masking for maximum sharpness
        gaussian = cv2.GaussianBlur(denoised, (0, 0), 1.5)
        unsharp = cv2.addWeighted(denoised, 2.0, gaussian, -1.0, 0)

        # 3. High contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(unsharp)

        # 4. Edge enhancement
        laplacian = cv2.Laplacian(enhanced, cv2.CV_64F, ksize=3)
        laplacian = np.uint8(np.absolute(laplacian))
        result = cv2.addWeighted(enhanced, 0.9, laplacian, 0.1, 0)

    elif method == 'auto_adaptive':
        # Comprehensive adaptive enhancement
        # 1. Noise reduction
        denoised = cv2.fastNlMeansDenoising(gray, h=10)

        # 2. Adaptive histogram equalization
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)

        # 3. Edge-preserving filtering
        smoothed = cv2.bilateralFilter(enhanced, 9, 75, 75)

        # 4. Unsharp masking
        gaussian = cv2.GaussianBlur(smoothed, (0, 0), 2.0)
        unsharp = cv2.addWeighted(smoothed, 1.5, gaussian, -0.5, 0)

        result = unsharp

    elif method == 'handwriting_optimized':
        # Specifically for handwriting
        # 1. Denoise
        denoised = cv2.fastNlMeansDenoising(gray, h=8)

        # 2. Enhance contrast for handwriting
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(6,6))
        enhanced = clahe.apply(denoised)

        # 3. Morphological closing to connect broken strokes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        closed = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)

        # 4. Slight sharpening
        kernel_sharp = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(closed, -1, kernel_sharp)

        result = sharpened

    elif method == 'high_contrast':
        # Maximum contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)

        # Gamma correction
        gamma = 0.8
        lookup_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        result = cv2.LUT(enhanced, lookup_table)

    elif method == 'noise_reduction':
        # Advanced noise reduction
        # Multiple denoising passes
        denoised1 = cv2.fastNlMeansDenoising(gray, h=10)
        denoised2 = cv2.bilateralFilter(denoised1, 9, 80, 80)

        # Median filtering for salt-and-pepper noise
        result = cv2.medianBlur(denoised2, 3)

    elif method == 'edge_sharpening':
        # Edge enhancement and sharpening
        # Laplacian edge detection
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = np.uint8(np.absolute(laplacian))

        # Add edges back to original
        result = cv2.addWeighted(gray, 0.8, laplacian, 0.2, 0)

    elif method == 'brightness_contrast':
        # Brightness and contrast adjustment
        alpha = 1.3  # Contrast control
        beta = 20    # Brightness control
        result = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

    elif method == 'histogram_equalization':
        # Histogram equalization
        result = cv2.equalizeHist(gray)

    elif method == 'unsharp_masking':
        # Unsharp masking for sharpening
        gaussian = cv2.GaussianBlur(gray, (0, 0), 2.0)
        result = cv2.addWeighted(gray, 1.8, gaussian, -0.8, 0)

    elif method == 'morphological':
        # Morphological operations
        # Opening followed by closing
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        result = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    elif method == 'wiener_deconvolution':
        # Wiener deconvolution for blur removal
        try:
            from skimage import restoration
            # Create a motion blur kernel
            psf = np.ones((5, 5)) / 25
            result_float = restoration.wiener(gray, psf, balance=0.1)
            result = (result_float * 255).astype(np.uint8)
        except:
            # Fallback to simple sharpening if scikit-image restoration is not available
            kernel_sharp = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            result = cv2.filter2D(gray, -1, kernel_sharp)

    else:
        result = gray

    # Convert back to RGB
    if len(result.shape) == 2:
        result_rgb = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
    else:
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    return Image.fromarray(result_rgb)

def get_ocr_config(mode, language):
    """Get OCR configuration based on mode with specialized number recognition"""

    # Comprehensive character sets for different number types
    basic_numbers = '0123456789'
    decimal_numbers = '0123456789.,'
    measurement_chars = '0123456789.,-+()[]{}mkcglbftinMKCGLBFTIN°%'
    scientific_chars = '0123456789.,-+eE()[]{}x×*'
    currency_chars = '0123456789.,$€£¥₹₽¢₩₪₦₨₱₡₲₴₵₸₹'
    coordinate_chars = '0123456789.,-+°′″NSEW'
    handwriting_chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,+-=()[]{}/"' + "'"

    configs = {
        'numbers_precise': f'--oem 3 --psm 8 -l {language} -c tessedit_char_whitelist={decimal_numbers} -c classify_bln_numeric_mode=1',
        'measurements': f'--oem 3 --psm 8 -l {language} -c tessedit_char_whitelist={measurement_chars} -c classify_bln_numeric_mode=1',
        'scientific_notation': f'--oem 3 --psm 8 -l {language} -c tessedit_char_whitelist={scientific_chars} -c classify_bln_numeric_mode=1',
        'currency': f'--oem 3 --psm 8 -l {language} -c tessedit_char_whitelist={currency_chars} -c classify_bln_numeric_mode=1',
        'coordinates': f'--oem 3 --psm 8 -l {language} -c tessedit_char_whitelist={coordinate_chars} -c classify_bln_numeric_mode=1',
        'handwriting': f'--oem 3 --psm 8 -l {language} -c tessedit_char_whitelist={handwriting_chars}',
        'print': f'--oem 3 --psm 6 -l {language}',
        'mixed': f'--oem 3 --psm 3 -l {language}',
        'numbers': f'--oem 3 --psm 8 -l {language} -c tessedit_char_whitelist={measurement_chars}',
        'single_word': f'--oem 3 --psm 8 -l {language}'
    }
    return configs.get(mode, configs['numbers_precise'])

def extract_text_from_image(img, ocr_mode='handwriting', language='eng'):
    """Perform OCR on image"""
    try:
        config = get_ocr_config(ocr_mode, language)
        text = pytesseract.image_to_string(img, config=config)
        return text.strip()
    except Exception as e:
        st.error(f"OCR Error: {str(e)}")
        return ""

def multi_attempt_ocr(img, language='eng'):
    """Try multiple OCR approaches with focus on numbers"""
    try:
        results = []

        # Try different OCR modes with priority on number recognition
        number_modes = ['numbers_precise', 'measurements', 'scientific_notation', 'currency', 'coordinates']
        other_modes = ['handwriting', 'single_word', 'print', 'mixed']

        all_modes = number_modes + other_modes

        # First pass: Try all modes on original enhanced image
        for ocr_mode in all_modes:
            try:
                config = get_ocr_config(ocr_mode, language)
                text = pytesseract.image_to_string(img, config=config).strip()
                if text and text not in results:
                    results.append(f"[{ocr_mode}] {text}")
            except:
                continue

        # Second pass: Try with additional processing optimized for numbers
        try:
            # Convert to grayscale and apply strong threshold for crisp numbers
            gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)

            # Multiple threshold techniques
            threshold_methods = [
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
            ]

            for thresh_method in threshold_methods:
                try:
                    _, binary = cv2.threshold(gray, 0, 255, thresh_method)
                    binary_img = Image.fromarray(binary)

                    # Try number-specific modes on threshold images
                    for ocr_mode in number_modes:
                        try:
                            config = get_ocr_config(ocr_mode, language)
                            text = pytesseract.image_to_string(binary_img, config=config).strip()
                            if text and f"[{ocr_mode}_thresh] {text}" not in results:
                                results.append(f"[{ocr_mode}_thresh] {text}")
                        except:
                            continue
                except:
                    continue
        except:
            pass

        # Third pass: Try with morphological operations for better digit separation
        try:
            gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)

            # Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

            # Opening to separate connected digits
            opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
            opened_img = Image.fromarray(opened)

            # Closing to connect broken digits
            closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
            closed_img = Image.fromarray(closed)

            for processed_img, suffix in [(opened_img, '_opened'), (closed_img, '_closed')]:
                for ocr_mode in number_modes[:3]:  # Try top 3 number modes
                    try:
                        config = get_ocr_config(ocr_mode, language)
                        text = pytesseract.image_to_string(processed_img, config=config).strip()
                        if text and f"[{ocr_mode}{suffix}] {text}" not in results:
                            results.append(f"[{ocr_mode}{suffix}] {text}")
                    except:
                        continue
        except:
            pass

        # Clean up results and remove method labels for final display
        clean_results = []
        for result in results:
            if '] ' in result:
                clean_text = result.split('] ', 1)[1]
                if clean_text and clean_text not in clean_results:
                    clean_results.append(clean_text)

        return results, clean_results

    except Exception as e:
        st.error(f"Multi-attempt error: {str(e)}")
        return [], []

def crop_image_from_canvas(image, canvas_result):
    """Extract cropped image from canvas selection"""
    if canvas_result.json_data is None or len(canvas_result.json_data["objects"]) == 0:
        return None
    
    # Get the rectangle coordinates
    rect = canvas_result.json_data["objects"][0]
    left = int(rect["left"])
    top = int(rect["top"])
    width = int(rect["width"])
    height = int(rect["height"])
    
    # Crop the image
    img_array = np.array(image)
    cropped_array = img_array[top:top+height, left:left+width]
    
    if cropped_array.size > 0:
        return Image.fromarray(cropped_array)
    return None

# Initialize session state
if 'original_image' not in st.session_state:
    st.session_state.original_image = None
if 'enhanced_image' not in st.session_state:
    st.session_state.enhanced_image = None
if 'last_result' not in st.session_state:
    st.session_state.last_result = ""
if 'cropped_image' not in st.session_state:
    st.session_state.cropped_image = None

# Main app
st.title("🔧 Enhanced Image OCR Extractor")
st.markdown("*Optimized for Number Recognition with Advanced Image Enhancement*")

# Sidebar for controls
with st.sidebar:
    st.header("🎨 Image Enhancement")
    
    uploaded_file = st.file_uploader(
        "Upload Image", 
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        help="Upload an image for OCR processing"
    )
    
    enhancement_methods = {
        "number_optimized": "🔢 Number Recognition Optimized",
        "measurement_enhanced": "📏 Measurement Text Enhanced", 
        "digit_sharpening": "🎯 Digital/Printed Numbers",
        "auto_adaptive": "🤖 Auto Adaptive Enhancement",
        "handwriting_optimized": "✍️ Handwriting Optimized",
        "high_contrast": "⚡ High Contrast Boost",
        "noise_reduction": "🧹 Advanced Noise Reduction",
        "edge_sharpening": "📐 Edge Sharpening",
        "brightness_contrast": "💡 Brightness & Contrast",
        "histogram_equalization": "📊 Histogram Equalization",
        "unsharp_masking": "🔍 Unsharp Masking",
        "morphological": "🔄 Morphological Enhancement",
        "wiener_deconvolution": "🌟 Wiener Deconvolution"
    }
    
    enhancement_method = st.selectbox(
        "Enhancement Method",
        list(enhancement_methods.keys()),
        format_func=lambda x: enhancement_methods[x],
        index=0,
        help="Choose the best enhancement method for your image type"
    )
    
    enhance_button = st.button("🚀 Generate Enhanced Image", type="primary")
    
    st.header("🔢 OCR Settings")
    
    ocr_modes = {
        "numbers_precise": "🔢 Precise Number Recognition",
        "measurements": "📏 Measurements (12.51m, 3.4kg, etc.)",
        "scientific_notation": "🧪 Scientific Numbers (1.5e-3, etc.)",
        "currency": "💰 Currency & Financial Numbers",
        "coordinates": "🗺️ Coordinates & GPS Numbers",
        "handwriting": "📝 Handwriting Optimized",
        "print": "🖨️ Printed Text",
        "mixed": "🔀 Mixed Text",
        "numbers": "🔢 Basic Numbers",
        "single_word": "📄 Single Word"
    }
    
    ocr_mode = st.selectbox(
        "OCR Mode",
        list(ocr_modes.keys()),
        format_func=lambda x: ocr_modes[x],
        help="Choose OCR mode based on your text type"
    )
    
    languages = {
        "eng": "English",
        "eng+ara": "English + Arabic",
        "eng+chi_sim": "English + Chinese",
        "eng+fra": "English + French", 
        "eng+deu": "English + German",
        "eng+spa": "English + Spanish",
        "eng+rus": "English + Russian"
    }
    
    language = st.selectbox(
        "Language",
        list(languages.keys()),
        format_func=lambda x: languages[x],
        help="Select OCR language"
    )

# Main content area
if uploaded_file is not None:
    # Load original image
    st.session_state.original_image = Image.open(uploaded_file)
    
    # Generate enhanced image
    if enhance_button:
        with st.spinner("🔄 Generating enhanced image..."):
            try:
                st.session_state.enhanced_image = advanced_image_enhancement(
                    st.session_state.original_image, 
                    enhancement_method
                )
                st.success("✅ Enhanced image generated!")
            except Exception as e:
                st.error(f"Enhancement failed: {str(e)}")
    
    # Display images side by side
    if st.session_state.original_image is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Original Image")
            st.image(st.session_state.original_image, use_column_width=True)
        
        with col2:
            st.subheader("✨ Enhanced Image")
            if st.session_state.enhanced_image is not None:
                st.image(st.session_state.enhanced_image, use_column_width=True)
            else:
                st.info("Click 'Generate Enhanced Image' to see the enhanced version")
    
    # Interactive canvas for selection (only if enhanced image exists)
    if st.session_state.enhanced_image is not None:
        st.subheader("🎯 Select Text Areas for OCR")
        st.markdown("*Draw a rectangle around the text you want to extract*")
        
        # Create canvas with enhanced image as background
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.2)",  # Orange with transparency
            stroke_width=2,
            stroke_color="#FF4444",
            background_image=st.session_state.enhanced_image,
            update_streamlit=True,
            height=min(400, st.session_state.enhanced_image.height),
            width=min(600, st.session_state.enhanced_image.width),
            drawing_mode="rect",
            point_display_radius=0,
            key="canvas",
        )
        
        # OCR Controls
        st.subheader("🔍 OCR Operations")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("✍️ Extract Selected Area", type="primary"):
                if canvas_result.json_data is not None and len(canvas_result.json_data["objects"]) > 0:
                    with st.spinner("🔄 Processing selected area..."):
                        cropped_img = crop_image_from_canvas(st.session_state.enhanced_image, canvas_result)
                        if cropped_img:
                            st.session_state.cropped_image = cropped_img
                            result = extract_text_from_image(cropped_img, ocr_mode, language)
                            st.session_state.last_result = result
                            st.success("✅ OCR completed on selected area!")
                        else:
                            st.error("Failed to crop selected area")
                else:
                    st.warning("Please draw a rectangle on the image first")
        
        with col2:
            if st.button("📄 Process Full Image", type="secondary"):
                with st.spinner("🔄 Processing full enhanced image..."):
                    result = extract_text_from_image(
                        st.session_state.enhanced_image, 
                        ocr_mode, 
                        language
                    )
                    st.session_state.last_result = result
                    st.success("✅ OCR completed on full image!")
        
        with col3:
            if st.button("🔄 Multiple Attempts", type="secondary"):
                if canvas_result.json_data is not None and len(canvas_result.json_data["objects"]) > 0:
                    # Use selected area
                    cropped_img = crop_image_from_canvas(st.session_state.enhanced_image, canvas_result)
                    target_img = cropped_img if cropped_img else st.session_state.enhanced_image
                else:
                    # Use full image
                    target_img = st.session_state.enhanced_image
                
                with st.spinner("🔄 Trying multiple approaches..."):
                    detailed_results, clean_results = multi_attempt_ocr(target_img, language)
                    
                    if detailed_results:
                        st.subheader("🔢 Multiple Attempt Results")
                        with st.expander("View All Attempts", expanded=False):
                            for i, result in enumerate(detailed_results, 1):
                                st.text(f"Attempt {i}: {result}")
                        
                        if clean_results:
                            st.subheader("🎯 Clean Results")
                            for i, result in enumerate(clean_results, 1):
                                st.text(f"{i}. {result}")
                            
                            # Set best result as last result
                            best_result = max(clean_results, key=len) if clean_results else ""
                            st.session_state.last_result = best_result
                            st.success("✅ Multiple attempts completed!")
        
        # Show cropped preview if available
        if st.session_state.cropped_image:
            st.subheader("🔍 Selected Area Preview")
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(st.session_state.cropped_image, caption="Cropped Area", width=200)
        
        # Display last result
        if st.session_state.last_result:
            st.subheader("📋 OCR Result")
            result_text = st.text_area(
                "Extracted Text", 
                st.session_state.last_result, 
                height=150,
                help="The extracted text from OCR processing"
            )
            
            # Copy button functionality
            col1, col2 = st.columns([3, 1])
            with col2:
                if st.button("📋 Copy to Clipboard"):
                    # Create a JavaScript snippet to copy text
                    copy_js = f"""
                    <script>
                    navigator.clipboard.writeText(`{st.session_state.last_result.replace('`', '\\`')}`).then(function() {{
                        console.log('Text copied to clipboard');
                    }});
                    </script>
                    """
                    components.html(copy_js, height=0)
                    st.success("Text copied to clipboard!")
        
        # Manual crop section as backup
        with st.expander("📐 Manual Crop Selection (Alternative)", expanded=False):
            st.info("Use this if the canvas selection doesn't work properly")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                crop_x = st.number_input("X", min_value=0, value=0, key="manual_x")
            with col2:
                crop_y = st.number_input("Y", min_value=0, value=0, key="manual_y")
            with col3:
                crop_w = st.number_input("Width", min_value=1, value=100, key="manual_w")
            with col4:
                crop_h = st.number_input("Height", min_value=1, value=100, key="manual_h")
            
            if st.button("✍️ Extract from Manual Coordinates"):
                if st.session_state.enhanced_image is not None:
                    # Crop the image
                    img_array = np.array(st.session_state.enhanced_image)
                    
                    # Validate coordinates
                    max_y, max_x = img_array.shape[:2]
                    if crop_x + crop_w <= max_x and crop_y + crop_h <= max_y:
                        cropped_array = img_array[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
                        
                        if cropped_array.size > 0:
                            cropped_img = Image.fromarray(cropped_array)
                            st.session_state.cropped_image = cropped_img
                            
                            with st.spinner("🔄 Processing manually selected area..."):
                                result = extract_text_from_image(cropped_img, ocr_mode, language)
                                st.session_state.last_result = result
                            
                            # Show cropped preview
                            st.image(cropped_img, caption="Manually Cropped Area", width=300)
                        else:
                            st.error("Invalid crop coordinates - resulting area is empty")
                    else:
                        st.error(f"Invalid crop coordinates - exceeds image bounds ({max_x}x{max_y})")

else:
    # Welcome screen
    st.markdown("""
    ## 👋 Welcome to Enhanced Image OCR Extractor
    
    Upload an image to get started with advanced OCR processing optimized for number recognition!
    
    ### 🚀 Quick Start:
    1. **📁 Upload an image** using the file uploader in the sidebar
    2. **🎨 Choose enhancement method** - try "Number Recognition Optimized" for numbers
    3. **🚀 Generate enhanced image** to see the improvement
    4. **🎯 Select text areas** using the interactive canvas
    5. **🔍 Extract text** with specialized OCR modes
    """)
    
    # Show feature highlights
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🔢 Number Recognition
        - Precise number extraction
        - Measurement text (12.5m, 3kg)
        - Scientific notation
        - Currency amounts
        - GPS coordinates
        """)
    
    with col2:
        st.markdown("""
        ### 🎨 Image Enhancement
        - 13 enhancement methods
        - Noise reduction
        - Contrast boosting
        - Edge sharpening
        - Handwriting optimization
        """)
    
    with col3:
        st.markdown("""
        ### 🎯 Interactive Features
        - Canvas selection tool
        - Side-by-side comparison
        - Multiple OCR attempts
        - Multi-language support
        - Copy to clipboard
        """)

# Tips section
with st.expander("💡 Tips for Better Results"):
    st.markdown("""
    ### 🎯 Workflow:
    1. **Upload your original image**
    2. **Choose enhancement method and generate enhanced image**
    3. **Compare original vs enhanced image**
    4. **Work on the enhanced image for all OCR operations**
    5. **Select text areas or process full enhanced image**

    ### 🔢 NUMBER RECOGNITION ENHANCEMENTS:
    - **Number Recognition Optimized** - Best for pure numbers and digits
    - **Measurement Text Enhanced** - Perfect for measurements like '12.51m', '3.4kg'
    - **Digital/Printed Numbers** - Optimized for LCD/LED displays and printed digits

    ### 🎯 SPECIALIZED OCR MODES FOR NUMBERS:
    - **Precise Number Recognition** - Pure numbers with decimal points
    - **Measurements** - Numbers with units (m, kg, cm, ft, etc.)
    - **Scientific Numbers** - Scientific notation (1.5e-3, 2×10⁵)
    - **Currency** - Money amounts ($123.45, €99.99, ¥1000)
    - **Coordinates** - GPS coordinates (40.7128°N, -74.0060°W)

    ### 📖 BEST PRACTICES:
    ✅ Use 'Number Recognition Optimized' enhancement for best number clarity  
    ✅ Choose the right OCR mode for your number type  
    ✅ Select tight crops around just the numbers you need  
    ✅ Try 'Multiple Attempts' - it tests all number recognition methods  
    ✅ Enhanced image provides significantly better number accuracy  
    ✅ Compare original vs enhanced to see digit improvement
    """)

# Footer
st.markdown("---")
st.markdown("**🔧 Enhanced Image OCR Extractor** - Optimized for Number Recognition | Built with ❤️ using Streamlit")
