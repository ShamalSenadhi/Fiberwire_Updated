import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
import pytesseract
import io
import base64
from scipy import ndimage
from skimage import morphology, exposure, restoration, filters

# Set page config
st.set_page_config(
    page_title="🔧 Enhanced Image OCR Extractor",
    page_icon="🔧",
    layout="wide"
)

def advanced_image_enhancement(img, method='auto_adaptive'):
    """Apply advanced enhancement methods to generate improved image"""
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    if method == 'number_optimized':
        # Specialized enhancement for number recognition
        denoised = cv2.fastNlMeansDenoising(gray, h=12)
        clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(6,6))
        enhanced = clahe.apply(denoised)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        cleaned = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
        smoothed = cv2.bilateralFilter(cleaned, 9, 80, 80)
        binary = cv2.adaptiveThreshold(smoothed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        result = cv2.dilate(binary, kernel_dilate, iterations=1)

    elif method == 'measurement_enhanced':
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4,4))
        enhanced = clahe.apply(denoised)
        kernel_sharp = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
        sharpened = cv2.filter2D(enhanced, -1, kernel_sharp)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        result = cv2.morphologyEx(sharpened, cv2.MORPH_OPEN, kernel)

    elif method == 'high_contrast':
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        gamma = 0.8
        lookup_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        result = cv2.LUT(enhanced, lookup_table)

    else:  # auto_adaptive
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        smoothed = cv2.bilateralFilter(enhanced, 9, 75, 75)
        gaussian = cv2.GaussianBlur(smoothed, (0, 0), 2.0)
        result = cv2.addWeighted(smoothed, 1.5, gaussian, -0.5, 0)

    # Convert back to RGB
    if len(result.shape) == 2:
        result_rgb = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
    else:
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    return Image.fromarray(result_rgb)

def get_ocr_config(mode, language):
    """Get OCR configuration based on mode"""
    basic_numbers = '0123456789'
    decimal_numbers = '0123456789.,'
    measurement_chars = '0123456789.,-+()[]{}mkcglbftinMKCGLBFTIN°%'
    
    configs = {
        'numbers_precise': f'--oem 3 --psm 8 -l {language} -c tessedit_char_whitelist={decimal_numbers}',
        'measurements': f'--oem 3 --psm 8 -l {language} -c tessedit_char_whitelist={measurement_chars}',
        'handwriting': f'--oem 3 --psm 8 -l {language}',
        'print': f'--oem 3 --psm 6 -l {language}',
        'mixed': f'--oem 3 --psm 3 -l {language}',
        'single_word': f'--oem 3 --psm 8 -l {language}'
    }
    return configs.get(mode, configs['numbers_precise'])

def perform_ocr(img, ocr_mode='mixed', language='eng'):
    """Perform OCR on the image"""
    try:
        config = get_ocr_config(ocr_mode, language)
        text = pytesseract.image_to_string(img, config=config)
        return text.strip()
    except Exception as e:
        st.error(f"OCR Error: {str(e)}")
        return ""

def multi_attempt_ocr(img, language='eng'):
    """Try multiple OCR approaches on the image"""
    try:
        results = []
        modes = ['numbers_precise', 'measurements', 'handwriting', 'mixed', 'single_word']
        
        for ocr_mode in modes:
            try:
                config = get_ocr_config(ocr_mode, language)
                text = pytesseract.image_to_string(img, config=config).strip()
                if text and text not in [r.split('] ', 1)[1] for r in results if '] ' in r]:
                    results.append(f"[{ocr_mode}] {text}")
            except:
                continue
        
        # Try with different thresholding
        try:
            gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binary_img = Image.fromarray(binary)
            
            for ocr_mode in ['numbers_precise', 'measurements']:
                try:
                    config = get_ocr_config(ocr_mode, language)
                    text = pytesseract.image_to_string(binary_img, config=config).strip()
                    if text and f"[{ocr_mode}_thresh] {text}" not in results:
                        results.append(f"[{ocr_mode}_thresh] {text}")
                except:
                    continue
        except:
            pass
        
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

# Initialize session state
if 'original_image' not in st.session_state:
    st.session_state.original_image = None
if 'enhanced_image' not in st.session_state:
    st.session_state.enhanced_image = None
if 'ocr_result' not in st.session_state:
    st.session_state.ocr_result = ""
if 'crop_coords' not in st.session_state:
    st.session_state.crop_coords = None

# Main UI
st.title("🔧 Enhanced Image OCR Extractor")
st.markdown("**Optimized for Number Recognition with Advanced Image Enhancement**")

# Sidebar for settings
st.sidebar.header("⚙️ Settings")

# File upload
uploaded_file = st.file_uploader(
    "Upload Image", 
    type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
    help="Upload an image containing text or numbers to extract"
)

if uploaded_file is not None:
    # Load original image
    original_image = Image.open(uploaded_file)
    st.session_state.original_image = original_image
    
    # Enhancement settings
    st.sidebar.subheader("🎨 Image Enhancement")
    enhancement_method = st.sidebar.selectbox(
        "Enhancement Method",
        [
            ('number_optimized', '🔢 Number Recognition Optimized'),
            ('measurement_enhanced', '📏 Measurement Text Enhanced'),
            ('auto_adaptive', '🤖 Auto Adaptive Enhancement'),
            ('high_contrast', '⚡ High Contrast Boost'),
        ],
        format_func=lambda x: x[1]
    )[0]
    
    # OCR settings
    st.sidebar.subheader("📝 OCR Settings")
    ocr_mode = st.sidebar.selectbox(
        "OCR Mode",
        [
            ('numbers_precise', '🔢 Precise Number Recognition'),
            ('measurements', '📏 Measurements (12.51m, 3.4kg, etc.)'),
            ('handwriting', '📝 Handwriting Optimized'),
            ('print', '🖨️ Printed Text'),
            ('mixed', '🔀 Mixed Text'),
            ('single_word', '📄 Single Word')
        ],
        format_func=lambda x: x[1]
    )[0]
    
    language = st.sidebar.selectbox(
        "Language",
        [
            ('eng', 'English'),
            ('eng+ara', 'English + Arabic'),
            ('eng+chi_sim', 'English + Chinese'),
        ],
        format_func=lambda x: x[1]
    )[0]
    
    # Generate enhanced image
    if st.sidebar.button("🚀 Generate Enhanced Image", type="primary"):
        with st.spinner("Generating enhanced image..."):
            enhanced_image = advanced_image_enhancement(original_image, enhancement_method)
            st.session_state.enhanced_image = enhanced_image
            st.sidebar.success("✅ Enhanced image generated!")
    
    # Display images
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Original Image")
        st.image(original_image, use_column_width=True)
        st.caption(f"Size: {original_image.size[0]}×{original_image.size[1]}")
    
    with col2:
        if st.session_state.enhanced_image is not None:
            st.subheader("✨ Enhanced Image")
            st.image(st.session_state.enhanced_image, use_column_width=True)
            st.caption(f"Size: {st.session_state.enhanced_image.size[0]}×{st.session_state.enhanced_image.size[1]}")
        else:
            st.subheader("✨ Enhanced Image")
            st.info("Click 'Generate Enhanced Image' to see the enhanced version")
    
    # Manual crop and OCR section
    if st.session_state.enhanced_image is not None:
        st.header("🔍 Text Selection & OCR")
        st.info("📐 Select the area containing text by setting the crop coordinates below:")
        
        # Get image dimensions
        img_width, img_height = st.session_state.enhanced_image.size
        
        # Crop coordinates input
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            crop_x = st.number_input("X (Left)", min_value=0, max_value=img_width-1, value=0, key="crop_x")
        with col2:
            crop_y = st.number_input("Y (Top)", min_value=0, max_value=img_height-1, value=0, key="crop_y")
        with col3:
            crop_w = st.number_input("Width", min_value=1, max_value=img_width, value=min(200, img_width), key="crop_w")
        with col4:
            crop_h = st.number_input("Height", min_value=1, max_value=img_height, value=min(100, img_height), key="crop_h")
        
        # Validate and show crop preview
        if crop_x + crop_w <= img_width and crop_y + crop_h <= img_height:
            try:
                # Create cropped image
                cropped_image = st.session_state.enhanced_image.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))
                
                # Show preview
                st.subheader("🖼️ Selected Area Preview")
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.image(cropped_image, caption="Selected Area", width=250)
                    st.caption(f"Selection: {crop_w}×{crop_h} pixels at ({crop_x}, {crop_y})")
                
                with col2:
                    # OCR buttons
                    st.subheader("🎯 Extract Text")
                    
                    if st.button("✍️ Extract Text from Selection", type="primary", key="extract_selection"):
                        with st.spinner("Extracting text from selection..."):
                            result = perform_ocr(cropped_image, ocr_mode, language)
                            st.session_state.ocr_result = result
                            if result:
                                st.success(f"✅ Extracted: {result}")
                            else:
                                st.warning("No text detected in selection")
                    
                    if st.button("🔄 Try Multiple Methods", key="multi_extract"):
                        with st.spinner("Trying multiple OCR approaches..."):
                            all_results, clean_results = multi_attempt_ocr(cropped_image, language)
                            
                            if all_results:
                                st.subheader("🔢 Multiple Attempt Results:")
                                for i, result in enumerate(all_results, 1):
                                    st.text(f"{i}. {result}")
                                
                                if clean_results:
                                    best_result = max(clean_results, key=len)
                                    st.session_state.ocr_result = best_result
                                    st.success(f"🎯 Best Result: {best_result}")
                            else:
                                st.warning("No text detected with any method")
                    
                    if st.button("📄 Process Full Image", key="full_image"):
                        with st.spinner("Processing full enhanced image..."):
                            result = perform_ocr(st.session_state.enhanced_image, ocr_mode, language)
                            st.session_state.ocr_result = result
                            if result:
                                st.success(f"✅ Full image result available below")
                            else:
                                st.warning("No text detected in full image")
                
            except Exception as e:
                st.error(f"Crop error: {str(e)}")
        else:
            st.error("❌ Invalid crop coordinates! Selection goes beyond image boundaries.")
        
        # Display results
        if st.session_state.ocr_result:
            st.header("📋 OCR Results")
            
            result_container = st.container()
            with result_container:
                st.text_area(
                    "Extracted Text",
                    value=st.session_state.ocr_result,
                    height=150,
                    key="result_text"
                )
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Characters", len(st.session_state.ocr_result))
                with col2:
                    st.metric("Words", len(st.session_state.ocr_result.split()))
                with col3:
                    # Copy button functionality
                    if st.button("📋 Copy Text"):
                        st.code(st.session_state.ocr_result)
                        st.success("✅ Text displayed above - you can copy it!")

# Quick preset selections
if st.session_state.enhanced_image is not None:
    st.header("⚡ Quick Presets")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔢 Top-Left Numbers"):
            # Set coordinates for top-left quarter
            st.session_state.crop_x = 0
            st.session_state.crop_y = 0
            st.session_state.crop_w = st.session_state.enhanced_image.size[0] // 2
            st.session_state.crop_h = st.session_state.enhanced_image.size[1] // 2
            st.rerun()
    
    with col2:
        if st.button("📏 Center Area"):
            # Set coordinates for center area
            w, h = st.session_state.enhanced_image.size
            st.session_state.crop_x = w // 4
            st.session_state.crop_y = h // 4
            st.session_state.crop_w = w // 2
            st.session_state.crop_h = h // 2
            st.rerun()
    
    with col3:
        if st.button("📄 Full Width"):
            # Set coordinates for full width, middle height
            w, h = st.session_state.enhanced_image.size
            st.session_state.crop_x = 0
            st.session_state.crop_y = h // 3
            st.session_state.crop_w = w
            st.session_state.crop_h = h // 3
            st.rerun()

# Tips section
with st.expander("💡 Tips for Better Results"):
    st.markdown("""
    ### 🎯 How to Use:
    1. **Upload an image** containing text or numbers
    2. **Choose enhancement method** - try "Number Recognition Optimized" for numbers
    3. **Generate enhanced image** - this creates an optimized version
    4. **Set crop coordinates** to select the text area (X, Y, Width, Height)
    5. **Preview your selection** to ensure it covers the text
    6. **Extract text** using the appropriate OCR mode
    
    ### 📐 Setting Crop Coordinates:
    - **X, Y**: Top-left corner of your selection (0,0 is top-left of image)
    - **Width, Height**: Size of the selection area
    - **Quick Presets**: Use the preset buttons for common selections
    - **Preview**: Always check the preview to ensure you've selected the right area
    
    ### 🔢 Best Practices for Numbers:
    - Use "Number Recognition Optimized" enhancement
    - Use "Precise Number Recognition" or "Measurements" OCR mode
    - Make tight selections around the numbers
    - Try "Multiple Methods" if single extraction doesn't work
    
    ### 🚀 Troubleshooting:
    - If no text is detected, try different enhancement methods
    - Adjust crop coordinates to better frame the text
    - Use "Multiple Methods" to try all approaches
    - Ensure text is clear and not too small in the selection
    """)

# Footer
st.markdown("---")
st.markdown("🔧 **Enhanced Image OCR Extractor** - Simple and Effective Text Recognition")
