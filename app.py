import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
import pytesseract
import io
import base64
from scipy import ndimage
from skimage import morphology, exposure, restoration, filters
import streamlit.components.v1 as components

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
            # Create a motion blur kernel
            psf = np.ones((5, 5)) / 25
            result_float = restoration.wiener(gray, psf, balance=0.1)
            result = (result_float * 255).astype(np.uint8)
        except:
            result = gray

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

def perform_ocr(img, ocr_mode='handwriting', language='eng'):
    """Perform OCR on the image"""
    try:
        config = get_ocr_config(ocr_mode, language)
        text = pytesseract.image_to_string(img, config=config)
        return text.strip()
    except Exception as e:
        st.error(f"OCR Error: {str(e)}")
        return ""

def multi_attempt_ocr(img, language='eng'):
    """Try multiple OCR approaches on the image with focus on numbers"""
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

def image_to_base64(image):
    """Convert PIL image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

def create_image_selector(image_b64, img_width, img_height):
    """Create HTML/JS image selector component"""
    
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            .image-container {{
                position: relative;
                display: inline-block;
                border: 2px solid #ddd;
                border-radius: 8px;
                margin: 10px 0;
            }}
            
            .selectable-image {{
                display: block;
                max-width: 800px;
                height: auto;
                cursor: crosshair;
            }}
            
            .selection-overlay {{
                position: absolute;
                border: 2px dashed #ff4444;
                background-color: rgba(255, 68, 68, 0.1);
                pointer-events: none;
                display: none;
            }}
            
            .controls {{
                margin-top: 10px;
                padding: 10px;
                background-color: #f8f9fa;
                border-radius: 5px;
                display: flex;
                gap: 10px;
                align-items: center;
                flex-wrap: wrap;
            }}
            
            .btn {{
                padding: 8px 16px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-weight: bold;
                transition: background-color 0.3s;
            }}
            
            .btn-primary {{ background-color: #007bff; color: white; }}
            .btn-primary:hover {{ background-color: #0056b3; }}
            .btn-secondary {{ background-color: #6c757d; color: white; }}
            .btn-secondary:hover {{ background-color: #545b62; }}
            .btn-success {{ background-color: #28a745; color: white; }}
            .btn-success:hover {{ background-color: #1e7e34; }}
            
            .selection-info {{
                font-family: monospace;
                font-size: 12px;
                color: #666;
                padding: 5px 10px;
                background-color: #e9ecef;
                border-radius: 3px;
            }}
            
            .instructions {{
                margin-bottom: 10px;
                padding: 10px;
                background-color: #e3f2fd;
                border-left: 4px solid #2196f3;
                border-radius: 0 5px 5px 0;
                font-size: 14px;
            }}
        </style>
    </head>
    <body>
        <div class="instructions">
            📐 <strong>How to select:</strong> Click and drag on the image to select the text area you want to extract. You can make multiple selections and extract each one.
        </div>
        
        <div class="image-container">
            <img id="selectableImage" src="{image_b64}" class="selectable-image" alt="Enhanced Image">
            <div id="selectionOverlay" class="selection-overlay"></div>
        </div>
        
        <div class="controls">
            <button id="clearBtn" class="btn btn-secondary">🗑️ Clear Selection</button>
            <button id="extractBtn" class="btn btn-primary" disabled>✍️ Extract Selection</button>
            <div id="selectionInfo" class="selection-info">No selection made</div>
        </div>
        
        <script>
            const image = document.getElementById('selectableImage');
            const overlay = document.getElementById('selectionOverlay');
            const clearBtn = document.getElementById('clearBtn');
            const extractBtn = document.getElementById('extractBtn');
            const selectionInfo = document.getElementById('selectionInfo');
            
            let isSelecting = false;
            let startX, startY, currentSelection = null;
            
            function getImageCoordinates(e) {{
                const rect = image.getBoundingClientRect();
                const scaleX = image.naturalWidth / image.clientWidth;
                const scaleY = image.naturalHeight / image.clientHeight;
                
                return {{
                    x: Math.round((e.clientX - rect.left) * scaleX),
                    y: Math.round((e.clientY - rect.top) * scaleY),
                    displayX: e.clientX - rect.left,
                    displayY: e.clientY - rect.top
                }};
            }}
            
            image.addEventListener('mousedown', (e) => {{
                isSelecting = true;
                const coords = getImageCoordinates(e);
                startX = coords.displayX;
                startY = coords.displayY;
                
                overlay.style.left = startX + 'px';
                overlay.style.top = startY + 'px';
                overlay.style.width = '0px';
                overlay.style.height = '0px';
                overlay.style.display = 'block';
                
                e.preventDefault();
            }});
            
            image.addEventListener('mousemove', (e) => {{
                if (!isSelecting) return;
                
                const coords = getImageCoordinates(e);
                const width = coords.displayX - startX;
                const height = coords.displayY - startY;
                
                overlay.style.width = Math.abs(width) + 'px';
                overlay.style.height = Math.abs(height) + 'px';
                overlay.style.left = (width < 0 ? coords.displayX : startX) + 'px';
                overlay.style.top = (height < 0 ? coords.displayY : startY) + 'px';
                
                e.preventDefault();
            }});
            
            image.addEventListener('mouseup', (e) => {{
                if (!isSelecting) return;
                
                isSelecting = false;
                const coords = getImageCoordinates(e);
                
                // Calculate actual selection coordinates
                const scaleX = image.naturalWidth / image.clientWidth;
                const scaleY = image.naturalHeight / image.clientHeight;
                
                const x1 = Math.round(startX * scaleX);
                const y1 = Math.round(startY * scaleY);
                const x2 = coords.x;
                const y2 = coords.y;
                
                const finalX = Math.min(x1, x2);
                const finalY = Math.min(y1, y2);
                const finalW = Math.abs(x2 - x1);
                const finalH = Math.abs(y2 - y1);
                
                if (finalW > 10 && finalH > 10) {{
                    currentSelection = {{
                        x: finalX,
                        y: finalY,
                        width: finalW,
                        height: finalH
                    }};
                    
                    selectionInfo.textContent = `Selection: ${{finalW}}×${{finalH}} pixels at (${{finalX}}, ${{finalY}})`;
                    extractBtn.disabled = false;
                }} else {{
                    clearSelection();
                }}
                
                e.preventDefault();
            }});
            
            function clearSelection() {{
                overlay.style.display = 'none';
                currentSelection = null;
                selectionInfo.textContent = 'No selection made';
                extractBtn.disabled = true;
            }}
            
            clearBtn.addEventListener('click', clearSelection);
            
            extractBtn.addEventListener('click', () => {{
                if (currentSelection) {{
                    // Send selection data to Streamlit
                    window.parent.postMessage({{
                        type: 'selection',
                        data: currentSelection
                    }}, '*');
                }}
            }});
            
            // Prevent context menu on image
            image.addEventListener('contextmenu', (e) => e.preventDefault());
        </script>
    </body>
    </html>
    """
    
    return html_code

# Initialize session state
if 'original_image' not in st.session_state:
    st.session_state.original_image = None
if 'enhanced_image' not in st.session_state:
    st.session_state.enhanced_image = None
if 'ocr_result' not in st.session_state:
    st.session_state.ocr_result = ""
if 'selection_data' not in st.session_state:
    st.session_state.selection_data = None

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
            ('digit_sharpening', '🎯 Digital/Printed Numbers'),
            ('auto_adaptive', '🤖 Auto Adaptive Enhancement'),
            ('handwriting_optimized', '✍️ Handwriting Optimized'),
            ('high_contrast', '⚡ High Contrast Boost'),
            ('noise_reduction', '🧹 Advanced Noise Reduction'),
            ('edge_sharpening', '📐 Edge Sharpening'),
            ('brightness_contrast', '💡 Brightness & Contrast'),
            ('histogram_equalization', '📊 Histogram Equalization'),
            ('unsharp_masking', '🔍 Unsharp Masking'),
            ('morphological', '🔄 Morphological Enhancement'),
            ('wiener_deconvolution', '🌟 Wiener Deconvolution')
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
            ('scientific_notation', '🧪 Scientific Numbers (1.5e-3, etc.)'),
            ('currency', '💰 Currency & Financial Numbers'),
            ('coordinates', '🗺️ Coordinates & GPS Numbers'),
            ('handwriting', '📝 Handwriting Optimized'),
            ('print', '🖨️ Printed Text'),
            ('mixed', '🔀 Mixed Text'),
            ('numbers', '🔢 Basic Numbers'),
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
            ('eng+fra', 'English + French'),
            ('eng+deu', 'English + German'),
            ('eng+spa', 'English + Spanish'),
            ('eng+rus', 'English + Russian')
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
    
    # OCR Operations with Interactive Selection
    if st.session_state.enhanced_image is not None:
        st.header("🔍 Interactive Text Selection & OCR")
        
        # Convert enhanced image to base64
        enhanced_b64 = image_to_base64(st.session_state.enhanced_image)
        img_width, img_height = st.session_state.enhanced_image.size
        
        # Create interactive image selector
        selector_html = create_image_selector(enhanced_b64, img_width, img_height)
        
        # Display the interactive selector
        selection_result = components.html(
            selector_html,
            height=600,
            scrolling=True
        )
        
        # OCR Control buttons
        st.subheader("🎯 OCR Operations")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("✍️ Extract from Selection", type="primary"):
                # For now, we'll show a message about selection
                # In a real implementation, you'd get the selection coordinates from the JavaScript
                st.info("🖱️ Please make a selection on the image above, then click the 'Extract Selection' button in the image interface.")
        
        with col2:
            if st.button("🔄 Multiple Attempts on Selection"):
                st.info("🖱️ Please make a selection on the image above first.")
        
        with col3:
            if st.button("📄 Process Full Enhanced Image"):
                with st.spinner("Processing full enhanced image..."):
                    result = perform_ocr(st.session_state.enhanced_image, ocr_mode, language)
                    st.session_state.ocr_result = result
        
        # Manual crop option as fallback
        with st.expander("📐 Manual Crop Alternative (if selection tool doesn't work)"):
            st.info("Use this if the interactive selection tool above is not working properly.")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                crop_x = st.number_input("X", min_value=0, max_value=st.session_state.enhanced_image.size[0], value=0, key="manual_x")
            with col2:
                crop_y = st.number_input("Y", min_value=0, max_value=st.session_state.enhanced_image.size[1], value=0, key="manual_y")
            with col3:
                crop_w = st.number_input("Width", min_value=1, max_value=st.session_state.enhanced_image.size[0], value=min(200, st.session_state.enhanced_image.size[0]), key="manual_w")
            with col4:
                crop_h = st.number_input("Height", min_value=1, max_value=st.session_state.enhanced_image.size[1], value=min(100, st.session_state.enhanced_image.size[1]), key="manual_h")
            
            # Show manual crop preview
            if crop_w > 0 and crop_h > 0:
                try:
                    cropped_image = st.session_state.enhanced_image.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))
                    st.image(cropped_image, caption="Manual Crop Preview", width=300)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("✍️ OCR Manual Crop", key="manual_ocr"):
                            with st.spinner("Extracting text from manual crop..."):
                                result = perform_ocr(cropped_image, ocr_mode, language)
                                st.session_state.ocr_result = result
                    
                    with col2:
                        if st.button("🔄 Multiple Attempts Manual Crop", key="manual_multi"):
                            with st.spinner("Trying multiple OCR approaches on manual crop..."):
                                all_results, clean_results = multi_attempt_ocr(cropped_image, language)
                                
                                st.subheader("🔢 Multiple Attempt Results (Manual Crop)")
                                for i, result in enumerate(all_results, 1):
                                    st.text(f"Attempt {i}: {result}")
                                
                                if clean_results:
                                    best_result = max(clean_results, key=len)
                                    st.session_state.ocr_result = best_result
                                    st.success(f"🎯 Best Result: {best_result}")
                except Exception as e:
                    st.error(f"Crop error: {str(e)}")
        
        # JavaScript communication handler (for future implementation)
        st.markdown("""
        <script>
        // Listen for messages from the image selector
        window.addEventListener('message', function(event) {
            if (event.data.type === 'selection') {
                // Store selection data in Streamlit session state
                console.log('Selection received:', event.data.data);
                // This would need to be implemented with a proper Streamlit-JS bridge
            }
        });
        </script>
        """, unsafe_allow_html=True)
        
        # Display current selection info (placeholder)
        if st.session_state.selection_data:
            st.info(f"📐 Current Selection: {st.session_state.selection_data}")
        
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
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Characters", len(st.session_state.ocr_result))
                with col2:
                    st.metric("Words", len(st.session_state.ocr_result.split()))
                
                # Copy button functionality
                if st.button("📋 Copy to Clipboard"):
                    st.code(st.session_state.ocr_result)
                    st.success("Text displayed above - you can now copy it!")

# Additional OCR tools section
if st.session_state.enhanced_image is not None:
    st.header("🛠️ Advanced OCR Tools")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔢 Batch Number Extraction")
        if st.button("🎯 Extract All Numbers", help="Automatically detect and extract all numbers from the image"):
            with st.spinner("Scanning for numbers..."):
                # Try multiple number-focused OCR modes
                number_results = []
                number_modes = ['numbers_precise', 'measurements', 'scientific_notation', 'currency']
                
                for mode in number_modes:
                    try:
                        result = perform_ocr(st.session_state.enhanced_image, mode, language)
                        if result and result not in number_results:
                            number_results.append(f"[{mode}] {result}")
                    except:
                        continue
                
                if number_results:
                    st.subheader("📊 All Detected Numbers:")
                    for result in number_results:
                        st.text(result)
                else:
                    st.warning("No numbers detected")
    
    with col2:
        st.subheader("📝 Quick Text Extraction")
        if st.button("⚡ Quick OCR (All Text)", help="Fast extraction of all text using optimal settings"):
            with st.spinner("Quick OCR processing..."):
                result = perform_ocr(st.session_state.enhanced_image, 'mixed', language)
                if result:
                    st.text_area("Quick OCR Result", result, height=100)
                    st.session_state.ocr_result = result
                else:
                    st.warning("No text detected")

# Tips section
with st.expander("💡 Tips for Better Results"):
    st.markdown("""
    ### 🎯 Tips for Better Number Recognition:
    - **For Pure Numbers:** Use "Number Recognition Optimized" enhancement + "Precise Number Recognition" OCR
    - **For Measurements:** Use "Measurement Text Enhanced" + "Measurements" mode for "12.51m", "3.4kg", etc.
    - **For Scientific Numbers:** Use "Scientific Numbers" mode for "1.5e-3", "2.4×10⁵", etc.
    - **For Currency:** Use "Currency & Financial Numbers" for "$123.45", "€99.99", etc.
    - **For GPS/Coordinates:** Use "Coordinates & GPS Numbers" for "40.7128°N", etc.
    - **Digital Displays:** Use "Digital/Printed Numbers" enhancement for LCD/LED numbers
    - **Best Practice:** Try "Multiple Attempts" - it tests all number recognition methods
    
    ### 📐 Using the Selection Tool:
    - **Interactive Selection:** Click and drag on the enhanced image to select text areas
    - **Precise Selection:** Make tight selections around the text you want to extract
    - **Multiple Selections:** You can make multiple selections and extract each one
    - **Clear Selection:** Use the "Clear Selection" button to start over
    - **Fallback Option:** Use manual crop if the interactive tool doesn't work
    
    ### 📋 General Tips:
    - **Step 1:** Choose the best enhancement method for your image type
    - **Step 2:** Generate the enhanced image (this is your working canvas)
    - **Step 3:** Use the interactive selection tool to select text areas
    - **Step 4:** Extract text using appropriate OCR mode
    - Try "Handwriting Optimized" for handwritten notes
    - Use "High Contrast Boost" for faded or low-contrast text
    - Compare original vs enhanced to see the improvement
    """)

# Performance metrics
if st.session_state.enhanced_image is not None:
    with st.expander("📊 Image Information"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Original Size", f"{st.session_state.original_image.size[0]}×{st.session_state.original_image.size[1]}")
        
        with col2:
            st.metric("Enhanced Size", f"{st.session_state.enhanced_image.size[0]}×{st.session_state.enhanced_image.size[1]}")
        
        with col3:
            # Calculate file size difference
            original_buffer = io.BytesIO()
            st.session_state.original_image.save(original_buffer, format='PNG')
            original_size = len(original_buffer.getvalue())
            
            enhanced_buffer = io.BytesIO()
            st.session_state.enhanced_image.save(enhanced_buffer, format='PNG')
            enhanced_size = len(enhanced_buffer.getvalue())
            
            st.metric("Size Change", f"{enhanced_size/original_size:.2f}x")

# Footer
st.markdown("---")
st.markdown("🔧 **Enhanced Image OCR Extractor** - Optimized for Number Recognition with Advanced Image Enhancement")
st.markdown("### 🔧 Features:")
st.markdown("""
- ✨ **13 Advanced Enhancement Methods** - Specialized algorithms for different image types
- 🔢 **10 Specialized OCR Modes** - Optimized for numbers, measurements, currency, and more
- 📐 **Interactive Selection Tool** - Click and drag to select text areas
- 🔄 **Multiple Attempt Mode** - Tries all methods automatically for best results
- 🌍 **Multi-language Support** - Support for multiple languages
- 📊 **Real-time Comparison** - See original vs enhanced images side by side
""")

# Debug information (only show if needed)
if st.checkbox("🔧 Show Debug Info", help="Show technical information for troubleshooting"):
    st.subheader("Debug Information")
    st.write("Session State Keys:", list(st.session_state.keys()))
    if st.session_state.enhanced_image:
        st.write("Enhanced Image Mode:", st.session_state.enhanced_image.mode)
        st.write("Enhanced Image Format:", getattr(st.session_state.enhanced_image, 'format', 'Unknown'))
    
    # Test Tesseract
    if st.button("🧪 Test Tesseract Installation"):
        try:
            version = pytesseract.get_tesseract_version()
            st.success(f"✅ Tesseract Version: {version}")
            
            # Create a simple test image
            test_img = Image.new('RGB', (200, 50), color='white')
            draw = ImageDraw.Draw(test_img)
            draw.text((10, 10), "Test 12345", fill='black')
            
            test_result = pytesseract.image_to_string(test_img, config='--psm 8')
            st.info(f"Test OCR Result: '{test_result.strip()}'")
            
        except Exception as e:
            st.error(f"❌ Tesseract Error: {e}")
            st.info("Please ensure Tesseract is properly installed. See setup instructions in the README.")
