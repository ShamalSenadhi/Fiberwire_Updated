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
        from skimage import restoration

        # Create a motion blur kernel
        psf = np.ones((5, 5)) / 25
        result_float = restoration.wiener(gray, psf, balance=0.1)
        result = (result_float * 255).astype(np.uint8)

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

def img_to_base64(img):
    """Convert PIL image to base64 string"""
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

def create_canvas_component(img_base64, canvas_id="canvas"):
    """Create interactive canvas component for image selection"""
    canvas_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ margin: 0; padding: 10px; font-family: Arial, sans-serif; }}
            canvas {{ 
                border: 2px solid #444; 
                cursor: crosshair; 
                border-radius: 5px; 
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                display: block;
                margin: 0 auto;
            }}
            .controls {{ 
                text-align: center; 
                margin: 10px 0;
                display: flex;
                gap: 10px;
                justify-content: center;
                flex-wrap: wrap;
            }}
            button {{ 
                padding: 8px 15px; 
                border: none; 
                border-radius: 5px; 
                cursor: pointer; 
                font-weight: bold;
                transition: background-color 0.3s;
            }}
            .clear-btn {{ background-color: #6c757d; color: white; }}
            .clear-btn:hover {{ background-color: #545b62; }}
            .extract-btn {{ background-color: #007bff; color: white; }}
            .extract-btn:hover {{ background-color: #0056b3; }}
            .status {{ 
                text-align: center; 
                margin: 10px 0; 
                padding: 8px; 
                background-color: #f8f9fa; 
                border-radius: 5px;
                font-family: monospace;
                min-height: 20px;
            }}
        </style>
    </head>
    <body>
        <div class="status" id="status">Drag on the image to select text areas for OCR extraction</div>
        <canvas id="{canvas_id}" width="800" height="600"></canvas>
        <div class="controls">
            <button class="clear-btn" onclick="clearSelection()">🗑️ Clear Selection</button>
            <button class="extract-btn" onclick="extractSelection()">✍️ Extract Selected Area</button>
        </div>
        
        <script>
            const canvas = document.getElementById('{canvas_id}');
            const ctx = canvas.getContext('2d');
            const status = document.getElementById('status');
            
            let img = new Image();
            let startX, startY, w, h;
            let dragging = false;
            let hasSelection = false;
            
            // Load and display image
            img.onload = function() {{
                const maxWidth = 800;
                const maxHeight = 600;
                let {{ width, height }} = img;
                
                // Calculate scaling to fit canvas
                const scale = Math.min(maxWidth/width, maxHeight/height);
                const scaledWidth = width * scale;
                const scaledHeight = height * scale;
                
                canvas.width = scaledWidth;
                canvas.height = scaledHeight;
                
                ctx.drawImage(img, 0, 0, scaledWidth, scaledHeight);
                status.innerText = `Image loaded: ${{scaledWidth.toFixed(0)}}×${{scaledHeight.toFixed(0)}} - Drag to select text areas`;
            }};
            
            img.src = '{img_base64}';
            
            // Mouse events for selection
            canvas.addEventListener('mousedown', function(e) {{
                dragging = true;
                hasSelection = false;
                const rect = canvas.getBoundingClientRect();
                startX = e.clientX - rect.left;
                startY = e.clientY - rect.top;
            }});
            
            canvas.addEventListener('mousemove', function(e) {{
                if (!dragging) return;
                
                const rect = canvas.getBoundingClientRect();
                const mx = e.clientX - rect.left;
                const my = e.clientY - rect.top;
                w = mx - startX;
                h = my - startY;
                
                // Redraw image
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                
                // Draw selection rectangle
                ctx.strokeStyle = '#ff4444';
                ctx.lineWidth = 2;
                ctx.setLineDash([8, 4]);
                ctx.strokeRect(startX, startY, w, h);
                
                ctx.fillStyle = 'rgba(255, 68, 68, 0.1)';
                ctx.fillRect(startX, startY, w, h);
                
                hasSelection = Math.abs(w) > 5 && Math.abs(h) > 5;
                
                if (hasSelection) {{
                    status.innerText = `Selection: ${{Math.abs(w).toFixed(0)}}×${{Math.abs(h).toFixed(0)}} pixels - Ready for OCR`;
                }}
            }});
            
            canvas.addEventListener('mouseup', function() {{
                dragging = false;
            }});
            
            function clearSelection() {{
                hasSelection = false;
                w = h = 0;
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
                status.innerText = 'Selection cleared. Drag to select text areas.';
            }}
            
            function extractSelection() {{
                if (!hasSelection) {{
                    alert('Please make a selection first');
                    return;
                }}
                
                // Calculate crop coordinates
                const scaleX = img.width / canvas.width;
                const scaleY = img.height / canvas.height;
                
                const cropX = (w > 0 ? startX : startX + w) * scaleX;
                const cropY = (h > 0 ? startY : startY + h) * scaleY;
                const cropW = Math.abs(w) * scaleX;
                const cropH = Math.abs(h) * scaleY;
                
                // Create cropped image
                const cropCanvas = document.createElement('canvas');
                const cropCtx = cropCanvas.getContext('2d');
                cropCanvas.width = cropW;
                cropCanvas.height = cropH;
                
                cropCtx.drawImage(img, cropX, cropY, cropW, cropH, 0, 0, cropW, cropH);
                
                // Send coordinates to Streamlit
                const selection = {{
                    x: Math.round(cropX),
                    y: Math.round(cropY),
                    width: Math.round(cropW),
                    height: Math.round(cropH),
                    dataURL: cropCanvas.toDataURL('image/png')
                }};
                
                window.parent.postMessage({{
                    type: 'selection',
                    data: selection
                }}, '*');
                
                status.innerText = 'Selection sent for OCR processing...';
            }}
        </script>
    </body>
    </html>
    """
    return canvas_html

# Initialize session state
if 'original_image' not in st.session_state:
    st.session_state.original_image = None
if 'enhanced_image' not in st.session_state:
    st.session_state.enhanced_image = None
if 'last_result' not in st.session_state:
    st.session_state.last_result = ""

# Main app
st.title("🔧 Enhanced Image OCR Extractor")

# Sidebar for controls
with st.sidebar:
    st.header("🎨 Image Enhancement")
    
    uploaded_file = st.file_uploader(
        "Upload Image", 
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        help="Upload an image for OCR processing"
    )
    
    enhancement_method = st.selectbox(
        "Enhancement Method",
        [
            "number_optimized", "measurement_enhanced", "digit_sharpening", 
            "auto_adaptive", "handwriting_optimized", "high_contrast",
            "noise_reduction", "edge_sharpening", "brightness_contrast",
            "histogram_equalization", "unsharp_masking", "morphological",
            "wiener_deconvolution"
        ],
        index=0,
        help="Choose the best enhancement method for your image type"
    )
    
    enhance_button = st.button("🚀 Generate Enhanced Image", type="primary")
    
    st.header("🔢 OCR Settings")
    
    ocr_mode = st.selectbox(
        "OCR Mode",
        [
            "numbers_precise", "measurements", "scientific_notation",
            "currency", "coordinates", "handwriting", "print", 
            "mixed", "numbers", "single_word"
        ],
        help="Choose OCR mode based on your text type"
    )
    
    language = st.selectbox(
        "Language",
        ["eng", "eng+ara", "eng+chi_sim", "eng+fra", "eng+deu", "eng+spa", "eng+rus"],
        help="Select OCR language"
    )

# Main content area
if uploaded_file is not None:
    # Load original image
    st.session_state.original_image = Image.open(uploaded_file)
    
    # Generate enhanced image
    if enhance_button:
        with st.spinner("🔄 Generating enhanced image..."):
            st.session_state.enhanced_image = advanced_image_enhancement(
                st.session_state.original_image, 
                enhancement_method
            )
        st.success("✅ Enhanced image generated!")
    
    # Display images side by side
    if st.session_state.original_image is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(st.session_state.original_image, use_column_width=True)
        
        with col2:
            st.subheader("Enhanced Image")
            if st.session_state.enhanced_image is not None:
                st.image(st.session_state.enhanced_image, use_column_width=True)
            else:
                st.info("Click 'Generate Enhanced Image' to see the enhanced version")
    
    # Interactive canvas for selection (only if enhanced image exists)
    if st.session_state.enhanced_image is not None:
        st.subheader("🎯 Select Text Areas for OCR")
        
        # Convert enhanced image to base64 for canvas
        enhanced_b64 = img_to_base64(st.session_state.enhanced_image)
        
        # Create canvas component
        canvas_html = create_canvas_component(enhanced_b64)
        
        # Display canvas
        canvas_result = components.html(canvas_html, height=700)
        
        # OCR Controls
        st.subheader("🔍 OCR Operations")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 Process Full Enhanced Image", type="secondary"):
                with st.spinner("🔄 Processing full enhanced image..."):
                    result = extract_text_from_image(
                        st.session_state.enhanced_image, 
                        ocr_mode, 
                        language
                    )
                    st.session_state.last_result = result
        
        with col2:
            if st.button("🔄 Multiple Attempts", type="secondary"):
                with st.spinner("🔄 Trying multiple approaches..."):
                    detailed_results, clean_results = multi_attempt_ocr(
                        st.session_state.enhanced_image, 
                        language
                    )
                    
                    st.subheader("🔢 Multiple Attempt Results")
                    for i, result in enumerate(detailed_results, 1):
                        st.text(f"Attempt {i}: {result}")
                    
                    if clean_results:
                        st.subheader("🎯 Clean Results")
                        for i, result in enumerate(clean_results, 1):
                            st.text(f"{i}. {result}")
                        
                        # Set best result as last result
                        best_result = max(clean_results, key=len) if clean_results else ""
                        st.session_state.last_result = best_result
        
        with col3:
            if st.session_state.last_result:
                if st.button("📋 Copy Result", type="secondary"):
                    # Note: Clipboard access requires HTTPS in browsers
                    st.code(st.session_state.last_result)
                    st.info("Result displayed above - copy manually")
        
        # Display last result
        if st.session_state.last_result:
            st.subheader("📋 OCR Result")
            st.text_area(
                "Extracted Text", 
                st.session_state.last_result, 
                height=150,
                help="The extracted text from OCR processing"
            )
        
        # Handle canvas selection (this would need JavaScript communication)
        # For now, we'll add a placeholder for manual crop coordinates
        st.subheader("📐 Manual Crop Selection (Alternative)")
        st.info("Use the canvas above for interactive selection, or manually specify crop coordinates below:")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            crop_x = st.number_input("X", min_value=0, value=0)
        with col2:
            crop_y = st.number_input("Y", min_value=0, value=0)
        with col3:
            crop_w = st.number_input("Width", min_value=1, value=100)
        with col4:
            crop_h = st.number_input("Height", min_value=1, value=100)
        
        if st.button("✍️ Extract from Manual Coordinates"):
            if st.session_state.enhanced_image is not None:
                # Crop the image
                img_array = np.array(st.session_state.enhanced_image)
                cropped_array = img_array[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
                
                if cropped_array.size > 0:
                    cropped_img = Image.fromarray(cropped_array)
                    
                    with st.spinner("🔄 Processing selected area..."):
                        result = extract_text_from_image(cropped_img, ocr_mode, language)
                        st.session_state.last_result = result
                    
                    # Display cropped area
                    st.image(cropped_img, caption="Cropped Area", width=300)
                else:
                    st.error("Invalid crop coordinates!")

else:
    # Welcome message and instructions
    st.info("📁 Upload an image to get started!")
    
    # Tips section
    with st.expander("💡 Tips for Better Results", expanded=True):
        st.markdown("""
        ### 🎯 NEW WORKFLOW:
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

        ### 🎨 OTHER ENHANCEMENT METHODS:
        - **Auto Adaptive Enhancement** - Best overall results
        - **Handwriting Optimized** - Specifically for handwritten text
        - **High Contrast Boost** - For faded or low-contrast images
        - **Advanced Noise Reduction** - For noisy/grainy images
        - **Edge Sharpening** - For blurry text
        - **Brightness & Contrast** - Basic adjustments
        - **Histogram Equalization** - Better light distribution
        - **Unsharp Masking** - Professional sharpening
        - **Morphological Enhancement** - Structure-based improvement
        - **Wiener Deconvolution** - Advanced blur removal

        ### 📖 BEST PRACTICES FOR NUMBERS:
        ✓ Use 'Number Recognition Optimized' enhancement for best number clarity  
        ✓ Choose the right OCR mode for your number type  
        ✓ Select tight crops around just the numbers you need  
        ✓ Try 'Multiple Attempts' - it tests all number recognition methods  
        ✓ Enhanced image provides significantly better number accuracy  
        ✓ Compare original vs enhanced to see digit improvement  
        """)

# JavaScript for handling canvas selection
components.html("""
<script>
window.addEventListener('message', function(event) {
    if (event.data.type === 'selection') {
        // Send selection data to Streamlit
        console.log('Selection received:', event.data.data);
        // You can use Streamlit's JavaScript API to send data back
    }
});
</script>
""", height=0)
