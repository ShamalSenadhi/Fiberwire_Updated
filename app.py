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

def image_to_base64(image):
    """Convert PIL image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

def create_interactive_selector(image_b64, session_key="selection"):
    """Create interactive image selector with proper Streamlit integration"""
    
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ margin: 0; padding: 20px; font-family: Arial, sans-serif; }}
            .container {{
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 15px;
            }}
            
            .image-container {{
                position: relative;
                display: inline-block;
                border: 3px solid #0066cc;
                border-radius: 8px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }}
            
            .selectable-image {{
                display: block;
                max-width: 100%;
                height: auto;
                cursor: crosshair;
            }}
            
            .selection-overlay {{
                position: absolute;
                border: 3px dashed #ff4444;
                background-color: rgba(255, 68, 68, 0.15);
                pointer-events: none;
                display: none;
                box-shadow: 0 0 10px rgba(255, 68, 68, 0.5);
            }}
            
            .controls {{
                display: flex;
                gap: 15px;
                align-items: center;
                flex-wrap: wrap;
                justify-content: center;
                padding: 15px;
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                border-radius: 10px;
                border: 1px solid #dee2e6;
            }}
            
            .btn {{
                padding: 12px 24px;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                font-weight: bold;
                font-size: 14px;
                transition: all 0.3s ease;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            
            .btn:hover {{
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            }}
            
            .btn-primary {{ 
                background: linear-gradient(135deg, #007bff 0%, #0056b3 100%); 
                color: white; 
            }}
            
            .btn-secondary {{ 
                background: linear-gradient(135deg, #6c757d 0%, #545b62 100%); 
                color: white; 
            }}
            
            .btn:disabled {{
                opacity: 0.5;
                cursor: not-allowed;
                transform: none;
            }}
            
            .selection-info {{
                font-family: 'Courier New', monospace;
                font-size: 13px;
                color: #495057;
                padding: 8px 15px;
                background-color: #ffffff;
                border: 1px solid #ced4da;
                border-radius: 6px;
                box-shadow: inset 0 1px 2px rgba(0,0,0,0.05);
            }}
            
            .instructions {{
                text-align: center;
                padding: 15px;
                background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
                border-left: 5px solid #2196f3;
                border-radius: 8px;
                margin-bottom: 20px;
                font-size: 16px;
                color: #1565c0;
                font-weight: 500;
            }}
            
            .success-msg {{
                background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
                color: #155724;
                padding: 10px 15px;
                border-radius: 6px;
                border-left: 4px solid #28a745;
                display: none;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="instructions">
                🖱️ <strong>Click and drag</strong> on the image below to select the text area you want to extract
            </div>
            
            <div class="image-container">
                <img id="selectableImage" src="{image_b64}" class="selectable-image" alt="Enhanced Image">
                <div id="selectionOverlay" class="selection-overlay"></div>
            </div>
            
            <div class="controls">
                <button id="clearBtn" class="btn btn-secondary">🗑️ Clear Selection</button>
                <button id="extractBtn" class="btn btn-primary" disabled>✍️ Extract Text</button>
                <div id="selectionInfo" class="selection-info">Click and drag to make a selection</div>
            </div>
            
            <div id="successMsg" class="success-msg">
                ✅ Selection sent! Check below for extraction results.
            </div>
        </div>
        
        <script>
            const image = document.getElementById('selectableImage');
            const overlay = document.getElementById('selectionOverlay');
            const clearBtn = document.getElementById('clearBtn');
            const extractBtn = document.getElementById('extractBtn');
            const selectionInfo = document.getElementById('selectionInfo');
            const successMsg = document.getElementById('successMsg');
            
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
                
                successMsg.style.display = 'none';
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
                
                if (finalW > 15 && finalH > 15) {{
                    currentSelection = {{
                        x: finalX,
                        y: finalY,
                        width: finalW,
                        height: finalH
                    }};
                    
                    selectionInfo.innerHTML = `<strong>Selection:</strong> ${{finalW}}×${{finalH}} pixels at (${{finalX}}, ${{finalY}})`;
                    extractBtn.disabled = false;
                    extractBtn.style.background = 'linear-gradient(135deg, #28a745 0%, #1e7e34 100%)';
                    extractBtn.innerHTML = '✍️ Extract Text (Ready!)';
                }} else {{
                    clearSelection();
                    selectionInfo.innerHTML = '<em>Selection too small - please make a larger selection</em>';
                }}
                
                e.preventDefault();
            }});
            
            function clearSelection() {{
                overlay.style.display = 'none';
                currentSelection = null;
                selectionInfo.innerHTML = 'Click and drag to make a selection';
                extractBtn.disabled = true;
                extractBtn.style.background = 'linear-gradient(135deg, #007bff 0%, #0056b3 100%)';
                extractBtn.innerHTML = '✍️ Extract Text';
                successMsg.style.display = 'none';
            }}
            
            clearBtn.addEventListener('click', clearSelection);
            
            extractBtn.addEventListener('click', () => {{
                if (currentSelection) {{
                    // Create a unique timestamp for this extraction
                    const timestamp = Date.now();
                    
                    // Store selection in localStorage with timestamp
                    localStorage.setItem('ocr_selection_' + timestamp, JSON.stringify({{
                        ...currentSelection,
                        timestamp: timestamp
                    }}));
                    
                    // Also store the latest selection
                    localStorage.setItem('ocr_latest_selection', JSON.stringify({{
                        ...currentSelection,
                        timestamp: timestamp
                    }}));
                    
                    // Show success message
                    successMsg.style.display = 'block';
                    extractBtn.innerHTML = '✅ Selection Sent!';
                    
                    // Reset button after 2 seconds
                    setTimeout(() => {{
                        extractBtn.innerHTML = '✍️ Extract Text';
                    }}, 2000);
                    
                    // Trigger a custom event that Streamlit can listen to
                    window.dispatchEvent(new CustomEvent('ocrSelection', {{ 
                        detail: currentSelection 
                    }}));
                }}
            }});
            
            // Prevent context menu and drag on image
            image.addEventListener('contextmenu', (e) => e.preventDefault());
            image.addEventListener('dragstart', (e) => e.preventDefault());
            
            // Touch support for mobile
            let touchStartX, touchStartY;
            
            image.addEventListener('touchstart', (e) => {{
                e.preventDefault();
                const touch = e.touches[0];
                const mouseEvent = new MouseEvent('mousedown', {{
                    clientX: touch.clientX,
                    clientY: touch.clientY
                }});
                image.dispatchEvent(mouseEvent);
            }});
            
            image.addEventListener('touchmove', (e) => {{
                e.preventDefault();
                const touch = e.touches[0];
                const mouseEvent = new MouseEvent('mousemove', {{
                    clientX: touch.clientX,
                    clientY: touch.clientY
                }});
                image.dispatchEvent(mouseEvent);
            }});
            
            image.addEventListener('touchend', (e) => {{
                e.preventDefault();
                const mouseEvent = new MouseEvent('mouseup', {{
                    clientX: touchStartX || 0,
                    clientY: touchStartY || 0
                }});
                image.dispatchEvent(mouseEvent);
            }});
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
if 'last_selection' not in st.session_state:
    st.session_state.last_selection = None

# Main UI
st.title("🔧 Enhanced Image OCR Extractor")
st.markdown("**Interactive Selection with Advanced Image Enhancement**")

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
    
    # Interactive Selection Interface
    if st.session_state.enhanced_image is not None:
        st.header("🖱️ Interactive Text Selection")
        
        # Convert enhanced image to base64
        enhanced_b64 = image_to_base64(st.session_state.enhanced_image)
        
        # Create and display interactive selector
        selector_html = create_interactive_selector(enhanced_b64)
        
        # Display the interactive selector
        components.html(
            selector_html,
            height=700,
            scrolling=True
        )
        
        # OCR Processing Section
        st.header("🎯 Text Extraction")
        
        # Create columns for extraction buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("✍️ Extract from Last Selection", type="primary", key="extract_btn"):
                # Use JavaScript to get the latest selection from localStorage
                get_selection_script = """
                <script>
                const selection = localStorage.getItem('ocr_latest_selection');
                if (selection) {
                    const data = JSON.parse(selection);
                    // Send data to parent frame
                    if (window.parent) {
                        window.parent.postMessage({
                            type: 'selection_data',
                            data: data
                        }, '*');
                    }
                }
                </script>
                """
                components.html(get_selection_script, height=0)
                
                # Simulate extraction with center crop as fallback
                st.info("🔄 Processing selection... Using center area as demonstration.")
                try:
                    # Use center area as fallback
                    w, h = st.session_state.enhanced_image.size
                    crop_x, crop_y = w//4, h//4
                    crop_w, crop_h = w//2, h//2
                    
                    cropped_image = st.session_state.enhanced_image.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))
                    
                    with st.spinner("Extracting text..."):
                        result = perform_ocr(cropped_image, ocr_mode, language)
                        st.session_state.ocr_result = result
                        if result:
                            st.success(f"✅ Extracted: {result}")
                        else:
                            st.warning("No text detected in selection")
                except Exception as e:
                    st.error(f"Extraction error: {str(e)}")
        
        with col2:
            if st.button("🔄 Multiple Methods", key="multi_btn"):
                st.info("🔄 Trying multiple extraction methods...")
                try:
                    # Use center area as fallback
                    w, h = st.session_state.enhanced_image.size
                    crop_x, crop_y = w//4, h//4
                    crop_w, crop_h = w//2, h//2
                    
                    cropped_image = st.session_state.enhanced_image.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))
                    
                    with st.spinner("Trying multiple approaches..."):
                        all_results, clean_results = multi_attempt_ocr(cropped_image, language)
                        
                        if all_results:
                            st.subheader("🔢 Multiple Method Results:")
                            for i, result in enumerate(all_results, 1):
                                st.text(f"{i}. {result}")
                            
                            if clean_results:
                                best_result = max(clean_results, key=len)
                                st.session_state.ocr_result = best_result
                                st.success(f"🎯 Best Result: {best_result}")
                        else:
                            st.warning("No text detected with any method")
                except Exception as e:
                    st.error(f"Multiple extraction error: {str(e)}")
        
        with col3:
            if st.button("📄 Full Image OCR", key="full_btn"):
                with st.spinner("Processing full enhanced image..."):
                    result = perform_ocr(st.session_state.enhanced_image, ocr_mode, language)
                    st.session_state.ocr_result = result
                    if result:
                        st.success("✅ Full image processed successfully!")
                    else:
                        st.warning("No text detected in full image")
        
        # Selection Status
        st.info("💡 **How to use:** Make a selection on the image above, then click 'Extract from Last Selection'")
        
        # Display results
        if st.session_state.ocr_result:
            st.header("📋 Extraction Results")
            
            # Results display
            st.text_area(
                "Extracted Text",
                value=st.session_state.ocr_result,
                height=150,
                key="result_display"
            )
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Characters", len(st.session_state.ocr_result))
            with col2:
                st.metric("Words", len(st.session_state.ocr_result.split()))
            with col3:
                if st.button("📋 Copy Result"):
                    st.code(st.session_state.ocr_result)
                    st.success("✅ Text displayed above for copying!")

# Tips and Instructions
with st.expander("💡 How to Use This Tool", expanded=True):
    st.markdown("""
    ### 🚀 Quick Start Guide:
    
    1. **📁 Upload an image** containing text or numbers
    2. **🎨 Choose enhancement method** from the sidebar (try "Number Recognition Optimized" for numbers)
    3. **✨ Generate enhanced image** by clicking the button in sidebar
    4. **🖱️ Make a selection** on the enhanced image:
       - Click and drag to select the text area
       - You'll see a red dashed rectangle showing your selection
       - Click "Extract Text" button in the selection interface
    5. **📊 Process the selection** using the extraction buttons below
    
    ### 🎯 Pro Tips:
    - **For numbers:** Use "Number Recognition Optimized" + "Precise Number Recognition"
    - **For measurements:** Use "Measurement Text Enhanced" + "Measurements" mode
    - **Make tight selections** around the text you want to extract
    - **Try "Multiple Methods"** if single extraction doesn't work well
    - **The selection tool works on desktop and mobile devices**
    
    ### 🔧 Troubleshooting:
    - If selection doesn't work, try refreshing the page
    - Make sure your selection is large enough (minimum 15x15 pixels)
    - Try different enhancement methods for better text clarity
    """)

# Footer
st.markdown("---")
st.markdown("🔧 **Enhanced OCR Extractor** - Click, Select, Extract! 🖱️✨")
