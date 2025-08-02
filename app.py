import streamlit as st
import easyocr
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import io
import base64
from scipy import ndimage
from skimage import morphology, exposure, restoration, filters
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import RectangleSelector
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure Streamlit page
st.set_page_config(
    page_title="🔧 Enhanced OCR Extractor",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize EasyOCR reader (cached for performance)
@st.cache_resource
def load_ocr_reader(languages=['en']):
    """Initialize EasyOCR reader with specified languages"""
    return easyocr.Reader(languages, gpu=False)

def advanced_image_enhancement(img, method='auto_adaptive'):
    """Apply advanced enhancement methods to generate improved image"""
    # Convert PIL to OpenCV format
    if isinstance(img, Image.Image):
        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    else:
        cv_img = img
    
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
        # For measurements like "12.51m", "3.4kg", etc.
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4,4))
        enhanced = clahe.apply(denoised)
        kernel_sharp = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
        sharpened = cv2.filter2D(enhanced, -1, kernel_sharp)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        result = cv2.morphologyEx(sharpened, cv2.MORPH_OPEN, kernel)

    elif method == 'digit_sharpening':
        # For digital/printed numbers with maximum sharpness
        denoised = cv2.fastNlMeansDenoising(gray, h=8)
        gaussian = cv2.GaussianBlur(denoised, (0, 0), 1.5)
        unsharp = cv2.addWeighted(denoised, 2.0, gaussian, -1.0, 0)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(unsharp)
        laplacian = cv2.Laplacian(enhanced, cv2.CV_64F, ksize=3)
        laplacian = np.uint8(np.absolute(laplacian))
        result = cv2.addWeighted(enhanced, 0.9, laplacian, 0.1, 0)

    elif method == 'auto_adaptive':
        # Comprehensive adaptive enhancement
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        smoothed = cv2.bilateralFilter(enhanced, 9, 75, 75)
        gaussian = cv2.GaussianBlur(smoothed, (0, 0), 2.0)
        result = cv2.addWeighted(smoothed, 1.5, gaussian, -0.5, 0)

    elif method == 'handwriting_optimized':
        # Specifically for handwriting
        denoised = cv2.fastNlMeansDenoising(gray, h=8)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(6,6))
        enhanced = clahe.apply(denoised)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        closed = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
        kernel_sharp = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        result = cv2.filter2D(closed, -1, kernel_sharp)

    elif method == 'high_contrast':
        # Maximum contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        gamma = 0.8
        lookup_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        result = cv2.LUT(enhanced, lookup_table)

    elif method == 'noise_reduction':
        # Advanced noise reduction
        denoised1 = cv2.fastNlMeansDenoising(gray, h=10)
        denoised2 = cv2.bilateralFilter(denoised1, 9, 80, 80)
        result = cv2.medianBlur(denoised2, 3)

    elif method == 'edge_sharpening':
        # Edge enhancement and sharpening
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = np.uint8(np.absolute(laplacian))
        result = cv2.addWeighted(gray, 0.8, laplacian, 0.2, 0)

    elif method == 'brightness_contrast':
        alpha = 1.3  # Contrast control
        beta = 20    # Brightness control
        result = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

    elif method == 'histogram_equalization':
        result = cv2.equalizeHist(gray)

    elif method == 'unsharp_masking':
        gaussian = cv2.GaussianBlur(gray, (0, 0), 2.0)
        result = cv2.addWeighted(gray, 1.8, gaussian, -0.8, 0)

    elif method == 'morphological':
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        result = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    elif method == 'wiener_deconvolution':
        # Wiener deconvolution for blur removal
        try:
            from skimage import restoration
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

def get_easyocr_settings(mode, languages):
    """Get EasyOCR settings based on mode"""
    settings = {
        'allowlist': None,
        'blocklist': None,
        'width_ths': 0.7,
        'height_ths': 0.7,
        'paragraph': False
    }
    
    if mode in ['numbers_precise', 'measurements', 'scientific_notation', 'currency', 'coordinates']:
        settings['allowlist'] = '0123456789.,-+()[]{}mkcglbftinMKCGLBFTIN°%$€£¥₹eE×x*'
        settings['paragraph'] = False
        
    elif mode == 'handwriting':
        settings['paragraph'] = True
        
    return settings

def crop_image(image, x1, y1, x2, y2):
    """Crop image based on coordinates"""
    # Ensure coordinates are within image bounds
    height, width = image.shape[:2] if len(image.shape) == 3 else (image.size[1], image.size[0])
    
    x1, x2 = max(0, min(x1, x2)), min(width, max(x1, x2))
    y1, y2 = max(0, min(y1, y2)), min(height, max(y1, y2))
    
    if isinstance(image, Image.Image):
        return image.crop((x1, y1, x2, y2))
    else:
        return image[y1:y2, x1:x2]

def perform_ocr(image, reader, mode='auto', languages=['en']):
    """Perform OCR using EasyOCR"""
    try:
        # Convert PIL to numpy array if needed
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
            
        # Get settings for the mode
        settings = get_easyocr_settings(mode, languages)
        
        # Perform OCR
        results = reader.readtext(img_array, **settings)
        
        # Extract text
        extracted_text = []
        for (bbox, text, confidence) in results:
            if confidence > 0.3:  # Filter low confidence results
                extracted_text.append(text)
                
        return ' '.join(extracted_text), results
        
    except Exception as e:
        st.error(f"OCR Error: {str(e)}")
        return "", []

def main():
    st.title("🔧 Enhanced Image OCR Extractor")
    st.markdown("**Powered by EasyOCR with Advanced Image Enhancement**")
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎨 Enhancement Settings")
        
        enhancement_method = st.selectbox(
            "Choose Enhancement Method",
            options=[
                'number_optimized', 'measurement_enhanced', 'digit_sharpening',
                'auto_adaptive', 'handwriting_optimized', 'high_contrast',
                'noise_reduction', 'edge_sharpening', 'brightness_contrast',
                'histogram_equalization', 'unsharp_masking', 'morphological',
                'wiener_deconvolution'
            ],
            format_func=lambda x: {
                'number_optimized': '🔢 Number Recognition Optimized',
                'measurement_enhanced': '📏 Measurement Text Enhanced',
                'digit_sharpening': '🎯 Digital/Printed Numbers',
                'auto_adaptive': '🤖 Auto Adaptive Enhancement',
                'handwriting_optimized': '✍️ Handwriting Optimized',
                'high_contrast': '⚡ High Contrast Boost',
                'noise_reduction': '🧹 Advanced Noise Reduction',
                'edge_sharpening': '📐 Edge Sharpening',
                'brightness_contrast': '💡 Brightness & Contrast',
                'histogram_equalization': '📊 Histogram Equalization',
                'unsharp_masking': '🔍 Unsharp Masking',
                'morphological': '🔄 Morphological Enhancement',
                'wiener_deconvolution': '🌟 Wiener Deconvolution'
            }[x],
            index=0
        )
        
        st.header("🔍 OCR Settings")
        
        ocr_mode = st.selectbox(
            "OCR Mode",
            options=[
                'numbers_precise', 'measurements', 'scientific_notation',
                'currency', 'coordinates', 'handwriting', 'print', 'mixed'
            ],
            format_func=lambda x: {
                'numbers_precise': '🔢 Precise Number Recognition',
                'measurements': '📏 Measurements (12.51m, 3.4kg, etc.)',
                'scientific_notation': '🧪 Scientific Numbers',
                'currency': '💰 Currency & Financial',
                'coordinates': '🗺️ Coordinates & GPS',
                'handwriting': '📝 Handwriting Optimized',
                'print': '🖨️ Printed Text',
                'mixed': '🔀 Mixed Text'
            }[x],
            index=0
        )
        
        languages = st.multiselect(
            "Languages",
            options=['en', 'fr', 'de', 'es', 'zh', 'ja', 'ko', 'ar', 'hi', 'ru'],
            default=['en'],
            help="Select languages for OCR recognition"
        )
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        help="Upload an image file for OCR processing"
    )
    
    if uploaded_file is not None:
        # Load and display original image
        original_image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Original Image")
            st.image(original_image, use_column_width=True)
            st.info(f"Size: {original_image.size[0]}×{original_image.size[1]} pixels")
        
        # Generate enhanced image
        if st.button("🚀 Generate Enhanced Image", type="primary"):
            with st.spinner("Enhancing image..."):
                enhanced_image = advanced_image_enhancement(original_image, enhancement_method)
                st.session_state.enhanced_image = enhanced_image
                st.session_state.original_image = original_image
        
        # Display enhanced image if available
        if 'enhanced_image' in st.session_state:
            with col2:
                st.subheader("✨ Enhanced Image")
                st.image(st.session_state.enhanced_image, use_column_width=True)
                st.success(f"Enhanced using: {enhancement_method}")
            
            # OCR Processing section
            st.header("🔍 OCR Processing")
            
            # Initialize OCR reader
            with st.spinner("Loading OCR model..."):
                reader = load_ocr_reader(languages)
            
            # Create tabs for different processing options
            tab1, tab2, tab3 = st.tabs(["📄 Full Image OCR", "✂️ Crop & Extract", "🔄 Multiple Attempts"])
            
            with tab1:
                st.subheader("Process Full Enhanced Image")
                if st.button("🔍 Extract Text from Full Image"):
                    with st.spinner("Performing OCR on full image..."):
                        extracted_text, results = perform_ocr(
                            st.session_state.enhanced_image, 
                            reader, 
                            ocr_mode, 
                            languages
                        )
                        
                        if extracted_text:
                            st.success("✅ Text Extracted Successfully!")
                            st.text_area("Extracted Text:", extracted_text, height=150)
                            
                            # Display confidence scores
                            if results:
                                st.subheader("📊 Detection Results")
                                for i, (bbox, text, confidence) in enumerate(results):
                                    st.write(f"**Text {i+1}:** {text} (Confidence: {confidence:.2f})")
                        else:
                            st.warning("No text detected in the image.")
            
            with tab2:
                st.subheader("Crop Selection for Targeted OCR")
                st.info("Use the sliders below to define a crop area, then extract text from that region.")
                
                # Get image dimensions
                img_width, img_height = st.session_state.enhanced_image.size
                
                # Crop controls
                col_x1, col_y1, col_x2, col_y2 = st.columns(4)
                
                with col_x1:
                    x1 = st.slider("Left (X1)", 0, img_width, 0)
                with col_y1:
                    y1 = st.slider("Top (Y1)", 0, img_height, 0)
                with col_x2:
                    x2 = st.slider("Right (X2)", 0, img_width, img_width)
                with col_y2:
                    y2 = st.slider("Bottom (Y2)", 0, img_height, img_height)
                
                # Show crop preview
                if x2 > x1 and y2 > y1:
                    cropped_image = crop_image(st.session_state.enhanced_image, x1, y1, x2, y2)
                    st.image(cropped_image, caption=f"Crop Preview ({x2-x1}×{y2-y1} pixels)")
                    
                    if st.button("✍️ Extract Text from Selection"):
                        with st.spinner("Performing OCR on selected area..."):
                            extracted_text, results = perform_ocr(
                                cropped_image, 
                                reader, 
                                ocr_mode, 
                                languages
                            )
                            
                            if extracted_text:
                                st.success("✅ Text Extracted from Selection!")
                                st.text_area("Extracted Text:", extracted_text, height=100)
                            else:
                                st.warning("No text detected in the selected area.")
            
            with tab3:
                st.subheader("Multiple OCR Attempts")
                st.info("Try different enhancement and OCR combinations for better results.")
                
                if st.button("🔄 Run Multiple Attempts"):
                    methods = ['number_optimized', 'auto_adaptive', 'high_contrast', 'handwriting_optimized']
                    modes = ['numbers_precise', 'measurements', 'handwriting', 'mixed']
                    
                    results_data = []
                    
                    progress_bar = st.progress(0)
                    total_attempts = len(methods) * len(modes)
                    attempt = 0
                    
                    for method in methods:
                        enhanced_img = advanced_image_enhancement(original_image, method)
                        
                        for mode in modes:
                            attempt += 1
                            progress_bar.progress(attempt / total_attempts)
                            
                            try:
                                text, ocr_results = perform_ocr(enhanced_img, reader, mode, languages)
                                if text.strip():
                                    results_data.append({
                                        'Enhancement': method,
                                        'OCR Mode': mode,
                                        'Text': text.strip(),
                                        'Length': len(text.strip())
                                    })
                            except:
                                continue
                    
                    progress_bar.empty()
                    
                    if results_data:
                        st.success(f"✅ Found {len(results_data)} valid results!")
                        
                        # Display results
                        for i, result in enumerate(results_data, 1):
                            with st.expander(f"Result {i}: {result['Enhancement']} + {result['OCR Mode']} ({result['Length']} chars)"):
                                st.code(result['Text'])
                        
                        # Find best result (longest text)
                        best_result = max(results_data, key=lambda x: x['Length'])
                        st.subheader("🎯 Best Result:")
                        st.success(f"**Method:** {best_result['Enhancement']} + {best_result['OCR Mode']}")
                        st.text_area("Best Extracted Text:", best_result['Text'], height=100)
                    else:
                        st.warning("No text detected with any combination.")
    
    # Tips and information
    with st.expander("💡 Tips for Better Results"):
        st.markdown("""
        ### 🎯 Best Practices for Number Recognition:
        - **Pure Numbers:** Use "Number Recognition Optimized" enhancement + "Precise Number Recognition" OCR
        - **Measurements:** Use "Measurement Text Enhanced" + "Measurements" mode for "12.51m", "3.4kg", etc.
        - **Digital Displays:** Use "Digital/Printed Numbers" enhancement for LCD/LED numbers
        - **Scientific Numbers:** Use "Scientific Numbers" mode for "1.5e-3", "2×10⁵", etc.
        
        ### 🎨 Enhancement Method Guide:
        - **Auto Adaptive:** Best overall results for mixed content
        - **Handwriting Optimized:** Specifically for handwritten text
        - **High Contrast Boost:** For faded or low-contrast images
        - **Noise Reduction:** For noisy/grainy images
        - **Edge Sharpening:** For blurry text
        
        ### 🔍 OCR Tips:
        - Try cropping to focus on specific text areas
        - Use multiple attempts to test different combinations
        - Select appropriate languages for better recognition
        - Compare original vs enhanced images to see improvement
        """)

if __name__ == "__main__":
    main()
