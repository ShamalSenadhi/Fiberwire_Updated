import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import io
import base64

# Set page config
st.set_page_config(
    page_title="Simple Image Text Extractor",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .enhancement-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .result-box {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        font-family: monospace;
        white-space: pre-wrap;
        max-height: 300px;
        overflow-y: auto;
    }
    .tips-box {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 10px 10px 0;
    }
    .warning-box {
        background-color: #fff8e1;
        border: 1px solid #ffcc02;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def simple_image_enhancement(img, method='contrast_enhance'):
    """Apply simple enhancement methods using only PIL"""
    
    if method == 'contrast_enhance':
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(img)
        enhanced = enhancer.enhance(2.0)
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(enhanced)
        enhanced = enhancer.enhance(1.5)
        
        return enhanced
        
    elif method == 'brightness_adjust':
        # Adjust brightness
        enhancer = ImageEnhance.Brightness(img)
        enhanced = enhancer.enhance(1.2)
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(enhanced)
        enhanced = enhancer.enhance(1.8)
        
        return enhanced
        
    elif method == 'sharpen_heavy':
        # Heavy sharpening
        enhancer = ImageEnhance.Sharpness(img)
        enhanced = enhancer.enhance(3.0)
        
        # Apply unsharp mask
        enhanced = enhanced.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
        
        return enhanced
        
    elif method == 'grayscale_enhance':
        # Convert to grayscale and enhance
        gray = img.convert('L')
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(gray)
        enhanced = enhancer.enhance(2.5)
        
        # Convert back to RGB
        return enhanced.convert('RGB')
        
    elif method == 'edge_enhance':
        # Edge enhancement
        enhanced = img.filter(ImageFilter.EDGE_ENHANCE_MORE)
        
        # Sharpen
        enhancer = ImageEnhance.Sharpness(enhanced)
        enhanced = enhancer.enhance(2.0)
        
        return enhanced
        
    elif method == 'smooth_sharpen':
        # Smooth then sharpen
        smoothed = img.filter(ImageFilter.SMOOTH)
        
        # Sharpen
        enhancer = ImageEnhance.Sharpness(smoothed)
        enhanced = enhancer.enhance(2.5)
        
        return enhanced
        
    elif method == 'high_contrast':
        # Maximum contrast
        enhancer = ImageEnhance.Contrast(img)
        enhanced = enhancer.enhance(3.0)
        
        # Adjust brightness
        enhancer = ImageEnhance.Brightness(enhanced)
        enhanced = enhancer.enhance(0.9)
        
        return enhanced
        
    elif method == 'detail_enhance':
        # Detail enhancement
        enhanced = img.filter(ImageFilter.DETAIL)
        
        # Sharpen
        enhancer = ImageEnhance.Sharpness(enhanced)
        enhanced = enhancer.enhance(1.8)
        
        return enhanced
        
    else:
        return img

def create_threshold_variations(img):
    """Create different threshold variations of the image"""
    variations = []
    
    # Convert to grayscale
    gray = img.convert('L')
    
    # Simple threshold variations
    thresholds = [128, 100, 150, 80, 180, 60, 200]
    
    for threshold in thresholds:
        # Create binary image
        binary = gray.point(lambda x: 255 if x > threshold else 0, mode='1')
        rgb_binary = binary.convert('RGB')
        variations.append((f"Threshold_{threshold}", rgb_binary))
    
    # Inverted versions
    inverted = ImageOps.invert(gray)
    variations.append(("Inverted", inverted.convert('RGB')))
    
    # Inverted threshold
    inv_binary = inverted.point(lambda x: 255 if x > 128 else 0, mode='1')
    variations.append(("Inverted_Threshold", inv_binary.convert('RGB')))
    
    return variations

def analyze_image_properties(img):
    """Analyze basic image properties"""
    # Convert to numpy array
    img_array = np.array(img)
    
    properties = {
        'size': img.size,
        'mode': img.mode,
        'mean_brightness': np.mean(img_array),
        'std_brightness': np.std(img_array),
        'min_value': np.min(img_array),
        'max_value': np.max(img_array)
    }
    
    return properties

def mock_ocr_analysis(img):
    """Mock OCR analysis - provides image analysis instead of actual OCR"""
    properties = analyze_image_properties(img)
    
    analysis = f"""
IMAGE ANALYSIS RESULTS:
======================

Image Properties:
- Size: {properties['size'][0]} x {properties['size'][1]} pixels
- Color Mode: {properties['mode']}
- Average Brightness: {properties['mean_brightness']:.2f}
- Brightness Variation: {properties['std_brightness']:.2f}
- Value Range: {properties['min_value']} - {properties['max_value']}

Recommendations:
"""
    
    if properties['mean_brightness'] < 100:
        analysis += "- Image appears dark - try 'brightness_adjust' enhancement\n"
    elif properties['mean_brightness'] > 200:
        analysis += "- Image appears bright - try 'high_contrast' enhancement\n"
    else:
        analysis += "- Image has good brightness - try 'contrast_enhance'\n"
        
    if properties['std_brightness'] < 30:
        analysis += "- Low contrast detected - use 'high_contrast' method\n"
    else:
        analysis += "- Good contrast detected - try 'sharpen_heavy' method\n"
        
    analysis += "\nNote: This is image analysis only. For actual text extraction, "
    analysis += "pytesseract needs to be installed."
    
    return analysis

# Main Streamlit App
def main():
    st.markdown('<h1 class="main-header">🔍 Simple Image Text Extractor</h1>', unsafe_allow_html=True)
    
    # Warning about OCR
    st.markdown('<div class="warning-box">', unsafe_allow_html=True)
    st.warning("⚠️ **OCR Functionality Limited**: This app provides image enhancement and analysis. For actual text extraction, pytesseract needs to be installed.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.header("🎨 Configuration")
    
    # Enhancement method selection
    enhancement_method = st.sidebar.selectbox(
        "Enhancement Method",
        [
            "contrast_enhance", "brightness_adjust", "sharpen_heavy", 
            "grayscale_enhance", "edge_enhance", "smooth_sharpen",
            "high_contrast", "detail_enhance"
        ],
        index=0,
        help="Choose the best enhancement method for your image type"
    )
    
    # Additional options
    st.sidebar.subheader("📋 Options")
    show_variations = st.sidebar.checkbox("Show Threshold Variations", value=False)
    show_analysis = st.sidebar.checkbox("Show Image Analysis", value=True)
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload an image", 
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        help="Upload an image containing text or numbers"
    )
    
    if uploaded_file is not None:
        # Load original image
        original_img = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Original Image")
            st.image(original_img, caption="Original Image", use_column_width=True)
            
            if show_analysis:
                properties = analyze_image_properties(original_img)
                st.subheader("📊 Original Image Properties")
                st.write(f"**Size:** {properties['size'][0]} x {properties['size'][1]} pixels")
                st.write(f"**Mode:** {properties['mode']}")
                st.write(f"**Brightness:** {properties['mean_brightness']:.1f} (0-255)")
                st.write(f"**Contrast:** {properties['std_brightness']:.1f}")
        
        # Enhancement step
        st.markdown('<div class="enhancement-box">', unsafe_allow_html=True)
        st.subheader("🎨 Step 1: Image Enhancement")
        
        if st.button("🚀 Generate Enhanced Image", type="primary"):
            with st.spinner("Enhancing image..."):
                try:
                    enhanced_img = simple_image_enhancement(original_img, enhancement_method)
                    st.session_state.enhanced_img = enhanced_img
                    st.session_state.enhancement_method = enhancement_method
                    st.success(f"✅ Image enhanced using: {enhancement_method}")
                        
                except Exception as e:
                    st.error(f"Enhancement error: {str(e)}")
                    st.session_state.enhanced_img = original_img
                    st.warning("Using original image as fallback")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Show enhanced image if available
        if 'enhanced_img' in st.session_state:
            with col2:
                st.subheader("✨ Enhanced Image")
                st.image(st.session_state.enhanced_img, caption=f"Enhanced ({st.session_state.enhancement_method})", use_column_width=True)
                
                if show_analysis:
                    properties = analyze_image_properties(st.session_state.enhanced_img)
                    st.subheader("📊 Enhanced Image Properties")
                    st.write(f"**Brightness:** {properties['mean_brightness']:.1f}")
                    st.write(f"**Contrast:** {properties['std_brightness']:.1f}")
            
            # Analysis section
            st.subheader("🔍 Step 2: Image Analysis")
            
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                if st.button("📊 Analyze Enhanced Image", type="secondary"):
                    with st.spinner("Analyzing image..."):
                        analysis = mock_ocr_analysis(st.session_state.enhanced_img)
                        st.session_state.analysis_result = analysis
            
            with col_b:
                if st.button("🖼️ Generate Variations", type="secondary") and show_variations:
                    with st.spinner("Creating variations..."):
                        variations = create_threshold_variations(st.session_state.enhanced_img)
                        st.session_state.variations = variations
            
            with col_c:
                if st.button("🗑️ Clear Results", type="secondary"):
                    for key in ['analysis_result', 'variations']:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.success("Results cleared!")
            
            # Display analysis results
            if 'analysis_result' in st.session_state:
                st.subheader("📋 Image Analysis")
                st.markdown(f'<div class="result-box">{st.session_state.analysis_result}</div>', unsafe_allow_html=True)
            
            # Display variations
            if 'variations' in st.session_state and show_variations:
                st.subheader("🖼️ Threshold Variations")
                
                # Show variations in a grid
                cols = st.columns(3)
                for i, (name, var_img) in enumerate(st.session_state.variations):
                    with cols[i % 3]:
                        st.image(var_img, caption=name, width=200)
    
    # Tips and information
    st.markdown('<div class="tips-box">', unsafe_allow_html=True)
    st.subheader("💡 Usage Tips")
    
    st.markdown("""
    **This Simplified Version Provides:**
    - ✅ Image enhancement using PIL
    - ✅ Image property analysis
    - ✅ Threshold variations for text extraction preparation
    - ✅ Recommendations for optimal settings
    
    **For Full OCR Functionality:**
    1. Install pytesseract: `pip install pytesseract`
    2. Install Tesseract OCR engine on your system
    3. Use the full OCR version of this app
    
    **Enhancement Methods:**
    - **contrast_enhance**: General purpose enhancement
    - **brightness_adjust**: For dark images
    - **sharpen_heavy**: For blurry text
    - **grayscale_enhance**: For colored backgrounds
    - **high_contrast**: For faded text
    - **edge_enhance**: For outline text
    
    **How to Setup Full OCR:**
    ```bash
    # Install Python packages
    pip install pytesseract pillow numpy scipy scikit-image
    
    # Install Tesseract OCR (Ubuntu/Debian)
    sudo apt-get install tesseract-ocr
    
    # Install Tesseract OCR (Windows)
    # Download from: https://github.com/UB-Mannheim/tesseract/wiki
    
    # Install Tesseract OCR (Mac)
    brew install tesseract
    ```
    """)
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
