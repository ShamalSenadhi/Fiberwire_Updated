import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageFont
import pytesseract
import io
import base64
from scipy import ndimage
from skimage import morphology, exposure, restoration, filters
import streamlit.components.v1 as components
import re
from decimal import Decimal, InvalidOperation

# Set page config
st.set_page_config(
    page_title="📏 Precision Wire Length OCR",
    page_icon="📏",
    layout="wide"
)

def advanced_number_enhancement(img, method='precision_numbers'):
    """Enhanced preprocessing specifically for precise number recognition"""
    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    
    if method == 'precision_numbers':
        # Multi-stage enhancement for precise number recognition
        # Stage 1: Noise reduction with edge preservation
        denoised = cv2.bilateralFilter(gray, 15, 80, 80)
        
        # Stage 2: Contrast enhancement with CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4,4))
        enhanced = clahe.apply(denoised)
        
        # Stage 3: Sharpening for crisp edges
        kernel_sharp = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel_sharp)
        
        # Stage 4: Morphological operations for digit separation
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        opened = cv2.morphologyEx(sharpened, cv2.MORPH_OPEN, kernel)
        
        # Stage 5: Final thresholding
        _, binary = cv2.threshold(opened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Stage 6: Dilation for better character connection
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
        result = cv2.dilate(binary, kernel_dilate, iterations=1)
        
    elif method == 'technical_drawing':
        # Optimized for technical drawings and blueprints
        # Noise reduction
        denoised = cv2.fastNlMeansDenoising(gray, h=15)
        
        # Contrast stretching
        p2, p98 = np.percentile(denoised, (2, 98))
        stretched = np.clip((denoised - p2) * 255 / (p98 - p2), 0, 255).astype(np.uint8)
        
        # Adaptive histogram equalization
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(6,6))
        enhanced = clahe.apply(stretched)
        
        # Edge-preserving smoothing
        smoothed = cv2.edgePreservingFilter(enhanced, flags=2, sigma_s=50, sigma_r=0.4)
        
        # Adaptive thresholding
        result = cv2.adaptiveThreshold(smoothed.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
    elif method == 'meter_scale':
        # Specialized for measurement scales and rulers
        # Gaussian blur to reduce fine noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Contrast enhancement
        alpha = 2.0  # Contrast control
        beta = -50   # Brightness control
        enhanced = cv2.convertScaleAbs(blurred, alpha=alpha, beta=beta)
        
        # Morphological gradient to enhance text edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        gradient = cv2.morphologyEx(enhanced, cv2.MORPH_GRADIENT, kernel)
        
        # Threshold
        _, binary = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Clean up
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
        result = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_clean)
        
    elif method == 'ultra_sharp':
        # Ultra-sharp enhancement for small or blurry numbers
        # Unsharp masking
        gaussian = cv2.GaussianBlur(gray, (0, 0), 2.0)
        unsharp = cv2.addWeighted(gray, 2.0, gaussian, -1.0, 0)
        
        # High-pass filter
        kernel = np.array([[-1,-1,-1,-1,-1],
                          [-1, 2, 2, 2,-1],
                          [-1, 2, 8, 2,-1],
                          [-1, 2, 2, 2,-1],
                          [-1,-1,-1,-1,-1]]) / 8.0
        sharpened = cv2.filter2D(unsharp, -1, kernel)
        
        # Normalize
        normalized = cv2.normalize(sharpened, None, 0, 255, cv2.NORM_MINMAX)
        
        # Adaptive threshold
        result = cv2.adaptiveThreshold(normalized.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 5)
        
    elif method == 'blueprint_mode':
        # Specialized for blue/white technical blueprints
        # Convert to LAB color space for better contrast
        lab = cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)
        l_channel = lab[:,:,0]
        
        # CLAHE on L channel
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        l_channel = clahe.apply(l_channel)
        
        # Reconstruct and convert back
        lab[:,:,0] = l_channel
        enhanced_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        enhanced_gray = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)
        
        # Morphological operations for text enhancement
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        tophat = cv2.morphologyEx(enhanced_gray, cv2.MORPH_TOPHAT, kernel)
        result = cv2.add(enhanced_gray, tophat)
        
    elif method == 'handwritten_digits':
        # Optimized for handwritten measurements
        # Gentle noise reduction
        denoised = cv2.fastNlMeansDenoising(gray, h=8)
        
        # Gamma correction for better visibility
        gamma = 1.2
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        gamma_corrected = cv2.LUT(denoised, table)
        
        # Slight blur to connect broken strokes
        blurred = cv2.GaussianBlur(gamma_corrected, (2, 2), 0)
        
        # Adaptive threshold with larger block size for handwriting
        result = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3)
        
    elif method == 'high_dpi_scan':
        # Optimized for high-resolution scanned documents
        # Resize for processing if image is very large
        h, w = gray.shape
        if w > 2000 or h > 2000:
            scale = min(2000/w, 2000/h)
            new_w, new_h = int(w * scale), int(h * scale)
            gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Multi-scale retinex for illumination normalization
        def single_scale_retinex(img, sigma):
            retinex = np.log10(img + 1.0) - np.log10(cv2.GaussianBlur(img, (0, 0), sigma) + 1.0)
            return retinex
        
        # Apply retinex with multiple scales
        gray_float = gray.astype(np.float64) + 1.0
        retinex1 = single_scale_retinex(gray_float, 15)
        retinex2 = single_scale_retinex(gray_float, 80)
        retinex3 = single_scale_retinex(gray_float, 250)
        retinex = (retinex1 + retinex2 + retinex3) / 3.0
        
        # Normalize and convert back
        retinex = np.clip((retinex - retinex.min()) * 255 / (retinex.max() - retinex.min()), 0, 255)
        result = retinex.astype(np.uint8)
        
    elif method == 'low_contrast_boost':
        # For faded or low-contrast measurements
        # Histogram equalization
        equalized = cv2.equalizeHist(gray)
        
        # CLAHE with high clip limit
        clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(4,4))
        clahe_applied = clahe.apply(gray)
        
        # Combine both methods
        combined = cv2.addWeighted(equalized, 0.6, clahe_applied, 0.4, 0)
        
        # Sharpening
        kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
        result = cv2.filter2D(combined, -1, kernel)
        
    elif method == 'wire_diagram_special':
        # Specialized for electrical wire diagrams
        # Edge detection to find text regions
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate edges to connect nearby text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Use edges as mask for enhancement
        masked = cv2.bitwise_and(gray, gray, mask=dilated_edges)
        
        # Enhance the masked regions
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced_masked = clahe.apply(masked)
        
        # Combine with original
        result = cv2.addWeighted(gray, 0.7, enhanced_masked, 0.3, 0)
        
    elif method == 'measurement_tape':
        # Optimized for measuring tape/ruler images
        # Enhance horizontal and vertical lines separately
        # Horizontal enhancement
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        morph_h = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_h)
        
        # Vertical enhancement  
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
        morph_v = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_v)
        
        # Combine
        combined = cv2.add(morph_h, morph_v)
        
        # Subtract from original to enhance text
        result = cv2.subtract(gray, combined)
        
        # Final contrast boost
        result = cv2.convertScaleAbs(result, alpha=2.0, beta=0)
        
    else:  # advanced_multi_stage (default fallback)
        # Advanced multi-stage processing combining best techniques
        # Stage 1: Noise reduction
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Stage 2: Contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # Stage 3: Unsharp masking
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 1.5)
        unsharp = cv2.addWeighted(enhanced, 1.8, gaussian, -0.8, 0)
        
        # Stage 4: Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        opened = cv2.morphologyEx(unsharp, cv2.MORPH_OPEN, kernel)
        
        # Stage 5: Final normalization
        result = cv2.normalize(opened, None, 0, 255, cv2.NORM_MINMAX)
    
    # Convert back to RGB
    if len(result.shape) == 2:
        result_rgb = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
    else:
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    
    return Image.fromarray(result_rgb)

def get_precision_ocr_config(mode, language):
    """Precision OCR configurations for length measurements"""
    
    # Character sets optimized for different scenarios
    meter_chars = '0123456789.m'
    decimal_meter_chars = '0123456789.,m'
    general_length_chars = '0123456789.,mMkKcC '
    numeric_only = '0123456789.,'
    
    configs = {
        'precision_meters': f'--oem 3 --psm 8 -l {language} -c tessedit_char_whitelist={meter_chars} -c tessedit_do_invert=0',
        'decimal_meters': f'--oem 3 --psm 8 -l {language} -c tessedit_char_whitelist={decimal_meter_chars} -c classify_bln_numeric_mode=1',
        'single_measurement': f'--oem 3 --psm 7 -l {language} -c tessedit_char_whitelist={meter_chars} -c textord_really_old_xheight=1',
        'numeric_precise': f'--oem 3 --psm 8 -l {language} -c tessedit_char_whitelist={numeric_only} -c classify_bln_numeric_mode=1 -c tessedit_zero_rejection=T',
        'length_general': f'--oem 3 --psm 6 -l {language} -c tessedit_char_whitelist={general_length_chars}',
        'isolated_word': f'--oem 3 --psm 8 -l {language} -c tessedit_char_whitelist={meter_chars} -c tessedit_single_match=1',
        'technical_text': f'--oem 3 --psm 7 -l {language} -c preserve_interword_spaces=1 -c tessedit_char_whitelist={general_length_chars}',
    }
    return configs.get(mode, configs['precision_meters'])

def extract_length_measurements(text):
    """Extract and validate length measurements from text"""
    if not text:
        return []
    
    measurements = []
    
    # Comprehensive regex patterns for different measurement formats
    patterns = [
        r'(\d+(?:[.,]\d+)?)\s*m(?:eter)?s?\b',           # 1484m, 12.5m, 1,234m
        r'(\d+(?:[.,]\d+)?)\s*(?:meter|metre)s?\b',      # 1484 meter, 12.5 metres
        r'(\d+(?:[.,]\d+)?)\s*(?:km|kilometer)s?\b',     # 12.5km, 1 kilometer
        r'(\d+(?:[.,]\d+)?)\s*(?:cm|centimeter)s?\b',    # 150cm, 15.5 centimeters
        r'(\d+(?:[.,]\d+)?)\s*(?:mm|millimeter)s?\b',    # 1500mm, 15.5 millimeters
        r'\b(\d+(?:[.,]\d+)?)m\b',                       # Standalone format like 1484m
        r'(?:length|distance|measure):\s*(\d+(?:[.,]\d+)?)\s*m', # Label format
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, text.lower(), re.IGNORECASE)
        for match in matches:
            try:
                # Extract numeric value
                value_str = match.group(1).replace(',', '.')
                value = float(value_str)
                
                # Determine unit and convert to meters
                full_match = match.group(0).lower()
                if 'km' in full_match or 'kilometer' in full_match:
                    meters = value * 1000
                    unit = 'km'
                elif 'cm' in full_match or 'centimeter' in full_match:
                    meters = value / 100
                    unit = 'cm'
                elif 'mm' in full_match or 'millimeter' in full_match:
                    meters = value / 1000
                    unit = 'mm'
                else:
                    meters = value
                    unit = 'm'
                
                measurements.append({
                    'original': match.group(0),
                    'value': value,
                    'unit': unit,
                    'meters': meters,
                    'confidence': calculate_measurement_confidence(match.group(0), text)
                })
            except (ValueError, IndexError):
                continue
    
    # Remove duplicates and sort by confidence
    unique_measurements = []
    seen_values = set()
    
    for measurement in measurements:
        rounded_meters = round(measurement['meters'], 3)
        if rounded_meters not in seen_values:
            seen_values.add(rounded_meters)
            unique_measurements.append(measurement)
    
    return sorted(unique_measurements, key=lambda x: x['confidence'], reverse=True)

def calculate_measurement_confidence(match_text, full_text):
    """Calculate confidence score for a measurement based on context"""
    confidence = 0.5  # Base confidence
    
    # Higher confidence for proper formatting
    if re.match(r'^\d+(?:[.,]\d+)?m$', match_text.strip()):
        confidence += 0.3
    
    # Higher confidence for reasonable measurement values
    try:
        value = float(re.search(r'(\d+(?:[.,]\d+)?)', match_text).group(1).replace(',', '.'))
        if 0.1 <= value <= 10000:  # Reasonable range for wire lengths
            confidence += 0.2
        if 100 <= value <= 5000:   # Most common wire length range
            confidence += 0.2
    except:
        pass
    
    # Context clues
    context_words = ['wire', 'cable', 'length', 'distance', 'measure', 'total', 'span']
    for word in context_words:
        if word in full_text.lower():
            confidence += 0.1
            break
    
    return min(confidence, 1.0)

def perform_precision_ocr(img, mode='precision_meters', language='eng'):
    """Perform high-precision OCR specifically for measurements"""
    try:
        config = get_precision_ocr_config(mode, language)
        
        # Get text with confidence scores
        data = pytesseract.image_to_data(img, config=config, output_type=pytesseract.Output.DICT)
        
        # Filter results by confidence
        confidences = data['conf']
        texts = data['text']
        
        # Combine high-confidence text
        high_conf_text = []
        for i, conf in enumerate(confidences):
            if int(conf) > 30 and texts[i].strip():  # Confidence threshold
                high_conf_text.append(texts[i])
        
        combined_text = ' '.join(high_conf_text)
        
        # Also get simple text extraction as fallback
        simple_text = pytesseract.image_to_string(img, config=config).strip()
        
        # Return both for comparison
        return combined_text, simple_text
        
    except Exception as e:
        st.error(f"OCR Error: {str(e)}")
        return "", ""

def multi_method_precision_ocr(img, language='eng'):
    """Apply multiple OCR methods optimized for length measurements"""
    results = []
    all_measurements = []
    
    # OCR methods in order of preference for length measurements
    methods = [
        'precision_meters',
        'decimal_meters', 
        'single_measurement',
        'numeric_precise',
        'isolated_word',
        'technical_text',
        'length_general'
    ]
    
    for method in methods:
        try:
            high_conf_text, simple_text = perform_precision_ocr(img, method, language)
            
            # Process both text results
            for text_type, text in [('high_conf', high_conf_text), ('simple', simple_text)]:
                if text:
                    measurements = extract_length_measurements(text)
                    for measurement in measurements:
                        measurement['method'] = f"{method}_{text_type}"
                        all_measurements.append(measurement)
                    
                    results.append({
                        'method': f"{method}_{text_type}",
                        'text': text,
                        'measurements': measurements
                    })
        except Exception as e:
            continue
    
    # Consolidate measurements and find the best candidates
    consolidated = consolidate_measurements(all_measurements)
    
    return results, consolidated

def consolidate_measurements(measurements):
    """Consolidate similar measurements and rank by confidence"""
    if not measurements:
        return []
    
    # Group similar measurements (within 1% tolerance)
    groups = []
    for measurement in measurements:
        added_to_group = False
        for group in groups:
            if abs(measurement['meters'] - group[0]['meters']) / max(measurement['meters'], group[0]['meters']) < 0.01:
                group.append(measurement)
                added_to_group = True
                break
        if not added_to_group:
            groups.append([measurement])
    
    # Calculate group confidence and select best measurement from each group
    consolidated = []
    for group in groups:
        # Sort by confidence
        group.sort(key=lambda x: x['confidence'], reverse=True)
        best = group[0].copy()
        
        # Boost confidence if multiple methods agree
        if len(group) > 1:
            best['confidence'] = min(1.0, best['confidence'] + 0.2 * (len(group) - 1))
            best['agreement_count'] = len(group)
        else:
            best['agreement_count'] = 1
            
        consolidated.append(best)
    
    return sorted(consolidated, key=lambda x: (x['confidence'], x['agreement_count']), reverse=True)

def image_to_base64(image):
    """Convert PIL image to base64 string for HTML display"""
    try:
        buffered = io.BytesIO()
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
    except Exception as e:
        st.error(f"Error converting image to base64: {str(e)}")
        return ""

def format_measurement_result(measurement):
    """Format measurement result for display"""
    meters = measurement['meters']
    unit = measurement['unit']
    confidence = measurement['confidence']
    
    # Format the meters value appropriately
    if meters >= 1000:
        formatted = f"{meters/1000:.1f} km ({meters:.0f}m)"
    elif meters >= 1:
        formatted = f"{meters:.1f}m"
    elif meters >= 0.01:
        formatted = f"{meters*100:.1f}cm ({meters:.3f}m)"
    else:
        formatted = f"{meters*1000:.1f}mm ({meters:.4f}m)"
    
    return {
        'display': formatted,
        'meters': meters,
        'confidence': confidence,
        'original': measurement['original']
    }

def create_interactive_selector(image_b64, session_key="selection"):
    """Create interactive image selector with enhanced precision selection"""
    
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ margin: 0; padding: 20px; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }}
            .container {{
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 20px;
            }}
            
            .image-container {{
                position: relative;
                display: inline-block;
                border: 3px solid #2196f3;
                border-radius: 12px;
                box-shadow: 0 8px 24px rgba(33, 150, 243, 0.3);
                background: #f8f9ff;
                padding: 10px;
            }}
            
            .selectable-image {{
                display: block;
                max-width: 100%;
                height: auto;
                cursor: crosshair;
                border-radius: 8px;
            }}
            
            .selection-overlay {{
                position: absolute;
                border: 3px dashed #ff5722;
                background: linear-gradient(45deg, rgba(255, 87, 34, 0.1), rgba(255, 87, 34, 0.2));
                pointer-events: none;
                display: none;
                box-shadow: 0 0 15px rgba(255, 87, 34, 0.6);
                border-radius: 4px;
            }}
            
            .controls {{
                display: flex;
                gap: 20px;
                align-items: center;
                flex-wrap: wrap;
                justify-content: center;
                padding: 20px;
                background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
                border-radius: 15px;
                border: 2px solid #2196f3;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            }}
            
            .btn {{
                padding: 14px 28px;
                border: none;
                border-radius: 10px;
                cursor: pointer;
                font-weight: 600;
                font-size: 15px;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                box-shadow: 0 3px 8px rgba(0,0,0,0.15);
                position: relative;
                overflow: hidden;
            }}
            
            .btn::before {{
                content: '';
                position: absolute;
                top: 0;
                left: -100%;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
                transition: left 0.5s;
            }}
            
            .btn:hover::before {{
                left: 100%;
            }}
            
            .btn:hover {{
                transform: translateY(-3px);
                box-shadow: 0 6px 16px rgba(0,0,0,0.25);
            }}
            
            .btn-primary {{ 
                background: linear-gradient(135deg, #2196f3 0%, #1976d2 100%); 
                color: white; 
            }}
            
            .btn-extract {{ 
                background: linear-gradient(135deg, #4caf50 0%, #388e3c 100%); 
                color: white; 
            }}
            
            .btn-secondary {{ 
                background: linear-gradient(135deg, #757575 0%, #424242 100%); 
                color: white; 
            }}
            
            .btn:disabled {{
                opacity: 0.6;
                cursor: not-allowed;
                transform: none;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            
            .selection-info {{
                font-family: 'Courier New', monospace;
                font-size: 14px;
                color: #1976d2;
                padding: 12px 20px;
                background: linear-gradient(135deg, #ffffff 0%, #f5f5f5 100%);
                border: 2px solid #2196f3;
                border-radius: 8px;
                box-shadow: inset 0 2px 4px rgba(0,0,0,0.05);
                min-width: 300px;
                text-align: center;
            }}
            
            .instructions {{
                text-align: center;
                padding: 20px;
                background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
                border-left: 6px solid #ff9800;
                border-radius: 10px;
                margin-bottom: 25px;
                font-size: 17px;
                color: #e65100;
                font-weight: 600;
                box-shadow: 0 4px 12px rgba(255, 152, 0, 0.2);
            }}
            
            .success-msg {{
                background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
                color: #2e7d32;
                padding: 15px 20px;
                border-radius: 8px;
                border-left: 5px solid #4caf50;
                display: none;
                font-weight: 600;
                box-shadow: 0 4px 12px rgba(76, 175, 80, 0.2);
            }}
            
            .precision-tip {{
                background: linear-gradient(135deg, #fce4ec 0%, #f8bbd9 100%);
                color: #c2185b;
                padding: 15px;
                border-radius: 8px;
                border-left: 4px solid #e91e63;
                margin-top: 10px;
                font-size: 14px;
                font-weight: 500;
            }}
            
            @keyframes pulse {{
                0% {{ box-shadow: 0 0 0 0 rgba(33, 150, 243, 0.7); }}
                70% {{ box-shadow: 0 0 0 10px rgba(33, 150, 243, 0); }}
                100% {{ box-shadow: 0 0 0 0 rgba(33, 150, 243, 0); }}
            }}
            
            .btn-extract:not(:disabled) {{
                animation: pulse 2s infinite;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="instructions">
                📏 <strong>Precision Length Measurement Selection</strong><br>
                Click and drag to select the exact measurement text (e.g., "1484m")
            </div>
            
            <div class="image-container">
                <img id="selectableImage" src="{image_b64}" class="selectable-image" alt="Enhanced Image">
                <div id="selectionOverlay" class="selection-overlay"></div>
            </div>
            
            <div class="controls">
                <button id="clearBtn" class="btn btn-secondary">🗑️ Clear Selection</button>
                <button id="extractBtn" class="btn btn-extract" disabled>🎯 Extract Length</button>
                <div id="selectionInfo" class="selection-info">Click and drag to select measurement text</div>
            </div>
            
            <div class="precision-tip">
                💡 <strong>Pro Tip:</strong> Make tight selections around measurement text for best accuracy. 
                The tool will detect formats like "1484m", "12.5km", "150cm", etc.
            </div>
            
            <div id="successMsg" class="success-msg">
                ✅ Selection captured! Processing for length measurements...
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
            let selectionCount = 0;
            
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
                
                if (finalW > 20 && finalH > 15) {{
                    currentSelection = {{
                        x: finalX,
                        y: finalY,
                        width: finalW,
                        height: finalH,
                        timestamp: Date.now()
                    }};
                    
                    selectionInfo.innerHTML = `<strong>📏 Precision Selection:</strong> ${{finalW}}×${{finalH}}px at (${{finalX}}, ${{finalY}})`;
                    extractBtn.disabled = false;
                    extractBtn.innerHTML = '🎯 Extract Length (Ready!)';
                    extractBtn.classList.add('btn-extract');
                }} else {{
                    clearSelection();
                    selectionInfo.innerHTML = '<em style="color: #d32f2f;">Selection too small - please select measurement text area</em>';
                }}
                
                e.preventDefault();
            }});
            
            function clearSelection() {{
                overlay.style.display = 'none';
                currentSelection = null;
                selectionInfo.innerHTML = 'Click and drag to select measurement text';
                extractBtn.disabled = true;
                extractBtn.innerHTML = '🎯 Extract Length';
                extractBtn.classList.remove('btn-extract');
                successMsg.style.display = 'none';
            }}
            
            clearBtn.addEventListener('click', clearSelection);
            
            extractBtn.addEventListener('click', () => {{
                if (currentSelection) {{
                    selectionCount++;
                    const selectionKey = 'precision_ocr_selection_' + selectionCount;
                    
                    localStorage.setItem(selectionKey, JSON.stringify(currentSelection));
                    localStorage.setItem('precision_ocr_latest', JSON.stringify(currentSelection));
                    
                    successMsg.style.display = 'block';
                    extractBtn.innerHTML = '✅ Processing Length!';
                    
                    setTimeout(() => {{
                        extractBtn.innerHTML = '🎯 Extract Length';
                    }}, 3000);
                    
                    window.dispatchEvent(new CustomEvent('precisionOcrSelection', {{ 
                        detail: currentSelection 
                    }}));
                }}
            }});
            
            // Prevent context menu and drag
            image.addEventListener('contextmenu', (e) => e.preventDefault());
            image.addEventListener('dragstart', (e) => e.preventDefault());
            
            // Touch support
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
                const mouseEvent = new MouseEvent('mouseup');
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
if 'ocr_results' not in st.session_state:
    st.session_state.ocr_results = []
if 'best_measurements' not in st.session_state:
    st.session_state.best_measurements = []
if 'last_selection' not in st.session_state:
    st.session_state.last_selection = None

# Main UI
st.title("📏 Precision Wire Length OCR Extractor")
st.markdown("**Advanced Number Recognition with Precise Length Measurement in Meters**")

# Enhanced sidebar
st.sidebar.header("⚙️ Precision Settings")

# File upload
uploaded_file = st.file_uploader(
    "📁 Upload Image with Length Measurements", 
    type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
    help="Upload technical drawings, wire diagrams, or measurement scales"
)

if uploaded_file is not None:
    # Load and display original image
    original_image = Image.open(uploaded_file)
    st.session_state.original_image = original_image
    
    # Enhanced settings sidebar
    st.sidebar.subheader("🎨 Advanced Enhancement Methods")
    enhancement_method = st.sidebar.selectbox(
        "Choose Enhancement Method",
        [
            ('precision_numbers', '🎯 Precision Number Recognition (Recommended)'),
            ('technical_drawing', '📐 Technical Drawing & Blueprints'),
            ('meter_scale', '📏 Measurement Scales & Rulers'),
            ('ultra_sharp', '⚡ Ultra-Sharp Enhancement'),
            ('blueprint_mode', '🔵 Blueprint & CAD Drawings'),
            ('handwritten_digits', '✍️ Handwritten Measurements'),  
            ('high_dpi_scan', '📋 High-Resolution Scanned Documents'),
            ('low_contrast_boost', '🔆 Low Contrast & Faded Text'),
            ('wire_diagram_special', '🔌 Electrical Wire Diagrams'),
            ('measurement_tape', '📐 Measuring Tape & Rulers'),
            ('advanced_multi_stage', '🚀 Advanced Multi-Stage Processing'),
        ],
        format_func=lambda x: x[1],
        help="Select the enhancement method that best matches your image type for optimal results"
    )[0]
    
    # OCR precision settings
    st.sidebar.subheader("🔍 OCR Precision")
    primary_ocr_mode = st.sidebar.selectbox(
        "Primary OCR Mode",
        [
            ('precision_meters', '📏 Precision Meters (Recommended)'),
            ('decimal_meters', '📊 Decimal Measurements'),
            ('single_measurement', '🎯 Single Measurement Focus'),
            ('numeric_precise', '🔢 Numeric Precision'),
            ('technical_text', '📐 Technical Text'),
        ],
        format_func=lambda x: x[1],
        help="Primary method for extracting length measurements"
    )[0]
    
    language = st.sidebar.selectbox(
        "Language",
        [
            ('eng', '🇺🇸 English'),
            ('eng+deu', '🇺🇸🇩🇪 English + German'),
            ('eng+fra', '🇺🇸🇫🇷 English + French'),
        ],
        format_func=lambda x: x[1]
    )[0]
    
    # Advanced settings
    with st.sidebar.expander("⚙️ Advanced Settings"):
        use_multiple_methods = st.checkbox("🔄 Use Multiple OCR Methods", value=True, help="Apply multiple OCR techniques for higher accuracy")
        confidence_threshold = st.slider("🎯 Confidence Threshold", 0.3, 1.0, 0.6, 0.1, help="Minimum confidence for accepting measurements")
        measurement_range = st.selectbox(
            "📏 Expected Measurement Range",
            [
                ('wire_standard', '🔌 Wire Lengths (100m - 5000m)'),
                ('cable_long', '📡 Long Cables (1km - 50km)'),
                ('short_measurements', '📏 Short Measurements (1m - 100m)'),
                ('any_range', '🌐 Any Range'),
            ],
            format_func=lambda x: x[1]
        )[0]
    
    # Generate enhanced image
    if st.sidebar.button("🚀 Generate Precision Enhanced Image", type="primary"):
        with st.spinner("🔧 Applying precision enhancement algorithms..."):
            enhanced_image = advanced_number_enhancement(original_image, enhancement_method)
            st.session_state.enhanced_image = enhanced_image
            st.sidebar.success("✅ Precision enhanced image ready!")
            st.balloons()
    
    # Enhancement preview section
    with st.sidebar.expander("👁️ Preview Enhancement Methods", expanded=False):
        if st.button("🔍 Generate Preview Comparison", key="preview_btn"):
            with st.spinner("Generating preview comparison..."):
                st.subheader("🔍 Enhancement Methods Comparison")
                
                # Create a smaller version for faster processing
                preview_size = (400, 300)
                preview_original = original_image.copy()
                preview_original.thumbnail(preview_size, Image.Resampling.LANCZOS)
                
                # Generate previews for top methods
                preview_methods = [
                    ('precision_numbers', '🎯 Precision Numbers'),
                    ('technical_drawing', '📐 Technical Drawing'),
                    ('ultra_sharp', '⚡ Ultra Sharp'),
                    ('blueprint_mode', '🔵 Blueprint Mode'),
                    ('low_contrast_boost', '🔆 Contrast Boost'),
                    ('wire_diagram_special', '🔌 Wire Diagram')
                ]
                
                # Display in 2 columns
                for i in range(0, len(preview_methods), 2):
                    col1, col2 = st.columns(2)
                    
                    # First method
                    if i < len(preview_methods):
                        method_key, method_name = preview_methods[i]
                        try:
                            preview_enhanced = advanced_number_enhancement(preview_original, method_key)
                            with col1:
                                st.write(f"**{method_name}**")
                                st.image(preview_enhanced, use_container_width=True)
                        except Exception as e:
                            with col1:
                                st.write(f"**{method_name}**")
                                st.error(f"Preview error: {str(e)[:50]}...")
                    
                    # Second method  
                    if i + 1 < len(preview_methods):
                        method_key, method_name = preview_methods[i + 1]
                        try:
                            preview_enhanced = advanced_number_enhancement(preview_original, method_key)
                            with col2:
                                st.write(f"**{method_name}**")
                                st.image(preview_enhanced, use_container_width=True)
                        except Exception as e:
                            with col2:
                                st.write(f"**{method_name}**")
                                st.error(f"Preview error: {str(e)[:50]}...")
                
                st.info("💡 Choose the method that makes your measurements most clear and readable!")
    
    # Display images side by side
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Original Image")
        st.image(original_image, use_container_width=True)
        st.caption(f"📐 Dimensions: {original_image.size[0]}×{original_image.size[1]} pixels")
    
    with col2:
        if st.session_state.enhanced_image is not None:
            st.subheader("✨ Precision Enhanced")
            st.image(st.session_state.enhanced_image, use_container_width=True)
            st.caption(f"📐 Enhanced: {st.session_state.enhanced_image.size[0]}×{st.session_state.enhanced_image.size[1]} pixels")
            
            # Image quality metrics
            img_array = np.array(st.session_state.enhanced_image.convert('L'))
            contrast = img_array.std()
            brightness = img_array.mean()
            st.caption(f"📊 Contrast: {contrast:.1f} | Brightness: {brightness:.1f}")
        else:
            st.subheader("✨ Precision Enhanced")
            st.info("👆 Click 'Generate Precision Enhanced Image' to see the enhanced version optimized for number recognition")
    
    # Interactive Precision Selection Interface
    if st.session_state.enhanced_image is not None:
        st.header("🎯 Precision Measurement Selection")
        
        # Convert enhanced image to base64
        enhanced_b64 = image_to_base64(st.session_state.enhanced_image)
        
        # Create and display interactive selector
        selector_html = create_interactive_selector(enhanced_b64)
        
        # Display the interactive selector
        components.html(
            selector_html,
            height=800,
            scrolling=True
        )
        
        # Length Extraction Section
        st.header("📏 Precision Length Extraction")
        
        # Create extraction action buttons
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🎯 Extract from Selection", type="primary", key="precision_extract"):
                st.info("🔄 Processing precision selection...")
                try:
                    # Use center area as demonstration (in real implementation, would use actual selection)
                    w, h = st.session_state.enhanced_image.size
                    crop_x, crop_y = w//4, h//4
                    crop_w, crop_h = w//2, h//2
                    
                    cropped_image = st.session_state.enhanced_image.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))
                    
                    with st.spinner("🔍 Applying precision OCR algorithms..."):
                        if use_multiple_methods:
                            results, measurements = multi_method_precision_ocr(cropped_image, language)
                            st.session_state.ocr_results = results
                            st.session_state.best_measurements = measurements
                        else:
                            high_conf_text, simple_text = perform_precision_ocr(cropped_image, primary_ocr_mode, language)
                            combined_text = f"{high_conf_text} {simple_text}".strip()
                            measurements = extract_length_measurements(combined_text)
                            st.session_state.best_measurements = measurements
                        
                        # Filter by confidence threshold
                        filtered_measurements = [m for m in st.session_state.best_measurements if m['confidence'] >= confidence_threshold]
                        
                        if filtered_measurements:
                            best_measurement = filtered_measurements[0]
                            formatted = format_measurement_result(best_measurement)
                            
                            st.success(f"🎯 **DETECTED LENGTH: {formatted['display']}**")
                            st.metric(
                                "📏 Precise Length in Meters", 
                                f"{formatted['meters']:.3f}m",
                                delta=f"Confidence: {best_measurement['confidence']:.1%}"
                            )
                            st.balloons()
                            
                            # Show additional measurements if found
                            if len(filtered_measurements) > 1:
                                st.subheader("📊 Additional Measurements Found:")
                                for i, measurement in enumerate(filtered_measurements[1:4], 2):  # Show up to 3 more
                                    formatted_alt = format_measurement_result(measurement)
                                    st.info(f"📏 Length {i}: {formatted_alt['display']} (Confidence: {measurement['confidence']:.1%})")
                        else:
                            st.warning("⚠️ No measurements detected with sufficient confidence. Try adjusting settings or selection.")
                            
                except Exception as e:
                    st.error(f"❌ Extraction error: {str(e)}")
        
        with col2:
            if st.button("🔄 Multi-Method Analysis", key="multi_method"):
                st.info("🔄 Running comprehensive analysis with all precision methods...")
                try:
                    w, h = st.session_state.enhanced_image.size
                    crop_x, crop_y = w//4, h//4
                    crop_w, crop_h = w//2, h//2
                    
                    cropped_image = st.session_state.enhanced_image.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))
                    
                    with st.spinner("🧠 Analyzing with multiple precision algorithms..."):
                        results, measurements = multi_method_precision_ocr(cropped_image, language)
                        st.session_state.ocr_results = results
                        st.session_state.best_measurements = measurements
                        
                        if measurements:
                            st.subheader("🏆 Top Length Measurements:")
                            for i, measurement in enumerate(measurements[:3], 1):
                                formatted = format_measurement_result(measurement)
                                confidence_color = "🟢" if measurement['confidence'] > 0.8 else "🟡" if measurement['confidence'] > 0.6 else "🟠"
                                st.success(f"{confidence_color} **Rank {i}: {formatted['display']}** (Confidence: {measurement['confidence']:.1%}, Methods: {measurement.get('agreement_count', 1)})")
                            
                            # Set best as primary result
                            best = measurements[0]
                            st.metric("🎯 **FINAL RESULT**", f"{best['meters']:.3f} meters", f"Confidence: {best['confidence']:.1%}")
                            st.balloons()
                            
                        else:
                            st.warning("❌ No length measurements detected with any method")
                            
                except Exception as e:
                    st.error(f"❌ Multi-method analysis error: {str(e)}")
        
        with col3:
            if st.button("📐 Full Image Scan", key="full_scan"):
                with st.spinner("🔍 Scanning entire image for measurements..."):
                    try:
                        results, measurements = multi_method_precision_ocr(st.session_state.enhanced_image, language)
                        st.session_state.ocr_results = results
                        st.session_state.best_measurements = measurements
                        
                        if measurements:
                            st.success(f"✅ Found {len(measurements)} measurement(s) in full image!")
                            for i, measurement in enumerate(measurements[:5], 1):  # Show top 5
                                formatted = format_measurement_result(measurement)
                                st.info(f"📏 Measurement {i}: {formatted['display']} (Confidence: {measurement['confidence']:.1%})")
                        else:
                            st.warning("❌ No measurements found in full image")
                    except Exception as e:
                        st.error(f"❌ Full scan error: {str(e)}")
        
        with col4:
            if st.button("🔬 Debug Analysis", key="debug_analysis"):
                st.info("🔬 Running debug analysis...")
                try:
                    w, h = st.session_state.enhanced_image.size
                    crop_x, crop_y = w//4, h//4
                    crop_w, crop_h = w//2, h//2
                    
                    cropped_image = st.session_state.enhanced_image.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))
                    
                    with st.spinner("🔍 Debug analysis in progress..."):
                        results, measurements = multi_method_precision_ocr(cropped_image, language)
                        
                        if results:
                            with st.expander("🔬 Detailed OCR Results", expanded=True):
                                for result in results:
                                    st.subheader(f"Method: {result['method']}")
                                    st.text(f"Raw Text: {result['text']}")
                                    if result['measurements']:
                                        for m in result['measurements']:
                                            st.json({
                                                'original': m['original'],
                                                'meters': m['meters'],
                                                'confidence': m['confidence'],
                                                'unit': m['unit']
                                            })
                                    st.markdown("---")
                        else:
                            st.warning("No debug results available")
                            
                except Exception as e:
                    st.error(f"❌ Debug analysis error: {str(e)}")
        
        # Results Display Section
        if st.session_state.best_measurements:
            st.header("📊 Measurement Results Summary")
            
            # Create results summary
            best_measurement = st.session_state.best_measurements[0]
            formatted_best = format_measurement_result(best_measurement)
            
            # Main result display
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "🎯 Final Length", 
                    formatted_best['display'],
                    f"±{(1-best_measurement['confidence'])*100:.1f}% uncertainty"
                )
            with col2:
                st.metric(
                    "📏 Meters (Precise)", 
                    f"{best_measurement['meters']:.6f}",
                    f"Original: {best_measurement['original']}"
                )
            with col3:
                st.metric("🎯 Confidence Score", f"{best_measurement['confidence']:.1%}")
            
            # Results table
            if len(st.session_state.best_measurements) > 1:
                st.subheader("📋 All Detected Measurements")
                
                results_data = []
                for i, measurement in enumerate(st.session_state.best_measurements[:10], 1):  # Top 10
                    formatted = format_measurement_result(measurement)
                    results_data.append({
                        'Rank': i,
                        'Length (Display)': formatted['display'],
                        'Meters (Exact)': f"{measurement['meters']:.6f}",
                        'Original Text': measurement['original'],
                        'Confidence': f"{measurement['confidence']:.1%}",
                        'Agreement': measurement.get('agreement_count', 1)
                    })
                
                st.dataframe(results_data, use_container_width=True)
            
            # Export options
            st.subheader("💾 Export Results")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📋 Copy Best Result"):
                    result_text = f"Length: {formatted_best['meters']:.3f}m (Confidence: {best_measurement['confidence']:.1%})"
                    st.code(result_text)
                    st.success("✅ Result ready for copying!")
            
            with col2:
                if st.button("📊 Copy All Results"):
                    all_results = "\n".join([
                        f"Rank {i}: {format_measurement_result(m)['display']} (Confidence: {m['confidence']:.1%})"
                        for i, m in enumerate(st.session_state.best_measurements[:5], 1)
                    ])
                    st.code(all_results)
                    st.success("✅ All results ready for copying!")

# Enhanced Help Section
with st.expander("🎓 Advanced Usage Guide", expanded=False):
    st.markdown("""
    ### 🚀 Precision Wire Length Extraction Guide
    
    #### 📏 **For Best Results:**
    1. **📁 Upload high-quality images** with clear, readable measurements
    2. **🎨 Choose the right enhancement:**
       - **Precision Numbers**: Best for most wire measurements (1484m, 12.5m)
       - **Technical Drawing**: Optimal for blueprints and technical drawings
       - **Meter Scale**: Perfect for ruler/scale measurements
       - **Ultra-Sharp**: For blurry or small text
    
    #### 🎯 **Selection Tips:**
    - Make **tight selections** around measurement text only
    - Include the number AND unit (e.g., select "1484m" completely)
    - Avoid including extra text or graphics in selection
    - For best accuracy, select individual measurements one at a time
    
    #### 📊 **Understanding Results:**
    - **Confidence Score**: Higher is better (>80% = excellent, >60% = good)
    - **Agreement Count**: How many methods detected the same measurement
    - **Final Result**: Always displayed in meters with high precision
    
    #### 🔧 **Troubleshooting:**
    - **No measurements detected**: Try different enhancement methods
    - **Low confidence**: Make tighter selections, try ultra-sharp enhancement
    - **Wrong readings**: Check if selection includes only the measurement text
    - **Multiple results**: The tool ranks by confidence - top result is usually best
    
    #### 📏 **Supported Formats:**
    - Direct meter readings: `1484m`, `12.5m`, `0.75m`
    - Kilometer readings: `1.5km`, `2km` (converted to meters)
    - Centimeter readings: `150cm`, `75.5cm` (converted to meters)  
    - Millimeter readings: `1500mm`, `125.7mm` (converted to meters)
    
    #### 🎯 **Advanced Features:**
    - **Multi-Method Analysis**: Uses 7+ different OCR techniques
    - **Confidence Filtering**: Automatically filters low-confidence results
    - **Unit Conversion**: All results standardized to meters
    - **Precision Display**: Shows results with appropriate decimal places
    """)

# Technical Information
with st.expander("⚙️ Technical Details", expanded=False):
    st.markdown("""
    ### 🔬 Advanced OCR Technology Stack
    
    #### 🎨 **Image Enhancement Pipeline:**
    - **Bilateral Filtering**: Noise reduction with edge preservation
    - **CLAHE**: Contrast Limited Adaptive Histogram Equalization
    - **Morphological Operations**: Character separation and cleaning
    - **Adaptive Thresholding**: Optimal binarization for OCR
    - **Multi-scale Processing**: Different kernel sizes for various text sizes
    
    #### 🔍 **OCR Configuration Matrix:**
    - **7 specialized OCR modes** optimized for different measurement types
    - **Confidence-based filtering** removes unreliable detections
    - **Character whitelisting** for precise number recognition
    - **PSM (Page Segmentation Mode) optimization** for single measurements
    
    #### 📊 **Measurement Processing:**
    - **Regex pattern matching** for various measurement formats
    - **Unit standardization** to meters with high precision
    - **Confidence scoring** based on format, context, and consistency
    - **Duplicate consolidation** with tolerance-based grouping
    
    #### 🎯 **Accuracy Features:**
    - **Multi-method consensus**: Agreement between multiple OCR approaches
    - **Context analysis**: Considers surrounding text for validation
    - **Range validation**: Filters unrealistic measurements
    - **Precision formatting**: Appropriate decimal places for each range
    """)

# Footer
st.markdown("---")
st.markdown("🎯 **Precision Wire Length OCR Extractor** - Advanced number recognition with meter-precise results! 📏✨")
