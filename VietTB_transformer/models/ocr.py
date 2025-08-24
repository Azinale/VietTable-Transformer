import easyocr

def init_easyocr(lang_list=["en", "vi"]):
    """Khởi tạo OCR reader"""
    return easyocr.Reader(lang_list)

def run_ocr(reader, image_np):
    """Chạy OCR trên ảnh crop"""
    if reader is None or image_np is None:
        return ""
    results = reader.readtext(image_np)
    text_lines = [res[1] for res in results]
    return " ".join(text_lines)
