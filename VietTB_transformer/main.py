import time
from configs import settings
from utils import timer, file_io
from processing import pdf_utils, image_utils, table_extractor
from models import yolo_wrapper, ocr_easyocr

def main():
    # Bắt đầu tính giờ toàn bộ pipeline
    overall_timer = timer.Timer("Pipeline")
    overall_timer.start()

    # 1. Chọn file PDF đầu vào
    input_pdf = file_io.choose_file_dialog(file_types=[("PDF files", "*.pdf")])
    if not input_pdf:
        print("Không chọn file nào, thoát chương trình.")
        return

    # 2. Convert PDF -> list ảnh
    pdf_images = pdf_utils.pdf_to_images(input_pdf)
    print(f"Đã convert {len(pdf_images)} trang từ PDF.")

    # 3. Load YOLO model
    yolo_model, yolo_classes = yolo_wrapper.load_yolo_model_safe(
        settings.YOLO_MODEL_PATH, settings.DEVICE
    )

    # 4. Khởi tạo OCR (EasyOCR)
    ocr_reader = ocr_easyocr.init_easyocr(lang_list=settings.LANG_LIST)

    # 5. Xử lý từng trang
    results_all = []
    for i, page_img in enumerate(pdf_images, start=1):
        print(f"\n--- Xử lý trang {i} ---")
        
        # (a) YOLO detect bảng
        detections = yolo_wrapper.detect_tables(yolo_model, page_img)

        # (b) Cắt bảng & chạy OCR
        for det in detections:
            crop_img = image_utils.crop_bbox(page_img, det["bbox"])
            ocr_text = ocr_easyocr.run_ocr(ocr_reader, crop_img)
            results_all.append({"page": i, "bbox": det["bbox"], "text": ocr_text})

        # Hiển thị ảnh annotate nếu bật debug
        image_utils.display_image_with_bboxes(
            f"Trang {i}", page_img,
            cv_bbox_abs=None, tt_bbox_abs=None
        )

    # 6. Lưu kết quả
    output_csv = file_io.save_results_to_csv(results_all, "outputs/results.csv")
    print(f"Kết quả OCR đã lưu tại: {output_csv}")

    # In tổng thời gian
    overall_timer.stop()
    overall_timer.report()


if __name__ == "__main__":
    main()
