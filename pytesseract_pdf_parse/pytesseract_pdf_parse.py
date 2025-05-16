import os
import json
import logging
import tempfile
import shutil
from typing import Generator, Dict, List, Optional, Union
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import time

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def is_tesseract_available() -> bool:
    """Проверяет доступность Tesseract в системе и наличие необходимых языковых пакетов."""
    try:
        version = pytesseract.get_tesseract_version()
        lang_list = pytesseract.get_languages(config='')
        if 'eng' not in lang_list:
            logging.warning("Язык 'eng' не найден в доступных языках Tesseract.")
        return True
    except Exception as e:
        logging.error(f"Tesseract не установлен или недоступен: {e}")
        return False

def validate_file(path: str) -> None:
    """Проверяет существование и доступность файла для чтения."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Файл не найден: {path}")
    if not os.access(path, os.R_OK):
        raise PermissionError(f"Нет прав на чтение файла: {path}")

def ocr_image(image_path: str, lang: str = "eng", timeout: int = 10) -> Optional[str]:
    """
    Выполняет OCR над изображением из файла с контролем времени выполнения.
    
    Args:
        image_path: Путь к изображению
        lang: Язык распознавания
        timeout: Максимальное время выполнения OCR в секундах
        
    Returns:
        Распознанный текст или None в случае ошибки
    """
    try:
        start_time = time.time()
        text = pytesseract.image_to_string(image_path, lang=lang)
        elapsed = time.time() - start_time
        
        if elapsed > timeout:
            logging.warning(f"OCR для {image_path} занял {elapsed:.2f} секунд (> {timeout}s)")
            
        return text.strip() if text.strip() else None
    except Exception as e:
        logging.warning(f"Ошибка OCR: {e}")
        return None

def process_pdf(pdf_path: str, output_path: str, lang: str = "eng", 
                dpi: int = 200, timeout: int = 10, clean_temp: bool = True) -> Dict:
    """Обрабатывает PDF: извлекает текст через OCR и сохраняет в JSON."""
    if not is_tesseract_available():
        return {"error": "Tesseract недоступен или отсутствуют необходимые языковые пакеты"}
    
    result = {"pages": [], "errors": [], "total_pages": 0}
    
    try:
        validate_file(pdf_path)
        
        with fitz.open(pdf_path) as doc:
            total_pages = len(doc)
            result["total_pages"] = total_pages
            logging.info(f"Начата обработка PDF с {total_pages} страницами.")
            
            for i in range(total_pages):
                page_start_time = time.time()  # Время начала обработки страницы
                try:
                    page = doc.load_page(i)
                    pix = page.get_pixmap(dpi=dpi)
                    
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                        pix.save(tmpfile.name)
                        tmpfile_path = tmpfile.name
                    
                    ocr_start_time = time.time()  # Время начала OCR
                    text = ocr_image(tmpfile_path, lang, timeout)
                    ocr_duration = time.time() - ocr_start_time
                    
                    page_result = {
                        "page": i + 1,
                        "text": text,
                    }
                    
                    if text is None:
                        page_result["error"] = "Не удалось распознать текст"
                        result["errors"].append("Не удалось распознать текст")
                    
                    result["pages"].append(page_result)
                    total_duration = time.time() - page_start_time
                    
                    logging.info(f"Обработана страница {i + 1}/{total_pages}. "
                                 f"OCR: {ocr_duration:.2f}s | Общее время: {total_duration:.2f}s")
                    
                    if clean_temp:
                        os.remove(tmpfile_path)
                        
                except Exception as e:
                    total_duration = time.time() - page_start_time
                    logging.error(f"Ошибка обработки страницы {i+1}: {e}. "
                                 f"Общее время: {total_duration:.2f}s")
                    result["pages"].append({
                        "page": i + 1,
                        "error": str(e)
                    })
                    result["errors"].append(str(e))
                    
    except Exception as e:
        result["error"] = f"Ошибка во время обработки: {e}"
        return result

    # Сохранение результата в JSON
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        logging.info(f"Результат сохранён в {output_path}")
    except Exception as e:
        result["error"] = f"Ошибка сохранения JSON: {e}"
        return result

    return result

if __name__ == "__main__":
    pdf_file = "da8b0f0b-cfea-40c4-b020-e0924a142f4b.pdf"
    output_file = "extracted_text.json"
    language = "eng"
    
    result = process_pdf(pdf_file, output_file, language)
    if "error" in result:
        print(f"Ошибка: {result['error']}")
    else:
        print("Обработка завершена успешно.")