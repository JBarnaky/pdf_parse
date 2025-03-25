import argparse
import itertools
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import fitz  # PyMuPDF - pip install pymupdf
from PIL import Image
from dateutil.relativedelta import relativedelta
from pyzbar.pyzbar import decode  # !apt install libzbar0

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True
)

def open_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        if doc.is_encrypted:
            if not doc.authenticate(""):
                logging.error(f"Файл {pdf_path} защищен паролем.")
                raise ValueError("Файл защищен паролем.")
        return doc
    except FileNotFoundError:
        logging.error(f"Файл {pdf_path} не найден.")
        raise
    except fitz.FileDataError:
        logging.error(f"Файл {pdf_path} поврежден или не является PDF.")
        raise ValueError("Файл поврежден или не является PDF.")
    except Exception as e:
        logging.error(f"Неизвестная ошибка при открытии файла {pdf_path}: {str(e)}")
        raise

def get_page_text(page):
    try:
        return page.get_text("text").strip()
    except Exception as e:
        logging.error(f"Ошибка при обработке страницы {page.number + 1}: {e}")
        return ""

def parse_data_text(text):
    data_dict = {}
    current_key = None
    for line in text.split('\n'):
        if ':' in line:
            key, _, value = line.partition(':')
            current_key = key.strip()
            data_dict[current_key] = value.strip()
        elif current_key:
            data_dict[current_key] += ' ' + line.strip()
    return data_dict

def preprocess_image(pix):
    if not hasattr(pix, "width") or not hasattr(pix, "height"):
        raise AttributeError("Объект pix должен содержать атрибуты width и height")
    if pix.width <= 0 or pix.height <= 0:
        raise ValueError(f"Некорректные размеры изображения: {pix.width}x{pix.height}")
    if pix.n < 4:
        mode = "RGB" if pix.colorspace else "L"
        return Image.frombytes(
            mode,
            (pix.width, pix.height),
            pix.samples
        )
    else:
        return Image.frombytes(
            "RGBA",
            (pix.width, pix.height),
            pix.samples
        ).convert("RGB")

def batch_process(executor, func, items, batch_size=10):
    items_iter = iter(items)
    batches = iter(lambda: list(itertools.islice(items_iter, batch_size)), [])
    return itertools.chain.from_iterable(
        executor.map(func, batch) for batch in batches
    )

def process_page_text(params):
    pdf_path, page_num = params
    try:
        doc = open_pdf(pdf_path)
        page = doc.load_page(page_num)
        text = get_page_text(page)
        doc.close()
        return text
    except Exception as e:
        logging.error(f"Ошибка при обработке страницы {page_num + 1}: {e}")
        return ""

def read_pdf_to_dict(pdf_path, num_threads):
    doc = open_pdf(pdf_path)
    num_pages = len(doc)
    doc.close()
    with ProcessPoolExecutor(max_workers=num_threads) as executor:
        items = [(pdf_path, page_num) for page_num in range(num_pages)]
        pages_text = batch_process(executor, process_page_text, items)
        page_generator = (
            (f"Page_{page_num+1}", parse_data_text(text))
            for page_num, text in enumerate(pages_text)
        )
        page_dict = dict(page_generator)
    return page_dict

def validate_date(date_str):
    formats = [
        "%d.%m.%Y",
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%d %m %Y"
    ]
    for fmt in formats:
        try:
            date = datetime.strptime(date_str, fmt)
            return date <= datetime.now() + relativedelta(years=10)
        except ValueError:
            continue
    return False

def validate_structure(template_data, test_data):
    missing_keys = set(template_data.keys()) - set(test_data.keys())
    if missing_keys:
        logging.error(f"Отсутствующие ключи: {missing_keys}")
        return False
    extra_keys = set(test_data.keys()) - set(template_data.keys())
    if extra_keys:
        logging.error(f"Лишние ключи: {extra_keys}")
        return False

    def is_valid_number(value):
        return value.isdigit()

    def is_valid_text(value, min_length=None):
        stripped = value.strip()
        return bool(stripped) and (not min_length or len(stripped) >= min_length)

    checks = [
        ("PN", lambda v: not is_valid_number(v.strip()), "PN должен быть текстом (не только цифры)"),
        ("SN", lambda v: is_valid_number(v) and len(v) >= 6, "SN должен содержать 6 или более цифр"),
        ("DESCRIPTION", lambda v: is_valid_text(v), "DESCRIPTION не может быть пустым"),
        ("LOCATION", lambda v: is_valid_number(v), "LOCATION должен содержать только цифры"),
        ("CONDITION", lambda v: is_valid_text(v, 2), "CONDITION должен содержать 2+ символа"),
        ("RECEIVER#", lambda v: is_valid_number(v), "RECEIVER# должен содержать только цифры"),
        ("UOM", lambda v: is_valid_text(v, 2), "UOM должен содержать 2+ символа"),
        ("EXP DATE", lambda v: validate_date(v), "Неправильный формат даты или дата превышает 10 лет"),
        ("PO", lambda v: is_valid_text(v, 4), "PO должен содержать 4+ символа"),
        ("CERT SOURCE", lambda v: is_valid_text(v), "CERT SOURCE не может быть пустым"),
        ("REC.DATE", lambda v: validate_date(v), "Неправильный формат даты или дата превышает 10 лет"),
        ("MFG", lambda v: is_valid_text(v), "MFG не может быть пустым"),
        ("BATCH#", lambda v: is_valid_number(v), "BATCH должен содержать только цифры"),
        ("DOM", lambda v: validate_date(v), "Неправильный формат даты или дата превышает 10 лет"),
        ("LOT#", lambda v: is_valid_number(v), "LOT# должен содержать только цифры"),
        ("Qty", lambda v: is_valid_number(v), "Qty должен содержать только цифры"),
        ("NOTES", lambda v: is_valid_text(v), "NOTES не может быть пустым"),
    ]
    for field, validator, message in checks:
        value = test_data.get(field, "")
        if not validator(value):
            logging.error(f"Ошибка в поле {field}: {message}")
            return False
    return True

def process_page_for_barcodes(params):
    pdf_path, page_num, dpi = params
    try:
        doc = open_pdf(pdf_path)
        page = doc.load_page(page_num)
        if dpi <= 150 or dpi >= 600:
            logging.error(f"Ошибка: Значение DPI {dpi} для страницы {page_num + 1} вне допустимого диапазона (150-600).")
            return []
        pix = page.get_pixmap(dpi=dpi)
        img = preprocess_image(pix)
        decoded_objects = decode(img)
        doc.close()
        return [obj.data.decode('utf-8', errors='ignore') for obj in decoded_objects]
    except Exception as e:
        logging.error(f"Ошибка при обработке страницы {page_num + 1}: {e}", exc_info=True)
        return []

def extract_barcodes(pdf_path, num_threads, dpi):
    doc = open_pdf(pdf_path)
    num_pages = len(doc)
    doc.close()
    with ProcessPoolExecutor(max_workers=num_threads) as executor:
        items = [(pdf_path, page_num, dpi) for page_num in range(num_pages)]
        page_barcodes = batch_process(executor, process_page_for_barcodes, items)
        return {
            f"Page_{page_num+1}": barcodes
            for page_num, barcodes in enumerate(page_barcodes)
        }

def main():
    parser = argparse.ArgumentParser(description="Обработка PDF-файлов и проверка их по шаблону.")
    parser.add_argument("template_pdf", nargs='?', default="test_task.pdf", help="Путь к PDF-файлу с шаблоном.")
    parser.add_argument("test_pdf", nargs='?', default="test_task.pdf", help="Путь к PDF-файлу для проверки.")
    parser.add_argument("-threads", type=int, default=min(8, os.cpu_count() or 2),
                        help="Максимальное количество потоков.")
    parser.add_argument("--dpi", type=int, default=200, help="DPI для распознавания штрих-кодов.")
    args = parser.parse_args()
    try:
        template_doc = read_pdf_to_dict(args.template_pdf, args.threads)
        logging.info("Шаблон: %s", template_doc)
        test_doc = read_pdf_to_dict(args.test_pdf, args.threads)
        logging.info("Тестовые данные: %s", test_doc)
        for page_num in template_doc:
            template_page = template_doc.get(page_num, {})
            test_page = test_doc.get(page_num, {})
            if not validate_structure(template_page, test_page):
                logging.error(f"Структура страницы {page_num} не соответствует шаблону")
            else:
                logging.info(f"Структура страницы {page_num} корректна")
        barcodes = extract_barcodes(args.test_pdf, args.threads, args.dpi)
        logging.info("Найденные баркоды по страницам: %s", barcodes)
    except Exception as e:
        logging.error(f"Ошибка: {e}", exc_info=True)

if __name__ == "__main__":
    main()