import argparse
# import cProfile
import itertools
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import fitz  # PyMuPDF - pip install pymupdf
from PIL import Image
from dateutil.relativedelta import relativedelta
# from memory_profiler import profile
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

def read_pdf_to_dict(pdf_path, num_threads, batch_size):
    doc = open_pdf(pdf_path)
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        pages_text = batch_process(executor, get_page_text, doc, batch_size=batch_size)
        page_generator = (
            (f"Page_{page_num}", parse_data_text(text))
            for page_num, text in enumerate(pages_text, 1)
        )
        page_dict = dict(page_generator)
    return page_dict

def batch_process(executor, func, items, batch_size):
    items_iter = iter(items)
    batches = iter(lambda: list(itertools.islice(items_iter, batch_size)), [])
    return itertools.chain.from_iterable(
        executor.map(func, batch) for batch in batches
    )

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

def validate_structure(template_data, test_data):
    template_keys = set(template_data.keys())
    test_keys = set(test_data.keys())
    if template_keys != test_keys:
        missing = template_keys - test_keys
        extra = test_keys - template_keys
        if missing:
            logging.error(f"Отсутствующие ключи: {missing}")
        if extra:
            logging.error(f"Лишние ключи: {extra}")
        return False

    field_validations = {
        "PN": {
            "validator": lambda v: not v.strip().isdigit(),
            "message": "PN должен быть текстом (не только цифры)"
        },
        "SN": {
            "validator": lambda v: v.isdigit() and len(v) >= 6,
            "message": "SN должен содержать 6 или более цифр"
        },
        "DESCRIPTION": {
            "validator": lambda v: v.strip() != "",
            "message": "DESCRIPTION не может быть пустым"
        },
        "LOCATION": {
            "validator": lambda v: v.isdigit(),
            "message": "LOCATION должен содержать только цифры"
        },
        "CONDITION": {
            "validator": lambda v: len(v.strip()) >= 2,
            "message": "CONDITION должен содержать 2+ символа"
        },
        "RECEIVER#": {
            "validator": lambda v: v.isdigit(),
            "message": "RECEIVER# должен содержать только цифры"
        },
        "UOM": {
            "validator": lambda v: len(v.strip()) >= 2,
            "message": "UOM должен содержать 2+ символа"
        },
        "EXP DATE": {
            "validator": validate_date,
            "message": "Неправильный формат даты или дата превышает 10 лет"
        },
        "PO": {
            "validator": lambda v: len(v.strip()) >= 4,
            "message": "PO должен содержать 4+ символа"
        },
        "CERT SOURCE": {
            "validator": lambda v: v.strip() != "",
            "message": "CERT SOURCE не может быть пустым"
        },
        "REC.DATE": {
            "validator": validate_date,
            "message": "Неправильный формат даты или дата превышает 10 лет"
        },
        "MFG": {
            "validator": lambda v: v.strip() != "",
            "message": "MFG не может быть пустым"
        },
        "BATCH#": {
            "validator": lambda v: v.isdigit(),
            "message": "BATCH должен содержать только цифры"
        },
        "DOM": {
            "validator": validate_date,
            "message": "Неправильный формат даты или дата превышает 10 лет"
        },
        "LOT#": {
            "validator": lambda v: v.isdigit(),
            "message": "LOT# должен содержать только цифры"
        },
        "Qty": {
            "validator": lambda v: v.isdigit(),
            "message": "Qty должен содержать только цифры"
        },
        "NOTES": {
            "validator": lambda v: v.strip() != "",
            "message": "NOTES не может быть пустым"
        }
    }

    for field in template_data.keys():
        value = test_data.get(field, "")
        validation = field_validations.get(field)
        if not validation:
            continue
        validator_func = validation['validator']
        message = validation['message']
        if not validator_func(value):
            logging.error(f"Ошибка в поле {field}: {message}")
            return False
    return True

def validate_date(date_str):
    formats = (
        "%d.%m.%Y",
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%d %m %Y"
    )
    for fmt in formats:
        try:
            date = datetime.strptime(date_str, fmt)
            return date <= datetime.now() + relativedelta(years=10)
        except ValueError:
            continue
    return False

def extract_barcodes(pdf_path, num_threads, dpi, batch_size):
    doc = open_pdf(pdf_path)
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        page_barcodes = batch_process(
            executor,
            lambda page: process_page_for_barcodes(page, dpi),
            doc,
            batch_size=batch_size
        )
        return {
            f"Page_{page_num}": barcodes
            for page_num, barcodes in enumerate(page_barcodes, start=1)
        }

def process_page_for_barcodes(page, dpi):
    if dpi < 150 or dpi > 600:
        logging.error(f"Ошибка: Значение DPI {dpi} для страницы {page.number + 1} вне допустимого диапазона (150-600).")
        return []
    try:
        pix = page.get_pixmap(dpi=dpi)
        img = preprocess_image(pix)
        decoded_objects = decode(img)
        return [obj.data.decode('utf-8', errors='ignore') for obj in decoded_objects]
    except Exception as e:
        logging.error(f"Ошибка при обработке страницы {page.number + 1}: {e}", exc_info=True)
        return []

def preprocess_image(pix):
    if not (hasattr(pix, "width") and hasattr(pix, "height")):
        raise AttributeError("Объект pix должен содержать атрибуты width и height")
    if pix.width <= 0 or pix.height <= 0:
        raise ValueError(f"Некорректные размеры изображения: {pix.width}x{pix.height}")
    mode = "RGBA" if pix.n >= 4 else ("RGB" if pix.colorspace else "L")
    image = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
    if pix.n >= 4:
        image = image.convert("RGB")
    return image

# @profile
def main():
    parser = argparse.ArgumentParser(
        description="Программа проверяет PDF-файл на соответствие структуре шаблона и распознаёт штрих-коды.",
        epilog="Пример использования: python script.py template.pdf test.pdf --threads 4 --batch-size=15 --dpi 300",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "template_pdf",
        nargs='?',
        default="test_task.pdf",
        help="Путь к PDF-файлу с шаблоном (по умолчанию: test_task.pdf)"
    )
    parser.add_argument(
        "test_pdf",
        nargs='?',
        default="test_task.pdf",
        help="Путь к PDF-файлу для проверки (по умолчанию: test_task.pdf)"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=min(8, os.cpu_count() or 2),
        help="Максимальное количество потоков (по умолчанию: min(8, количество CPU ядер или 2))"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Размер пакета для обработки страниц (по умолчанию: 10)"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="DPI для распознавания штрих-кодов (по умолчанию: 200)"
    )
    args = parser.parse_args()
    try:
        template_doc = read_pdf_to_dict(args.template_pdf, args.threads, args.batch_size)
        logging.info("Шаблон: %s", template_doc)
        test_doc = read_pdf_to_dict(args.test_pdf, args.threads, args.batch_size)
        logging.info("Тестовые данные: %s", test_doc)
        for page_num in template_doc:
            template_page = template_doc.get(page_num, {})
            test_page = test_doc.get(page_num, {})
            if not validate_structure(template_page, test_page):
                logging.error(f"Структура страницы {page_num} не соответствует шаблону")
            else:
                logging.info(f"Структура страницы {page_num} корректна")
        barcodes = extract_barcodes(args.test_pdf, args.threads, args.dpi, args.batch_size)
        logging.info("Найденные баркоды по страницам: %s", barcodes)
    except Exception as e:
        logging.error(f"Ошибка: {e}", exc_info=True)

if __name__ == "__main__":
    # cProfile.run('main()')
    main()
