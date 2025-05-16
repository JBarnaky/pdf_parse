import asyncio
import nest_asyncio
import json
import os
import uuid
import logging
import tempfile
from typing import List, Dict, Optional
from pathlib import Path
import gc

import fitz  # PyMuPDF
import easyocr

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_page_to_image(doc, page_num, dpi=150):
    """
    Синхронно извлекает страницу PDF и возвращает изображение в формате PNG в виде байтов.
    
    Args:
        doc: Открытый документ PyMuPDF.
        page_num: Номер страницы (0-based).
        dpi: Разрешение изображения (по умолчанию 200).
    
    Returns:
        Байты изображения в формате PNG.
    """
    try:
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=dpi)
        image_bytes = pix.tobytes("png")
        pix = None  # Освобождаем ресурсы, связанные с pixmap
        return image_bytes
    except Exception as e:
        logger.error(f"Ошибка извлечения изображения страницы {page_num + 1}: {e}")
        return b""

async def process_page(doc, page_num, reader, semaphore, temp_dir, dpi=150):
    """
    Асинхронно обрабатывает одну страницу PDF-документа.
    
    Args:
        doc: Открытый документ PyMuPDF.
        page_num: Номер страницы (0-based).
        reader: Экземпляр EasyOCR.Reader.
        semaphore: Семафор для ограничения параллелизма.
        temp_dir: Путь к временной директории для хранения изображений.
        dpi: Разрешение изображения для OCR.
    
    Returns:
        Кортеж (номер страницы, извлечённый текст).
    """
    logger.info(f"Начало обработки страницы {page_num + 1}")

    image_bytes = await asyncio.to_thread(load_page_to_image, doc, page_num, dpi)
    if not image_bytes:
        logger.warning(f"Не удалось получить изображение для страницы {page_num + 1}")
        return page_num + 1, ""

    logger.info(f"Страница {page_num + 1} конвертирована в изображение")

    image_path = os.path.join(temp_dir, f"page_{page_num + 1}_{uuid.uuid4().hex}.png")

    try:
        # Сохранение изображения в памяти во временный файл
        with open(image_path, "wb") as f:
            f.write(image_bytes)
        logger.info(f"Изображение страницы {page_num + 1} сохранено во временный файл")

        if not os.path.exists(image_path):
            logger.error(f"Файл {image_path} не найден после записи")
            return page_num + 1, ""
        
        # Выполнение OCR
        logger.info(f"Начало OCR для страницы {page_num + 1}")
        
        try:
            ocr_result = await asyncio.to_thread(reader.readtext, image_path, detail=0)
            logger.info(f"OCR завершён для страницы {page_num + 1}")
            return page_num + 1, " ".join(ocr_result)
        except Exception as e:
            logger.error(f"OCR ошибка для страницы {page_num + 1}: {e}")
            return page_num + 1, ""
    except Exception as e:
        logger.error(f"Ошибка обработки страницы {page_num + 1}: {e}")
        return page_num + 1, ""
    finally:
        # Автоматическое удаление временного файла благодаря контекстному менеджеру
        if os.path.exists(image_path):
            try:
                os.remove(image_path)
                logger.debug(f"Временный файл {image_path} удалён")
            except Exception as e:
                logger.error(f"Ошибка удаления временного файла {image_path}: {e}")

        gc.collect()

async def extract_text_from_pdf(pdf_path, languages=['en'], max_concurrent=2, dpi=150):
        """
    Асинхронно извлекает текст из PDF-документа с помощью OCR (EasyOCR).
    
    Args:
        pdf_path: Путь к PDF-файлу.
        languages: Список языков для OCR (например, ['en', 'ru']).
        max_concurrent: Максимальное количество параллельных задач.
        dpi: Разрешение изображения для OCR.
    
    Returns:
        Словарь с номерами страниц и извлечённым текстом.
    """
    if not os.path.isfile(pdf_path):
        logger.error(f"Файл '{pdf_path}' не найден.")
        return None

    try:
        logger.info(f"Начало обработки PDF-файла: {pdf_path}")
        # Инициализация OCR Reader
        logger.info(f"Инициализация EasyOCR Reader с языками: {languages}")
        reader = await asyncio.to_thread(easyocr.Reader, languages, gpu=True)

        # Открытие PDF документа
        logger.info(f"Открытие PDF-документа: {pdf_path}")
        doc = await asyncio.to_thread(fitz.open, pdf_path)
        logger.info(f"Документ содержит {len(doc)} страниц")

        # Создание временной директории для изображений
        with tempfile.TemporaryDirectory() as temp_dir:
            logger.info(f"Создана временная директория для изображений: {temp_dir}")

            # Ограничение параллелизма
            semaphore = asyncio.Semaphore(max_concurrent)

            # Создание задач для обработки страниц
            logger.info(f"Создание задач для обработки {len(doc)} страниц")
            tasks = [
                process_page(doc, page_num, reader, semaphore, temp_dir, dpi)
                for page_num in range(len(doc))
            ]

            # Выполнение задач
            logger.info("Начало параллельной обработки страниц")
            results = await asyncio.gather(*tasks)
            logger.info("Завершена обработка всех страниц")

            # Формирование результата
            text_data = {page_num: text for page_num, text in results}
            logger.info(f"Извлечено {sum(1 for text in text_data.values() if text)} страниц с текстом")
            return text_data

    except Exception as e:
        logger.error(f"Ошибка при обработке PDF: {e}")
        return None
    finally:
        try:
            # Закрытие документа
            if 'doc' in locals():
                await asyncio.to_thread(doc.close)
            logger.info("Документ успешно закрыт")
            if 'reader' in locals():
                del reader
            gc.collect()
        except Exception as e:
            logger.error(f"Ошибка при закрытии документа: {e}")

async def save_to_json(data, output_path):
    """
    Асинхронно сохраняет данные в JSON-файл.
    """
    try:
        logger.info(f"Начало сохранения результатов в JSON: {output_path}")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)  # Создаёт все родительские директории
        logger.info(f"Сохранение в: {output_path}")

        def write_file():
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

        await asyncio.to_thread(write_file)
        logger.info(f"Результат сохранён в '{output_path}'.")
    except Exception as e:
        logger.error(f"Ошибка при сохранении JSON: {e}")

async def main():
    """Основная функция для запуска обработки PDF."""
    pdf_file = "da8b0f0b-cfea-40c4-b020-e0924a142f4b.pdf"
    output_file = "extracted_text.json"

    # Извлечение текста
    logger.info("Начало основной функции обработки")
    text_data = await extract_text_from_pdf(pdf_file, languages=["en"], max_concurrent=2, dpi=150)

    # Сохранение результата
    if text_data:
        logger.info("Начало сохранения извлечённых данных")
        await save_to_json(text_data, output_file)
        logger.info("Обработка завершена успешно")
    else:
        logger.warning("Не удалось извлечь текст из PDF-файла")

if __name__ == "__main__":
    nest_asyncio.apply()
    asyncio.run(main())
