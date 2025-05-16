### Требования к окружению  

1. Установите зависимости:  
   ```bash
   pip install PyMuPDF pytesseract
   ```
2. Установите Tesseract OCR:  
   - Linux: `!sudo apt-get install tesseract-ocr`  
   - Windows: [https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)  
   - macOS: `brew install tesseract`  

3. Установите языковые пакеты (если требуется):  
   ```bash
   sudo apt-get install tesseract-ocr-rus  # Для русского языка
   ```

---

### Пример использования  

```bash
python ocr_script.py da8b0f0b-cfea-40c4-b020-e0924a142f4b.pdf rus
```

Результат сохранится в `da8b0f0b-cfea-40c4-b020-e0924a142f4b_ocr.json`.