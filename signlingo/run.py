"""
SignLingo — Real-Time ASL Recognition + Indian Language Translation
Entry point: python run.py
"""
import sys
import logging
import logging.handlers
from pathlib import Path
from datetime import datetime

import yaml
from dotenv import load_dotenv

load_dotenv()


def setup_logging(log_dir: str = 'logs', level: str = 'INFO'):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / f"signlingo_{datetime.now().strftime('%Y%m%d')}.log"

    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )

    # Rotating file handler (5MB, 3 backups)
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding='utf-8'
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    return logging.getLogger('signlingo')


def load_config(config_path: str = 'config/config.yaml') -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_language_prompts(prompts_path: str = 'config/language_prompts.yaml') -> dict:
    with open(prompts_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    # Load config
    config = load_config()
    app_cfg = config.get('app', {})

    # Setup logging
    logger = setup_logging(
        log_dir=app_cfg.get('log_dir', 'logs'),
        level=app_cfg.get('log_level', 'INFO')
    )
    logger.info("Starting SignLingo...")

    # Check CUDA
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory // (1024 ** 2)
            logger.info(f"GPU: {gpu} | VRAM: {vram} MB | CUDA: {torch.version.cuda}")
        else:
            logger.warning("CUDA not available — running on CPU")
    except ImportError:
        logger.error("PyTorch not installed. Run: pip install -r requirements.txt")
        sys.exit(1)

    # Load language prompts
    language_prompts = load_language_prompts()

    # Check Ollama
    from src.translation.ollama_client import OllamaClient
    trans_cfg = config.get('translation', {})
    ollama = OllamaClient(
        base_url=trans_cfg.get('ollama_base_url', 'http://localhost:11434'),
        model=trans_cfg.get('ollama_model', 'llama3.2:3b-instruct-q4_K_M')
    )
    if not ollama.check_connection():
        logger.warning(
            "Ollama not running. Start with: ollama serve\n"
            "SignLingo will display ASL gloss only until Ollama is available."
        )

    # Initialize pipeline
    from src.pipeline import SignLingoPipeline
    pipeline = SignLingoPipeline(config, language_prompts)

    # Initialize Qt application
    from PyQt6.QtWidgets import QApplication
    from src.ui.main_window import MainWindow

    app = QApplication(sys.argv)
    app.setApplicationName("SignLingo")
    app.setApplicationVersion("1.0.0")

    window = MainWindow(pipeline=pipeline)

    # Connect pipeline callbacks to UI
    pipeline.set_ui_callback(window.on_ui_update)
    pipeline.set_translation_callback(window.on_translation_result)

    # Start pipeline
    try:
        pipeline.start()
    except RuntimeError as e:
        logger.error(f"Failed to start pipeline: {e}")
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.critical(None, "Camera Error", str(e))
        sys.exit(1)

    window.show()
    exit_code = app.exec()

    # Cleanup
    pipeline.stop()
    logger.info("SignLingo exited cleanly")
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
