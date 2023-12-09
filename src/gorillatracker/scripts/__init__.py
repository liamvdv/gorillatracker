import logging
import time

timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
logger = logging.getLogger(__name__)
handler = logging.FileHandler(f"logs/scripts-{timestamp}.log")
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
