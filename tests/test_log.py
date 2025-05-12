import unittest
from deepsearcher.tools import log

class TestLog(unittest.TestCase):
    def test_log(self):
        log.color_print("hello world")

        log.color_print("----")
        log.info("dev info log")
        log.color_print("----")
        log.set_dev_mode(True)
        log.info("dev info log")
        log.color_print("====")
        log.debug("dev debug log")
        log.info("dev info log")
        log.warning("dev warning log")
        log.error("dev error log")
        log.critical("dev critical log")
        