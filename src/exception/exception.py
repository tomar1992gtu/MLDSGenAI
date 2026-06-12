import sys


class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        self.error_message = error_message

        # sys.exc_info() returns 3 things: (type, value, traceback), This function gives information about the most recent exception.
        _, _, exc_tb = error_detail.exc_info()

        # Gets the exact line number where the error happened
        self.lineno = exc_tb.tb_lineno

        # tb_frame → current execution frame, f_code → code object, co_filename → file name where error occurred
        # So this gives you: Which file caused the error
        self.file_name = exc_tb.tb_frame.f_code.co_filename

        # Passes your message to Python’s built-in Exception system
        super().__init__(self.error_message)

    def __str__(self):
        return f"{self.error_message} (File: {self.file_name}, Line: {self.lineno})"
