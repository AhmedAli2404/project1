import sys

from src.logging import logging


def error_msg_detail(error_msg,error_detail:sys):
    er_type,error_value,exc_tb=error_detail.exc_info()
    file_name=exc_tb.tb_frame.f_code.co_filename


    msg="Error occured in filename of {0} at line number {1} with the exception {2}".format(
        file_name,exc_tb.tb_lineno,str(error_msg)
    
        )
    return msg



class CustomException(Exception):

    def __init__(self,error_msg,error_detail):
        super().__init__(error_msg)
        self.error_msg=error_msg_detail(error_msg,error_detail=error_detail)

    

    def __str__(self):

        return self.error_msg

