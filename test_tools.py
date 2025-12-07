from pytest import raises
from tools import GestureHandler

test_handler = GestureHandler()

def test_no_file_index():
    index = test_handler.get_file_index("ADIOS.mp4")
    assert index == ""

def test_no_file_name():
    index = test_handler.get_file_index(".mp4")
    assert index == ""

def test_file_name_starts_with_index():
    index = test_handler.get_file_index("1ADIOS.mp4")
    assert index == "1"

def test_no_alpha_characters():
    label_name = test_handler.get_label_name("123456789.mp4")
    assert label_name == ""

def test_middle_dash_character():
    label_name = test_handler.get_label_name("BUENOS-DIAS.mp4")
    assert label_name == "BUENOS-DIAS"

def test_starts_with_period():
    with raises(ValueError, match="File name can't start with a period character."):
        test_handler.get_label_name(".DIAS.mp4")