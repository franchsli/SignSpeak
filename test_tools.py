from tools import GestureHandler

test_handler = GestureHandler()

def test_no_file_index():
    index = test_handler.get_file_index("ADIOS")
    assert index == ""

def test_no_file_name():
    index = test_handler.get_file_index("")
    assert index == ""

def test_file_name_starts_with_index():
    index = test_handler.get_file_index("1ADIOS")
    assert index == "1"