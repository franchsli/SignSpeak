from pytest import raises
from tools import GestureHandler, ImageHandler, VideoHandler

test_handler = GestureHandler()

def test_no_file_index():
    with raises(ValueError, match=f"File names must have a defined index."):
        test_handler.get_file_index("ADIOS.mp4")

def test_no_file_name():
    with raises(ValueError, match="File name must start with an alphabetic character."):
        test_handler.get_file_index(".mp4")

def test_file_name_starts_with_index():
    with raises(ValueError, match="File name must start with an alphabetic character."):
        test_handler.get_file_index("1ADIOS.mp4")

def test_index_found():
    index = test_handler.get_file_index("ADIOS1.mp4")
    assert index == "1"

def test_no_alpha_characters():
    with raises(ValueError, match="File name must start with an alphabetic character."):
        test_handler.get_label_name("123456789.mp4")

def test_label_name_found():
    label_name = test_handler.get_label_name("ADIOS1.mp4")
    assert label_name == "ADIOS"

def test_middle_dash_character():
    label_name = test_handler.get_label_name("BUENOS-DIAS.mp4")
    assert label_name == "BUENOS-DIAS"

def test_starts_with_period():
    with raises(ValueError, match="File name must start with an alphabetic character."):
        test_handler.get_label_name(".DIAS.mp4")

def test_load_frame_correct_shape():
    handler = ImageHandler(data_parent_folder="letters_data")
    signs = ["A", "C", "F"]
    landmarks, labels = handler._load_frame("letters_test", signs)
    assert landmarks.shape[1] == 126
    assert labels.shape[1] == len(signs)