from cv2 import imread
from numpy import all
from pytest import raises

from core.data_handlers import GestureHandler, ImageHandler, VideoHandler
from core.processor import MediaPipeProcessor

test_handler = GestureHandler()


def test_no_file_index():
    with raises(ValueError, match="File names must have a defined index."):
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
    handler = ImageHandler()
    signs = ["A", "C", "F"]
    landmarks, labels = handler._load_frame("testing/letters_dataset", signs)
    assert landmarks.shape[1] == 126
    assert labels.shape[1] == len(signs)


def test_create_sequences_length():
    handler = VideoHandler()
    SEQUENCE_LENGTH = 10
    signs = ["ADIOS", "BONITO"]
    sequences, _ = handler._create_sequences(
        "testing/words_dataset", signs, SEQUENCE_LENGTH
    )
    assert sequences.shape[1] == SEQUENCE_LENGTH
    assert sequences.shape[2] == 126


def test_holistic_keypoint_extraction():
    processor = MediaPipeProcessor()
    frame = imread("testing/letters_signs/A/A1.png")
    results, _ = processor.process_image(frame)
    keypoints = processor.keypoint_extraction(results)
    assert keypoints.shape == (126,)


def test_hands_keypoint_extraction():
    processor = MediaPipeProcessor(mode="hands")
    frame = imread("testing/letters_signs/A/A1.png")
    results, _ = processor.process_image(frame)
    keypoints = processor.keypoint_extraction(results)
    assert keypoints.shape == (126,)


def test_keypoint_shape_no_hands():
    processor = MediaPipeProcessor(mode="hands")
    frame = imread("testing/letters_signs/C/C3.jpg")
    results, _ = processor.process_image(frame)
    keypoints = processor.keypoint_extraction(results)
    assert keypoints.shape == (126,)
    assert all(keypoints == 0)
