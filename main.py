import sign_language_translator as slt

# download dataset or models (if you need them for personal use)
# (by default, resources are auto-downloaded within the install directory)
# slt.Assets.set_root_dir("path/to/folder")  # Helps preventing duplication across environments or using cloud synced data
# slt.Assets.download(r".*.json")  # downloads into asset_dir
# print(slt.Assets.FILE_TO_URL.keys())  # All downloadable resources

# print("All available models:")
# print(list(slt.ModelCodes))  # slt.ModelCodeGroups
# print(list(slt.TextLanguageCodes))
# print(list(slt.SignLanguageCodes))
# print(list(slt.SignFormatCodes))

# -------------------------- TRANSLATE: sign to text --------------------------
# -------------------------- THIS DOES NOT WORK --------------------------

# sign = slt.Video("path/to/video.mp4")
sign = slt.Video.load("datasets/10-words-slc-and-3-people/001_SLC.mp4")
sign.show_frames_grid()

# Extract Pose Vector for feature reduction
embedding_model = slt.models.MediaPipeLandmarksModel()      # pip install "sign_language_translator[mediapipe]"  # (or [all])
embedding = embedding_model.embed(sign.iter_frames())

slt.Landmarks(embedding.reshape((-1, 75, 5)),
            connections="mediapipe-world").show()

# Load sign-to-text model (pytorch) (COMING SOON!)
translation_model = slt.get_model(slt.ModelCodes.Gesture)
text = translation_model.translate(embedding)
print(text)