# Dataset creation

First, you have to download a "raw" dataset from the internet (or create one yourself).

("raw" means that python cannot use it to train directly.)

This "raw" dataset contains images or videos (but not both) with people signing different signs
from one sign language.

This dataset will be called "data" from now on.

Your data folder (where you store videos or images to train the models) should look like this:

``` ASCII
.
└── data/
    ├── LABEL/
    │   ├── LABEL1
    │   ├── LABEL2
    │   ├── LABEL3
    │   └── ...
    ├── ANOTHER_LABEL/
    │   ├── ANOTHER_LABEL1
    │   ├── ANOTHER_LABEL2
    │   ├── ANOTHER_LABEL3
    │   └── ...
    └── ...
```

Where "LABEL" is the uppercase word or letter corresponding to the signs showed in the files.
The files must be named after the label AND a unique number inside the folder.
For example, a valid filename could be HELLO1.mp4 or HELLO1.png (depending on the file extension)

The dataset creation relies on this structure.

After ensuring your data meets the criteria, you then create a dataset python can use for training:

## Video dataset creation

``` python3
# main.py in root
VideoSignHandler = VideoHandler(data_parent_folder="data")
# create datasets code
VideoSignHandler.create_dataset("dataset")
# training code (after the dataset is created created)
VideoSignHandler.train("dataset", "models/words_model.keras")
```

After using your data to create a dataset (via VideoHandler), the resulting dataset would look like this:

``` ASCII
.
└── dataset/
    ├── LABEL/
    │   ├── 1_frame_1.npy
    │   ├── 1_frame_15.npy
    │   ├── ...
    │   ├── 2_frame_73.npy
    │   └── ...
    └── ...
```

It uses the same labels inside your data folder to name the folders there and
obviously adds the numpy data from the files in data/.

The numpy files are named this way (if the dataset is created with VideoHandler):
  {source_file_index}frame{source_frame_index}.npy
  (with underscores between the "variables")

This is called like this for debugging purposes, this way you can know which is the first usable frame in a video.

## Image dataset creation

``` python3
# main.py in root
ImageSignHandler = ImageHandler("data")
ImageSignHandler.create_dataset("dataset")
# training code (after the dataset is created created)
ImageSignHandler.train("dataset", "models/letters.keras")
```

If the dataset is created via ImageHandler, the dataset would look like this:

``` ASCII
.
└── dataset/
    ├── LABEL/
    │   ├── 1.npy
    │   ├── 2.npy
    │   └── ...
    ├── ANOTHER_LABEL/
    │   ├── 1.npy
    │   ├── 2.npy
    │   └── ...
    └── ...
```

It uses the same labels inside your data folder to name the folders there and
obviously adds the numpy data from the files in data/.

The numpy files are named this way (if the dataset is created with ImageHandler):
  {source_frame_index}.npy
