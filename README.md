# SignSpeak

A sign language translator.

## TODO

- Create colombian sign language model:
    1. Data Collection:
        - Gather a collection of dictionary videos (word level) featuring individuals performing sign language gestures. These can be obtained from schools & organizations for the deaf. You should record multiple people perform the same sign to capture various accents of the sign. Set up multiple cameras in different locations in parallel to further augment the data.

        - Prepare a JSON file that maps dictionary video file names to corresponding text language words & phrases that are synonymous with the gestures.

        - Prepare a synthetic data parallel corpus containing text language sentences and sequences of sign language video filenames. You can use langauge models to generate these sentences & sequences.

        - Prepare a dataset of sign language sentence videos that are labeled with translations & glosses in multiple text languages.

    2. Language Processing:
        - Implement a subclass of slt.languages.TextLanguage:
            - Tokenize your text language and assign appropriate tags to the tokens for streamlined processing.

        - Create a subclass of slt.languages.SignLanguage:
            - Map text tokens to video filenames using the provided JSON data.
            - Rearrange the sequence of video filenames to align with the grammar and structure of sign language

    3. Rule-Based Translation:
        - Pass instances of your classes from the previous step to slt.models.ConcatenativeSynthesis class to obtain a rule-based translator object.
        - Construct sentences in your text language and use the rule-based translator to generate sign language translations. (You can use our language models to generate such texts.)

    4. Deep Learning Model Fine-Tuning:
        - Utilize the (synthetic & real) sign language videos and corresponding text sentences from the previous step.
        - Apply our training pipeline to fine-tune a chosen model for improved accuracy and translation quality.

## General Resources

[Sign language library docs](https://sign-language-translator.readthedocs.io/en/latest/#building-custom-translators)

[Datasets](https://www.kaggle.com/datasets?search=colombian+sign+language)

[More Data](https://www.youtube.com/watch?v=JMraBJsA9oI&list=PLI7rDimYXOdhyty-lEXsxQgiLfYKnnqmY&index=4)

[Very good dataset](https://bivl2ab.uis.edu.co/dataset-info)

## Important

Remember to **contribute** back to the community:

Share your data, code, and models by creating a pull request, allowing others to benefit from your efforts.
Create your own sign language translator (e.g. as your university thesis) and contribute to a more inclusive and accessible world
