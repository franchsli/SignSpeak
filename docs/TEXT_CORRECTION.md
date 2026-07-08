# Text correction

Since the models are trained to predict with labels, it's very probable that the
labels used for training and such are general, without accents or special characters whatsoever.

Because of that, this project also includes an optional text corrector that uses the [language tool](https://languagetool.org/)
to do that. This is how it's implemented:

If any language it's given to the SignLanguageTranslator, the code will try to start a Java server that will run
locally to correct text in the given language. Since the server is local, the server will need to be in your PC.
But don't worry, an embedded HTTP server will be downloaded the first time you use the translator
(or the text correction itself) and that will be cached for further uses.

However, the text correction is not all mighty, don't expect it to fully correct all the errors in a text.

Since it's a local version, [the AI-based rules are not present](https://dev.languagetool.org/http-server#:~:text=The%20AI%2Dbased%20rules%20are%20only%20available%20in%20the%20cloud.)
so think of it like a weaker version of the language tool correction.

## Server warm up

The first correction is always the slowest, so the code warms up the server by checking a single letter string so first the
actual correction takes way less time:

https://github.com/franchsli/SignSpeak/blob/bdcaba125a05abca12b06e598c965ce52f503e04/core/translator.py#L26-L29

However, it doesn't mean the time gets cut off, it just transfers to the translator's initialization.
If the first correction would've took around 2 seconds, now it'll only take around 0.05 (or even less)
but the translator will take 2 seconds more to initialize.

## Server shut down

It's adviced to use a translator per language:

``` python3
# main.py in root
from core.translator import SignLanguageTranslator
# translating sign languages in spanish
with SignLanguageTranslator("es") as translator:
    ...
# translating sign languages in english
with SignLanguageTranslator("en") as translator:
    ...
```

This way, your translations remain organized and the respective instances will shut down when they're done.

If for some reason you do not use "with" statements when translating, you **MUST** explictly close the translators used:

``` python3
# main.py in root
from core.translator import SignLanguageTranslator
...
translator = SignLanguageTranslator("es")
# load the models down here
...
translation = translator.translate_video("some_video.mp4")
print(translation)
# close the translator to shut down the Java server.
translator.close()
```

This is a must because even though the language tool server gets shut down when garbage collected,
if that collection takes a while [the process might not get deleted right away.](https://pypi.org/project/language_tool_python/2.9.4/#:~:text=the%20process%20might%20not%20get%20deleted%20right%20away.)
