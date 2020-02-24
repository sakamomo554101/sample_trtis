# Model Information

## sample instance

TBD

## sample sequence

TBD

## mecab model

This model split the sentences into words using Mecab.

ex)
    input text is "田中です"
    output text is "田中|です"

## face_recognition_model

This model performs face detection and face recognition.

### result format

Face recognition model return the json result.
This result format is following.

```
"result":
{
    "image_count": 3,
    "image_infos":
        [
            {
                "face_count": 2,
                "face_infos":
                [
                    {
                    "name": taro,
                    "box": 
                        {
                        "top": 134,
                        "bottom": 313,
                        "left": 234,
                        "right": 413
                        }
                    },
                    {
                    ...
                    },
                    ...
                ]
            },
            {
                ...
            },
        ]
}
```
