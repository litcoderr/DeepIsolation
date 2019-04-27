# DeepIsolation
Deep isolation using DeepLabv3++ Segmentation Model

# Prior Research
- "Looking to Listen" by Google : [https://ai.googleblog.com/2018/04/looking-to-listen-audio-visual-speech.html](https://ai.googleblog.com/2018/04/looking-to-listen-audio-visual-speech.html)
- "Phase aware speech enhancement with Deep U-Net" by Clova AI Research : [https://openreview.net/forum?id=SkeRTsAcYm](https://openreview.net/forum?id=SkeRTsAcYm)

#### Dataset I Used
- MusDB : [https://sigsep.github.io/datasets/musdb.html](https://sigsep.github.io/datasets/musdb.html) <br>
This dataset enables you to split music to different sources including Vocals, Drums, Bass, other.

# Usage

### 1. Import ModelAPI
```python
import DeepIsolation.API.Model as API
```

### 2. Get model
```python
api = API(n_gpu=1)
model = api.getModel()
```

### 3. Use Model
```python
model = model.to(device)
...
```
