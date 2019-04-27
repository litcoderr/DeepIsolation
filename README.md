# DeepIsolation
Deep isolation using DeepLabv3++ Segmentation Model

# Prior Research
- "Looking to Listen" by Google : [https://ai.googleblog.com/2018/04/looking-to-listen-audio-visual-speech.html](link)

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
