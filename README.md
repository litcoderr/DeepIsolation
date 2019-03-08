# DeepIsolation
Deep isolation using DeepLabv3++ Segmentation Model

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