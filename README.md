# DeepIsolation
Deep isolation using DeepLabv3++ Segmentation Model

# Usage
Warning: This part is in development process


### 1. Import ModelAPI
```python
import DeepIsolation.API.Model as modelAPI
```

### 2. Get model by calling api.getModel
```python
api = modelAPI()
model = api.getModel()
```

### 3. Use Model for Back ground Music Isolation
```python
model = model.to(device)
...
```