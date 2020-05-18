# Image Classification    
## CNN

* **Create CNN model**: Conv -> Max Pooling -> Conv -> Max Pooling -> Conv

  ```python
  self.model = models.Sequential([
              layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
              layers.MaxPooling2D((2, 2), strides=2),
              layers.Conv2D(64, (3, 3), activation='relu'),
              layers.MaxPooling2D((2, 2), strides=2),
              layers.Conv2D(64, (3, 3), activation='relu')
          ])
  ```

  
