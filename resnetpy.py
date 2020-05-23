#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from keras.applications import resnet50


# In[ ]:


model = resnet50.ResNet50(weights='imagenet')


# In[ ]:


model.layers[0].input


# In[ ]:


from keras.preprocessing import image


# In[ ]:


img = image.load_img('cat_or_dog_1.jpg', target_size=(224, 224))


# In[ ]:


img


# In[ ]:


img.size


# In[ ]:


type(img)


# In[ ]:


img_np = image.img_to_array(img)


# In[ ]:


img_np.shape


# In[ ]:


type(img_np)


# In[ ]:


import numpy as np


# In[ ]:


a = np.array([1,2])


# In[ ]:


a.shape


# In[ ]:


b = np.expand_dims(a, axis=0)


# In[ ]:


b.shape


# In[ ]:


b[0]


# In[ ]:


ae = np.expand_dims(img_np, axis=0)


# In[ ]:


ae.shape


# In[ ]:


from keras.applications.resnet50 import decode_predictions


# In[ ]:


from keras.applications.resnet50 import preprocess_input


# In[ ]:


finalimg = preprocess_input(ae)


# In[ ]:





# In[ ]:


pred = model.predict(finalimg)


# In[ ]:


decode_predictions(pred, top=3)[0]


# In[ ]:




