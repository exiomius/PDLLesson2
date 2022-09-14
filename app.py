#!/usr/bin/env python
# coding: utf-8

# ! [ -e /content ] && pip install -Uqq fastbook
import fastbook
fastbook.setup_book()
#hide
from fastbook import *
from fastai.vision.widgets import *


# In[3]:


#|export

import gradio as gr

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

def is_cat(x): return x[0].isupper() # Presquite code for the model to run

learn_inf = load_learner('model.pkl')

learn_inf.dls.vocab # Reminds us of the categories

categories = learn_inf.dls.vocab
def classify_image(img):
    pred, idx, probs = learn_inf.predict(img)
    return dict(zip(categories, map(float,probs)))

image = gr.inputs.Image(shape=(192,192)) 
label = gr.outputs.Label()
examples = ["Example 1.jpeg","Example 2.jpeg"]

intf = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples)
intf.launch(inline=False)

