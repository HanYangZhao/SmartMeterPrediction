# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 03:24:07 2018

@author: han
"""

import numpy as np
import requests
import json 
r = requests.get('http://localhost:5000/getTwoDaysUsage')
x = np.array(json.loads((r.text)))

r_1 = requests.get('http://localhost:5000/getData')
x_1 = json.loads((r_1.text))