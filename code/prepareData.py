import numpy as np

def prepareData(imArray, ambientImage):
    imArray -= np.expand_dims(ambientImage, axis=-1)
    imArray[imArray < 0] = 0
    return imArray / np.max(imArray)

