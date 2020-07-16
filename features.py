import numpy as np

# number of channels of the dataset image, 3 for color jpg, 1 for grayscale img
# you need to change it to reflect your dataset
CHANNEL_NUM = 3
#%%
temp_feat=np.array([])

def cal_dir_stat(im):
    pixel_num=0
    channel_sum = np.zeros(CHANNEL_NUM)
    channel_sum_squared = np.zeros(CHANNEL_NUM)
    # image in M*N*CHANNEL_NUM shape, channel in BGR order
    im = im/255.0
    pixel_num += (im.size/CHANNEL_NUM)
    channel_sum += np.sum(im, axis=(0, 1))
    channel_sum_squared += np.sum(np.square(im), axis=(0, 1))

    bgr_mean = channel_sum / pixel_num
    bgr_std = np.sqrt(channel_sum_squared / pixel_num - np.square(bgr_mean))
    
    # change the format from bgr to rgb
    rgb_mean = list(bgr_mean)[::-1]
    rgb_std = list(bgr_std)[::-1]
    
    return rgb_mean, rgb_std


def get_histogram(img):
    #%%Get histogram for 1 channel
    # array with size of bins, set to zeros
    histFeat=np.array([])
    height = img.shape[0]
    width = img.shape[1]
    hist = np.zeros((256))
    for i in range(3):
        channel= img[:,:,i]
        for i in np.arange(height):
            for j in np.arange(width):
                channel[i,j]=channel[i,j]*255
                hist[int(channel[i,j])]+=1
        histFeat=np.append(histFeat, hist.ravel())
        histFeat=np.reshape(histFeat, (1,histFeat.shape[0])) #1x768
    return histFeat    
#1D array of bins (256)= frequencies
#%%
def calc_feat(img):
    #for whole features in 3 channels
    histIndex=256
    feat=np.array([])
    histImg= get_histogram(img)
    meanImg,stdImg= cal_dir_stat(img)
    for i in range(3):   
        histogram=histImg[0,histIndex*(i):histIndex*(i+1)]
        feat= np.hstack((feat,histogram))  if feat.size else histImg[0,:histIndex]
        feat= np.hstack((feat, meanImg[i]))
        feat= np.hstack((feat, stdImg[i]))
    feat=np.reshape(feat, (1,feat.shape[0]))
    return feat