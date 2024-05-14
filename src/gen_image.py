import numpy as np
import matplotlib.pyplot as plt

def data2image(image_data):
    if len(image_data[0][0]) !=3:
        image_data = np.transpose(image_data, (1, 2, 0))
    print(image_data.shape)
    # 显示图像
    plt.imshow(image_data)
    plt.axis('off')  # 可以选择是否显示坐标轴
    plt.savefig('../results/savefig_example.png')
    return plt  

def get_images(data):
    for i in range(len(data)):
        plt = data2image(data[i])
        plt.savefig('../results/' + str(i) + '.png')