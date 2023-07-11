import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import scipy.integrate as spi
import cv2
from argparse import ArgumentParser

def parse_args(): 
    parser = ArgumentParser()
    parser.add_argument("img_path", help="Path to the image.")
    parser.add_argument("data_path", help="Path to the dynamics data, should be ended with the file name file.xlsx")
    args = parser.parse_args()
    return args

def main(args): 
    # print(args)
    # Load the workbook
    # path = "../../../../../oct_data/dynamic_model.xlsx"
    # img_path = "../../../../../oct_data/mouse_esophagus/00010.oct.png"
    # Load the data into a pandas DataFrame
    print("Loading data...")
    data = pd.read_excel(args.data_path, sheet_name='Sheet1')
    # Extract the x and y data
    time = data.iloc[::10, 1].astype(float)
    angular = data.iloc[::10, 2].astype(float)
    print("Finish loading.")
    # Rescale the time so it fits the image dimension
    time = time - 0.1
    # Below should be time * img_width*10
    time = time * 62380
    y_tens = np.zeros(104858) +30
    print("Displaying reference dynamics, press q to continue.")
    plt.plot(time, angular, color='blue')
    plt.plot(time, y_tens, color='red')
    plt.xlabel('time')
    plt.ylabel('Angular velocity and frequency')
    plt.title("Dynamics and sampling frequency at 30 Hz.")
    plt.xlim(0, 6238)
    plt.ylim(0,35)
    plt.show()

    interp_func = interp1d(time, angular, kind='cubic')
    angular_pred = interp_func(time)

    def simulated_func(x):
        return interp_func(x)

    cv_img = cv2.imread(args.img_path)
    cv_img2 = []

    print("Image size: ", cv_img.shape)

    end = cv_img.shape[1]
    for j in range(end):
        if (j)%10 != 0 :
            continue
        print(f"Creating image {int(100*j/6238)}%/100% ")
        # compute the integration area
        result2, _ = spi.quad(simulated_func, j, j+10) if j < int(end/10)*10 else spi.quad(simulated_func, j, end)
        scale_factor = (300)/result2 if j < int(end/10)*10 else (30*(end-j))/result2
        # print("scale_factor ",scale_factor)
        # scale_factor = scale_factor*2
        subregion = (int(j),0,int(j+10), 1024)
        # Crop the subregion from the original image
        cropped_image = cv_img[:, subregion[0]:subregion[2]] if j < int(end/10)*10 else cv_img[:, j:]
        # Calculate the new width and height after downsampling
        # Define the scale factor for downsampling
        new_width = int(cropped_image.shape[1]*scale_factor)
        new_height = int(cropped_image.shape[0])
        # Perform Bilinear interpolation to rescale the subregion
        rescale_subregion = cv2.resize(cropped_image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        cv_img2 = np.concatenate((cv_img2, rescale_subregion), axis=1) if cv_img2 != [] else rescale_subregion

    # Resize the image to original size
    resized_img = cv2.resize(cv_img2, (cv_img.shape[1], cv_img.shape[0]), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite("./result.png", cv_img2)
    cv2.imwrite("./result_rescale.png", resized_img)
    print("Image stored at ./result.png and ./result_rescaled.png")

if __name__ == "__main__":
    args = parse_args()
    main(args)