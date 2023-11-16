import glob
import cv2
from matplotlib import pyplot as plt
import numpy as np
import os

def sort_key(item):
    key = item[0]  # Get the key from the (key, value) pair
    # Separate the alphabetic part from the numeric part
    alpha_part, num_part = "", ""
    for char in key:
        if char.isalpha():
            alpha_part += char
        elif char.isdigit():
            num_part += char
        else:
            # Handle non-alphanumeric characters if necessary
            pass
    
    # Convert the numeric part to an integer
    num_part = int(num_part)
    
    return alpha_part, num_part

def create_video_from_images(image_paths, output_video_path, frame_rate):
    # Load the first image to get its dimensions
    first_image = cv2.imread(image_paths[0])
    height, width, _ = first_image.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Use mp4v codec for MP4 format (other codecs are available)
    video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

    for image_path in image_paths:
        frame = cv2.imread(image_path)
        video_writer.write(frame)

    # Release the video writer and close the file
    video_writer.release()

def get_slice_for_sorting(filename):
    # Split the filename by "_"
    parts = filename.split("_")
    
    # Extract the slice you want to sort by (in this case, the second slice at index 1)
    slice_for_sorting = int(parts[-1].replace(".tif",""))
    
    # Convert the slice to an integer (assuming it contains numbers)
    return int(slice_for_sorting)

def find_order(target, string_list):
    try:
        order = string_list.index(target)
        return order
    except ValueError:
        return -1 

def subplot_from_image_list(image_list):
    fig, ax = plt.subplots(8, 12, figsize=(20, 20), facecolor="black")

    for i, path in enumerate(image_list):
        
        alpha = ["A", "B", "C", "D", "E", "F", "G", "H"]
        Well = path.split("\\")[-1].split("_")[0]

        number = [i for i in Well][1]
        letter = [i for i in Well][0]

        row = find_order(letter,alpha)[0]
        col = number

        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        ax[row,col].imshow(img)
        ax[row,col].axis('off')
        ax[row,col].text(20, 80, Well, color='white', fontsize=16,bbox=dict(facecolor='black', edgecolor='none'))
    fig.subplots_adjust(hspace= 0.1, wspace=0.0001)

def file_list_to_dict(file_list):
    file_dictionary = {}

    for i in file_list:
        well = i.split("\\")[-1].split("_")[0]
        if well in file_dictionary:
            file_dictionary[well].append(i)
        else:
            file_dictionary[well] = [i]
    final_sorted_dict = {}

    for i in file_dictionary:
        sorted_file_list = sorted(file_dictionary[i], key=get_slice_for_sorting)
        final_sorted_dict[i] = sorted_file_list
    sorted_dict = dict(sorted(final_sorted_dict.items(), key=sort_key))    
    return sorted_dict

def time_order_dict(listed_dict, multiplate = "no"):
    wells = [i for i in listed_dict]
    final_dict = {}

    if multiplate != "no":
        plate_order = {}
        for x in wells:
            image_list = listed_dict[x]
            
            for i in image_list:
                time = i.split("\\")[-1].split("_")[-1].replace(".tif","")
                plate = i.split("\\")[-3].split("_")[-1]
                if plate not in plate_order:
                    plate_order[plate] = {time:[i]}
                elif time not in plate_order[plate]:
                    plate_order[plate][time] = [i]
                else:
                    plate_order[plate][time].append(i)
        return plate_order

    else:
        for x in wells:
            image_list = listed_dict[x]
            time_order = {}
            for i in image_list:
                time = i.split("\\")[-1].split("_")[-1].replace(".tif","")
                plate = i.split("\\")[-2].split("_")[-1]
                if time not in time_order:
                        time_order[time] = [i]
                else:
                    time_order[time].append(i)
            final_dict[x] = time_order
    return final_dict


def resize_image(img):
    height, width = img.shape
    target_size = (int(width/4), int(height/4))
    image = cv2.resize(img, target_size)
    return image

def de_novo_kinetic_plot(image_list, height, width, out_name, GFP_images = "GFP"):

    fig, ax = plt.subplots(height, width, figsize=(width*4, height*3), facecolor="black")

    for num, path in enumerate(image_list):
        row = num // width
        col = num % height

        image1 = cv2.imread(path)
        image2 = cv2.imread(GFP_images[num])

        green_channel_image = np.zeros_like(image2)
        green_channel_image[:, :, 1] = image2[:, :, 1]

        alpha = 0.6
        beta = 0.4

        result = cv2.addWeighted(image1, alpha, green_channel_image, beta, 0.0)

        ax[row,col].imshow(result)
        ax[row,col].axis('off')

    output_directory = f"Results\\{out_name}"
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    file_num = len(glob.glob(f"Results\\{out_name}\\*JPG"))+1

    output_filename = f"Results\\{out_name}\\{out_name}_{str(file_num)}.JPG"

    if os.path.isfile(output_filename):
        print(f"File '{output_filename}' already exists. Choose a different name or delete the existing file.")
        return

    fig.savefig(output_filename, dpi=150, bbox_inches='tight')
    plt.close()



def plot_from_dict(out_name, path_dict, image_list, threshold="tmp"):
    import numpy as np
    variables = locals()
    high = (len(path_dict)// 12)
    fig, ax = plt.subplots(high, 12, figsize=(16, high), facecolor="black")

    for Well in path_dict:

        path = path_dict[Well][0]
        alpha = ["A", "B", "C", "D", "E", "F", "G", "H"]

        number = "".join([i for i in Well][1:])
        letter = [i for i in Well][0]

        row = int(find_order(letter,alpha))
        col = int(number)-1
        img = image_list[Well]
        image = resize_image(img)
        
        

        if threshold != "tmp":
            equalized_image = cv2.equalizeHist(img)
            _, thresholded_image = cv2.threshold(equalized_image, threshold, 255, cv2.THRESH_BINARY)
            ax[row,col].imshow(thresholded_image)
        elif "Bright Field" in path:
            equalized_image = cv2.equalizeHist(img)
            ax[row,col].imshow(equalized_image , cmap="gray")
            #print("GREY")
        else:
            ax[row,col].imshow(image)
            #print(path)
        

 
        
        #bg_subtractor = cv2.createBackgroundSubtractorMOG2()
        #fg_mask = bg_subtractor.apply(img)

        #equalized_image = cv2.equalizeHist(img)

        
        #mean = round(cv2.mean(equalized_image)[0], 2)
        #window_size = 5
        #std = round(cv2.GaussianBlur(img, (window_size, window_size), 0).std(),2)
        
        #per = "{:.2f}".format(100*(1-(mean[0]/255)))

        #image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        #image[np.where(img == 255)] = [255, 0, 0] 
        
        #ax[row,col].imshow(image)

        ax[row,col].axis('off')
        #ax[row,col].text(20, 80,  str(mean) + "\n" + str(std) , color='white', fontsize=8,bbox=dict(facecolor='black', edgecolor='none'))
    files = glob.glob(f"Results\\*.JPG")
    file_num = len(files)+2
    fig.subplots_adjust(hspace= 0.1, wspace=0.1)
    fig.savefig(f"Results\\{out_name}_{str(file_num)}.JPG", dpi=150, bbox_inches='tight')

def open_images_to_(mylist):
    return [cv2.imread(path,cv2.IMREAD_GRAYSCALE) for path in mylist]

def open_images_to_dict(mydict):
    img_dict = {}
    for Well, path in mydict.items():
            img_dict[Well] = cv2.imread(path[0],cv2.IMREAD_GRAYSCALE)
    return img_dict


"""
Typical Usage

BF_images = [i for i in file_list if i.split("_")[-2] == "Bright Field"]
dict_list = file_list_to_dict(BF_images)

GFP_images = [i for i in file_list if i.split("_")[-2] == "GFP"]
RFP_images = [i for i in file_list if i.split("_")[-2] == "Texas Red"]
BF_images = [i for i in file_list if i.split("_")[-2] == "Bright Field"]

time_order_BF = time_order_dict(file_list_to_dict(BF_images))

GFP_dict = file_list_to_dict(GFP_images)
RFP_dict = file_list_to_dict(RFP_images)
"""