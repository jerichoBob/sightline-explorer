import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageDraw
import matplotlib.pyplot as plt

from base64 import b64encode
from io import BytesIO
from IPython.display import HTML


from dataclasses import dataclass


def show_image_stats(label, wl_image):
    print(f"---- {label} ------")
    # Number of elements in the image
    num_elements = np.prod(wl_image.shape)

    # Number of NaN values
    num_nan = np.sum(np.isnan(wl_image))

    # Number of negative values
    num_negative = np.sum(wl_image < 0)

    # Minimum and maximum values
    min_val = np.nanmin(wl_image)
    max_val = np.nanmax(wl_image)

    # Mean and standard deviation
    mean_val = np.nanmean(wl_image)
    std_val = np.nanstd(wl_image)

    print("---- BEFORE CORRECTIONS ------")
    print(f"Number of elements: {num_elements}")
    print(f"Number of NaN values: {num_nan}")
    print(f"Number of negative values: {num_negative}")
    print(f"Min value: {min_val}")
    print(f"Max value: {max_val}")
    print(f"Mean value: {mean_val}")
    print(f"Standard deviation: {std_val}")

def make_image_corrections(wl_image, contrast, brightness, sharpness, scale_factor):
    # Replace NaN values with 0
    wl_image = np.nan_to_num(wl_image)

    # Scale the data to the range 0-1
    wl_image = (wl_image - np.min(wl_image)) / (np.max(wl_image) - np.min(wl_image))

    # Flip the y-axis
    wl_image = np.flipud(wl_image)

    # Invert and convert to PIL image
    wl_image = Image.fromarray((wl_image * 255).astype(np.uint8))
    # wl_image = Image.fromarray((wl_image * 255).astype('uint8'), 'RGB')

    # original size
    original_size = wl_image.size

    # new size
    new_size = [dimension * scale_factor for dimension in original_size]

    # resize the image
    wl_image = wl_image.resize(new_size, Image.NEAREST)

    # Adjust contrast
    enhancer = ImageEnhance.Contrast(wl_image)
    wl_image = enhancer.enhance(contrast)

    # Adjust brightness
    enhancer = ImageEnhance.Brightness(wl_image)
    wl_image = enhancer.enhance(brightness)   

    # Adjust brightness
    enhancer = ImageEnhance.Sharpness(wl_image)
    wl_image = enhancer.enhance(sharpness)  

    return wl_image

def true_coordinates(coords, s, orig_image):
    """
    Rescale (x, y) coordinates from the displayed image back to the original image.

    Parameters:
    coords (dict): The dictionary containing 'x' and 'y' coordinates on the displayed image.
    s (float): The scaling factor used to resize the original image for display.
    flippedy (bool): Indicates whether the original image was vertically flipped for display.
    original_shape (tuple): The shape of the original image (height, width).

    Returns:
    dict: The dictionary with the rescaled 'x' and 'y' coordinates on the original image.
    """
    original_shape = np.shape(orig_image)
    # Calculate the original coordinates
    x_original = int(coords['x'] / s) + 1
    y_original = int(coords['y'] / s) + 1

    # Adjust y-coordinate if the image was vertically flipped
    y_original = original_shape[0] - y_original

    return {'x': x_original, 'y': y_original}

def display_coordinates(coords, s, orig_image):
    """
    Scale (x, y) coordinates from the original image to the displayed image.

    Parameters:
    coords (dict): The dictionary containing 'x' and 'y' coordinates on the original image.
    s (float): The scaling factor used to resize the original image for display.
    orig_image (Image): The original PIL image.

    Returns:
    dict: The dictionary with the rescaled 'x' and 'y' coordinates on the displayed image.
    """
    original_shape = np.shape(orig_image)
    
    # Calculate the display coordinates
    x_display = int(coords['x'] * s)
    y_display = original_shape[0] - int(coords['y'] * s)

    return {'x': x_display, 'y': y_display}


# handline dataframes

# add/append a row to the specified dataframe 
def append_row(df, new_row):
    new_row_df = pd.DataFrame([new_row])
    df = pd.concat([df, new_row_df], ignore_index=True)
    return df

# remove a row from the specified dataframe using the provided index
def remove_row_with_index(df, index):
    df = df.drop(index)
    return df

# remove a row (or rows) from the specified dataframe using a column label and a key value
def remove_row_with_key(df, column, key):
    df = df[df[column] != key]
    return df


def create_sparkline(data, **kwags):
    '''
    from: https://towardsdatascience.com/6-things-that-you-probably-didnt-know-you-could-do-with-pandas-d365b3362a55
    with this you can put a formatted HTML sparkline into a df table like this:
        df['Price History Line']  = df['Price History'].apply(create_sparkline)
        HTML(df.drop(columns = ["Price History"]).to_html(escape=False))

    '''
    # Convert data to a list
    data = list(data)
    
    # Create a figure and axis object with given size and keyword arguments
    fig, ax = plt.subplots(1, 1, figsize=(3, 0.25), **kwags)
    
    # Plot the data
    ax.plot(data)
    
    # Remove the spines from the plot
    for k,v in ax.spines.items():
        v.set_visible(False)
        
    # Remove the tick marks from the x and y axes
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Plot a red dot at the last point of the data
    plt.plot(len(data) - 1, data[len(data) - 1], 'r.')
    
    # Fill the area under the plot with alpha=0.1
    ax.fill_between(range(len(data)), data, len(data)*[min(data)], alpha=0.1)
    
    # Close the plot to prevent displaying it
    plt.close(fig)
    
    # Save the plot image as png and get its binary data
    img = BytesIO()    
    fig.savefig(img, format='png')
    encoded = b64encode(img.getvalue()).decode('utf-8')  
    
    # Return the encoded image data as an HTML image tag
    return '<img src="data:image/png;base64,{}"/>'.format(encoded)

@dataclass
class Sightline:
    """Class for managing Sightlines"""
    x: int # true (unscaled) coordinate x
    y: int # true (unscaled) coordinate x
    disp_x: int # display coordinate x
    disp_y: int # display coordinate y
    radius: int = 1
    color: str = "#f70707"
    label_alignment: str = "la" #https://pillow.readthedocs.io/en/stable/handbook/text-anchors.html#text-anchors
    snr: float = 0.

    def get_x(self):
        return self.x
    def get_y(self):
        return self.y
    
    def get_disp_x(self):
        return self.disp_x
    def get_disp_y(self):
        return self.disp_y    
    

    def get_radius(self):
        return self.radius
    
    def set_radius(self, value): 
        self.radius = value

    def get_color(self):
        return self.color
    
    def set_color(self, value): 
        self.color = value

    def get_snr(self):
        return self.snr
    
    def set_snr(self, value): 
        self.snr = value

    

# This just draws a box centered at (x,y) and sz from that center point in the n/e/s/w directions 
def plotbox(plt, x, y, labels, align, sz, c):
    for i in range(len(x)):
        plt.plot(
            [x[i]-sz, x[i]-sz, x[i]+sz, x[i]+sz, x[i]-sz], 
            [y[i]-sz, y[i]+sz, y[i]+sz, y[i]-sz, y[i]-sz], 
            '-', color=c)
        # ha_ = align[i][0]
        # va_ = align[i][1]
        plt.text(x[i], y[i]+1.5*sz, labels[i], color=c);    

def overlay_bbox(image, points, r, c):
    # Ensure we're working with a copy of the image, not the original
    image_copy = image.copy()

    # Create a draw object
    draw = ImageDraw.Draw(image_copy)

    # Loop through the points
    for x, y in points:
        # Draw a rectangle (a bounding box) centered on x, y
        box = [(x - r, y - r), (x + r, y + r)]
        draw.rectangle(box, outline=c, width=2)

    return image_copy
