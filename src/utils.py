import numpy as np
import pandas as pd
import PIL
from PIL import Image, ImageEnhance, ImageDraw, ImageFont
import matplotlib.pyplot as plt

from base64 import b64encode
from io import BytesIO
from IPython.display import HTML


from dataclasses import dataclass

def show_hdr(label, hdr):
    print(f"---- {label} ------")
    for i in hdr:
        print(i,": ", hdr[i])

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
    cleansed_image = np.nan_to_num(wl_image)

    # Scale the data to the range 0-1
    cleansed_image = (cleansed_image - np.min(cleansed_image)) / (np.max(cleansed_image) - np.min(cleansed_image))

    # Flip the y-axis
    # cleansed_image = np.flipud(cleansed_image)

    # Invert and convert to PIL image
    cleansed_image = Image.fromarray((cleansed_image * 255).astype(np.uint8))
    # cleansed_image = Image.fromarray((cleansed_image * 255).astype('uint8'), 'RGB')

    # original size
    original_size = cleansed_image.size

    # new size
    new_size = [dimension * scale_factor for dimension in original_size]
    # print("orig size: ", original_size, " new size: ", new_size)

    # resize the image
    enhanced_image = cleansed_image.resize(new_size, Image.NEAREST)
    
    # Flip the y-axis
    enhanced_image = enhanced_image.transpose(PIL.Image.FLIP_TOP_BOTTOM)

    # Adjust contrast
    enhancer = ImageEnhance.Contrast(enhanced_image)
    enhanced_image = enhancer.enhance(contrast)

    # Adjust brightness
    enhancer = ImageEnhance.Brightness(enhanced_image)
    enhanced_image = enhancer.enhance(brightness)   

    # Adjust brightness
    enhancer = ImageEnhance.Sharpness(enhanced_image)
    enhanced_image = enhancer.enhance(sharpness)  

    return cleansed_image, enhanced_image


def display_to_image(x, y, image_scale, wl_image_display):
    """
    Convert display coordinates to original image coordinates.
    """
    # original_x = int(round(x / image_scale)) + 1
    # original_y = int(round((wl_image_display.height - y) / image_scale)) + 1
    original_x = int(round(x / image_scale)) + 1
    original_y = int(round((wl_image_display.height - y) / image_scale)) + 1

    return original_x, original_y

def image_to_display(x, y, image_scale, wl_image_original):
    """
    Convert original image coordinates to display coordinates.
    """
    # original_shape = np.shape(wl_image_original)

    # display_x = int(round((x - 1) * image_scale))
    # display_y = int(round((original_shape[0] - y + 1) * image_scale))
    display_x = int(round((x - 1) * image_scale))
    display_y = wl_image_original.height - int(round((y - 1) * image_scale))

    return display_x, display_y

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
def create_sparkline_table():
    '''
    unfortunately only works inside a jupyter notebook
    '''

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

def get_square_bounds(center, radius):
    return (
        center[0] - radius,
        center[1] - radius,
        center[0] + radius,
        center[1] + radius,
    )    
    
def draw_bbox(draw,s, index, scale, line_width):
    bb = get_square_bounds([s.disp_x,s.disp_y], s.radius*scale)
    draw.rectangle(bb, outline =s.color, width=line_width)
    font = ImageFont.truetype("assets/Copilme-Regular.ttf", 3*scale)
    if s.radius <2:
        draw.text([bb[0],bb[1]], str(index), font = font, fill=s.color, anchor="rb")
    else:
        draw.text([bb[0]+1,bb[1]], str(index), font = font, fill=s.color, anchor="la")
    return draw



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
