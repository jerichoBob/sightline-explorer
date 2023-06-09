import streamlit as st
# import streamlit.components.v1 as components
# from streamlit_drawable_canvas import st_canvas
from streamlit_image_coordinates import streamlit_image_coordinates
from PIL import Image, ImageDraw, ImageFont

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

from kcwitools import io as kcwi_io
from kcwitools import spec as kcwi_s
from kcwitools import utils as kcwi_u
from kcwitools.image import build_whitelight
import utils as utils
from utils import Sightline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.text import TextToPath
from matplotlib.font_manager import FontProperties
from matplotlib.gridspec import GridSpec

print("==========================================")
matplotlib.use('Agg')

st.set_page_config(
    page_title="sightline-explorer",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        # 'Get Help': 'https://www.extremelycoolapp.com/help',
        # 'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# Sightline Explorer. DLA and CGM Exploration!"
    }
)
# Create file uploaders for the flux and variance data cubes
# flux_file = st.file_uploader("Upload flux data cube", type=['fits', 'fits.gz'])
# variance_file = st.file_uploader("Upload variance data cube", type=['fits', 'fits.gz'])
# to use this we would need to implement the streaming version of load_file found in kcwi_tools.
# so we go basic

st.title("J1429 Sightline Explorer")

@st.cache_data
def load_fits_files(flux_filename,var_filename):
    # load flux and variance fits file
    print("locading again")
    base_path = "/Users/robertseaton/Desktop/Physics-NCState/---Research/FITS-data/J1429/"

    print("Reading flux cube")
    hdr, flux = kcwi_io.open_kcwi_cube(base_path+flux_filename)
    # utils.show_hdr("flux hdr", hdr)
    print("flux cube read")

    wave = kcwi_u.build_wave(hdr)
    print("Reading variance cube")
    _, var = kcwi_io.open_kcwi_cube(base_path+var_filename)
    print("variance cube")

    return hdr, wave, flux, var

hdr, wave, flux, var = load_fits_files("J1429_rb_flux.fits", "J1429_rb_var.fits")

# initialize application state
if 'sightlines' not in st.session_state:
    st.session_state.sightlines = pd.DataFrame(columns=['x', 'y', 'radius', 'color', 'snr'])
if 'sl_current' not in st.session_state:
    st.session_state.sl_current = None
if 'spectra' not in st.session_state:
    st.session_state.spectra = pd.DataFrame(columns=['Spectrum'])
if "pixel" not in st.session_state:
    st.session_state.pixel = None
if 'border_color' not in st.session_state:
    st.session_state.border_color = "#F70707"
if 'line_width' not in st.session_state:
    st.session_state.line_width = 1
if 'radius' not in st.session_state:
    st.session_state.radius = 1
if 'lap_counter' not in st.session_state:
    st.session_state.lap_counter = 1
else:
    st.session_state.lap_counter += 1
spectra = []
print("Lap Counter: ", st.session_state.lap_counter)


# our sidebar
# image_scale = st.sidebar.slider('Image Scale', min_value=1, max_value=10, value=6, step=1)
brightness = st.sidebar.slider('Brightness', min_value=0.5, max_value=3.0, value=1.4, step=0.1)
contrast   = st.sidebar.slider('Contrast',   min_value=0.5, max_value=3.0, value=1.4, step=0.1)
sharpness  = st.sidebar.slider('Sharpness',  min_value=0.5, max_value=3.0, value=1.0, step=0.1)
# band_width = 5
wavelength = st.sidebar.slider('Wavelength Center:', min_value=3500, max_value=5500, value=4666)
band_width  = st.sidebar.slider('Width', min_value=0, max_value=int(5500-3500/2), value=5, step=1)
st.sidebar.write("Wavelength Range:", wavelength - band_width,"-", wavelength + band_width)

def draw_aperature_expander():
    with st.expander("Aperture"):
        ap1,ap2,ap3 = st.columns([1,1,1])

        r = ap1.slider("Radius", min_value=1, max_value=10, value=st.session_state.radius, step=1, key='radius')
        fig, ax = plt.subplots()
        mask = kcwi_s.create_circular_mask(r)
        ax.imshow(mask, cmap='gray')
        fig.canvas.draw()
        image_data = np.array(fig.canvas.renderer.buffer_rgba())
        ap2.image(image_data, use_column_width=True)

        ap3.color_picker('Color', value=st.session_state.border_color, key='border_color')
        ap3.slider("Line Width",min_value=1,max_value=5, value=st.session_state.line_width, step=1, key='line_width')

aperture_area, one, two = st.columns([2, 1, 2])

image_area, spectrum_area = st.columns([1,2])
image_scale = 8

def handle_accept_button_click():
    if st.session_state.sl_current is not None:
        st.session_state.sightlines = utils.append_row(st.session_state.sightlines, st.session_state.sl_current)    
        st.session_state.sl_current = None

with image_area:
    image_area.subheader("Image")
    with aperture_area:
        draw_aperature_expander()

    wl_image_original=build_whitelight(hdr, flux, minwave=wavelength - band_width, maxwave=wavelength + band_width)
    # utils.show_image_stats("BEFORE CORRECTIONS", wl_image_original) 
    wl_orig_PIL, wl_image_display = utils.make_image_corrections(wl_image_original, contrast, brightness, sharpness, image_scale)
    # utils.show_image_stats("AFTER CORRECTIONS", wl_image_original)
    # print("main: wl_orig_PIL size: ", wl_orig_PIL.size, " wl_image_display size: ", wl_image_display.size)


    with wl_image_display:
        image_rgb = wl_image_display.convert('RGB')

        draw = ImageDraw.Draw(image_rgb)

        # draw all of the "permanent" sightline bounding boxes
        for index, s in st.session_state.sightlines.iterrows():
            draw = utils.draw_bbox(draw, s, index, image_scale, st.session_state.line_width)
        # draw the current stightline bounding box
        if st.session_state.sl_current is not None:
            s = st.session_state.sl_current
            index = len(st.session_state.sightlines)
            utils.draw_bbox(draw, s, index, image_scale, st.session_state.line_width)

        value = streamlit_image_coordinates(image_rgb, key="pil")

        st.button("Accept", on_click=handle_accept_button_click, type="primary")
        if value is not None:
            # image_coords = utils.image_coordinates(value, image_scale, wl_image_original)
            pixel = utils.display_to_image(value["x"],value["y"], image_scale, wl_image_display)
            # pixel = image_coords[0], image_coords[1]
            # st.write(pixel)

            # display_coords = utils.image_to_display(pixel[0], pixel[1], image_scale, wl_orig_PIL)
            display_coords = utils.image_to_display(pixel[0], pixel[1], image_scale, wl_image_display)
            print("display coords (raw): ", value)
            print(value, "->", pixel,"->", display_coords)

            if pixel != st.session_state.pixel:  # if we are now on a new pixel
                st.session_state.pixel = pixel
                st.session_state.sl_current = Sightline(x=pixel[0], 
                                                        y =pixel[1], 
                                                        disp_x=value["x"], 
                                                        disp_y=value["y"], 
                                                        radius=st.session_state.radius, 
                                                        color=st.session_state.border_color, 
                                                        label_alignment="la",
                                                        snr=0.)
                st.experimental_rerun() # this immediately forces a run-run through the loop so that the last thing entered here will be drawn - kinda janky, but works for now

flux_color = '#0055ff99'
error_color = '#5C5B5B'
mean_color = '#00ff3399'

def draw_text(ax, text, fontsize=18):
    ax.text(x=0.0, y=0.0, s=text,
            va="center", ha="center", 
            fontsize=fontsize, color="black")
@st.cache_data
def draw_spectrum(index, x, y, wave, flux, var, radius, color):
    extracted_spectrum = kcwi_s.extract_rectangle(x, y, wave, flux, var)

    # extracted_spectrum = kcwi_s.extract_circle(x, y, wave, flux, var, radius)
    sp_wave=extracted_spectrum.wavelength.value
    sp_flux=extracted_spectrum.flux.value
    sp_error=extracted_spectrum.sig.value

    mosaic_layout = '''LPPPPPPPPPP'''
    fig, ax = plt.subplot_mosaic(mosaic_layout, figsize=(12, 2))

    draw_text(ax['L'], str(index), fontsize=18)
    ax['L'].set_axis_off()
    

    ax["P"].plot(sp_wave,sp_flux,'-', color=color)
    ax["P"].plot(sp_wave,sp_error,'-', color=error_color)

    plt.xlim([3500,5500])
    ax["P"].autoscale(enable=True, axis='y')
    plt.xlabel('Wavelength (A)')
    plt.ylabel('Flux')
    return fig
    #..................................................
    # Define the range of interest
    # continuum_range = (3700, 5500)

    # Select the data within this range
    # mask = (wave >= continuum_range[0]) & (wave <= continuum_range[1])
    # selected_flux = flux[mask]

    # Compute the mean flux and its standard deviation within this range
    # mean_flux = np.mean(selected_flux, axis=0)
    # stddev_flux = np.std(selected_flux, axis=0)

    # Compute the SNR
    # snr = mean_flux / stddev_flux
    #..................................................
    # print("mean flux between 4700-4800: ", mean_flux)
    # print("stddev flux between 4700-4800: ", stddev_flux)
    # print("snr between 4700-4800: ", snr)

    # return

with image_area:
    image_area.subheader("Sightline Data")
    image_area.dataframe(st.session_state.sightlines)

with spectrum_area:
    spectrum_area.subheader("Spectra")
    
    cols = st.columns((1, 6))
    for index, s in st.session_state.sightlines.iterrows():
        cols[1].pyplot(draw_spectrum(index, s.x, s.y, wave, flux, var, s.radius, s.color))

    if st.session_state.sl_current is not None:
        s = st.session_state.sl_current
        index = len(st.session_state.sightlines)
        cols[1].pyplot(draw_spectrum(index, s.x, s.y, wave, flux, var, s.radius, s.color))
