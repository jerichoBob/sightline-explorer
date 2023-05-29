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

matplotlib.use('Agg')

st.set_page_config(
    page_title="sightline-explorer",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        # 'Get Help': 'https://www.extremelycoolapp.com/help',
        # 'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# Sightline Explorer. CGM Exploration!"
    }
)
# Create file uploaders for the flux and variance data cubes
# flux_file = st.file_uploader("Upload flux data cube", type=['fits', 'fits.gz'])
# variance_file = st.file_uploader("Upload variance data cube", type=['fits', 'fits.gz'])
# to use this we would need to implement the streaming version of load_file found in kcwi_tools.
# so we go basic

st.title("CGM Sightline Explorer")

@st.cache_data
def load_fits_files(flux_filename,var_filename):
    # load flux and variance fits file
    print("locading again")
    base_path = "/Users/robertseaton/Desktop/Physics-NCState/---Research/FITS-data/J1429/"

    hdr, flux = kcwi_io.open_kcwi_cube(base_path+flux_filename)
    wave = kcwi_u.build_wave(hdr)
    _, var = kcwi_io.open_kcwi_cube(base_path+var_filename)
    return hdr, wave, flux, var

hdr, wave, flux, var = load_fits_files("J1429_rb_flux.fits","J1429_rb_var.fits")

# initialize all of our application state
if 'sightlines' not in st.session_state:
    st.session_state.sightlines = pd.DataFrame(columns=['x', 'y', 'radius', 'color', 'snr'])
if 'spectra' not in st.session_state:
    st.session_state.spectra = pd.DataFrame(columns=['Spectrum'])
if "points" not in st.session_state:
    st.session_state["points"] = []
if 'border_color' not in st.session_state:
    st.session_state.border_color = "#F70707"
if 'line_width' not in st.session_state:
    st.session_state.line_width = 1
if 'radius' not in st.session_state:
    st.session_state.radius = 1

# our sidebar
# image_scale = st.sidebar.slider('Image Scale', min_value=1, max_value=10, value=6, step=1)
brightness = st.sidebar.slider('Brightness', min_value=0.5, max_value=3.0, value=1.4, step=0.1)
contrast   = st.sidebar.slider('Contrast',   min_value=0.5, max_value=3.0, value=1.4, step=0.1)
sharpness  = st.sidebar.slider('Sharpness',  min_value=0.5, max_value=3.0, value=1.0, step=0.1)
band_width = 5
wavelength = st.sidebar.slider('Wavelength Center:', 3500, 5500, 4666)
st.sidebar.write("Wavelength Range:", wavelength - band_width,"-", wavelength + band_width)

image_area, sightlines_area, spectrum_area = st.columns([2, 1, 2])
image_scale = 6

def get_square_bounds(center, radius):
    return (
        center[0] - radius,
        center[1] - radius,
        center[0] + radius,
        center[1] + radius,
    )    

with image_area:
    image_area.subheader("Whitelight Image")
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


    wl_image_orig=build_whitelight(hdr, flux, minwave=wavelength - band_width, maxwave=wavelength + band_width)
    # utils.show_image_stats("BEFORE CORRECTIONS", wl_image) 
    wl_image = utils.make_image_corrections(wl_image_orig, contrast, brightness, sharpness, image_scale)
    # utils.show_image_stats("AFTER CORRECTIONS", wl_image)

    with wl_image:
        image_rgb = wl_image.convert('RGB')

        draw = ImageDraw.Draw(image_rgb)

        # draw all of the sightlines
        for index, s in st.session_state.sightlines.iterrows():
            bb = get_square_bounds([s.disp_x,s.disp_y], s.radius*image_scale)
            draw.rectangle(bb, outline =s.color, width=st.session_state.line_width)
            text = str(index)
            font = ImageFont.load_default()
            if s.radius <2:
                draw.text([bb[0],bb[1]], str(index), font = ImageFont.truetype("assets/Copilme-Regular.ttf", 3*image_scale), fill=s.color, anchor="rb")
            else:
                draw.text([bb[0]+1,bb[1]], str(index), font = ImageFont.truetype("assets/Copilme-Regular.ttf", 3*image_scale), fill=s.color, anchor="la")
        del draw # done with the draw variable

        value = streamlit_image_coordinates(image_rgb, key="pil")

        if value is not None:
            point = value["x"], value["y"]
            # st.write(point)


            if point not in st.session_state["points"]:  # if we need to add a new point
                st.session_state["points"].append(point)

                true_coords = utils.true_coordinates(value, image_scale, wl_image_orig)
                val = utils.display_coordinates(true_coords, image_scale, image_rgb)
                print(val)
                # image_area.write(true_coords)

                st.session_state.sightlines = utils.append_row(st.session_state.sightlines, 
                                                               Sightline(x=true_coords['x'], 
                                                                         y =true_coords['y'], 
                                                                         disp_x=value["x"], 
                                                                         disp_y=value["y"], 
                                                                         radius=st.session_state.radius, 
                                                                         color=st.session_state.border_color, 
                                                                         snr=0.))
                # image_area.write(st.session_state["points"])
                st.experimental_rerun()

with sightlines_area:
    sightlines_area.subheader("Sightlines")
    sightlines_area.dataframe(st.session_state.sightlines)

with spectrum_area:
    spectrum_area.subheader("Spectra")
    spectrum_area.table(st.session_state.spectra)