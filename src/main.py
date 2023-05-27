import streamlit as st
# import streamlit.components.v1 as components
from streamlit_drawable_canvas import st_canvas
from streamlit_image_coordinates import streamlit_image_coordinates

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

matplotlib.use('Agg')

st.set_page_config(
    page_title="sightline-explorer",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)
# Create file uploaders for the flux and variance data cubes
# flux_file = st.file_uploader("Upload flux data cube", type=['fits', 'fits.gz'])
# variance_file = st.file_uploader("Upload variance data cube", type=['fits', 'fits.gz'])
# to use this we would need to implement the streaming version of load_file found in kcwi_tools.
# so we go basic


@st.cache_data
def load_fits_files(flux_filename,var_filename):
    # load flux and variance fits file
    base_path = "/Users/robertseaton/Desktop/Physics-NCState/---Research/FITS-data/J1429/"

    hdr, flux = kcwi_io.open_kcwi_cube(base_path+flux_filename)
    wave = kcwi_u.build_wave(hdr)
    _, var = kcwi_io.open_kcwi_cube(base_path+var_filename)
    return hdr, wave, flux, var

hdr, wave, flux, var = load_fits_files("J1429_rb_flux.fits","J1429_rb_var.fits")


# print("-----")
# image_width = hdr["NAXIS1"]
# image_height = hdr["NAXIS2"]
# print("w: ", image_width, " h: ", image_height)
# for i in hdr:
#     print(i,": ", hdr[i])

# Create a figure and axes
# fig, ax = plt.subplots()
# ax.imshow(wl_image,interpolation="nearest",cmap="gray",vmin=0)

if 'sightline' not in st.session_state:
    st.session_state.sightline = pd.DataFrame(columns=['x','y','SNR'])
if 'spectra' not in st.session_state:
    st.session_state.spectra = pd.DataFrame(columns=['Spectrum', 'SNR'])

# how about a little layout
brightness = st.sidebar.slider('Brightness', min_value=0.5, max_value=3.0, value=1.4, step=0.1)
contrast   = st.sidebar.slider('Contrast',   min_value=0.5, max_value=3.0, value=1.4, step=0.1)
sharpness  = st.sidebar.slider('Sharpness',  min_value=0.5, max_value=3.0, value=1.0, step=0.1)

image_area, sightlines_area, spectrum_area = st.columns([2, 1, 2])
image_scale = 6

with image_area:
    image_area.write("Flux")
    band_width = 5
    wavelength = image_area.slider('Center Wavelength:', 3500, 5500, 4666, label_visibility="collapsed")

    wl_image_orig=build_whitelight(hdr, flux, minwave=wavelength - band_width, maxwave=wavelength + band_width)
    # utils.show_image_stats("BEFORE CORRECTIONS", wl_image) 
    wl_image = utils.make_image_corrections(wl_image_orig, contrast, brightness, sharpness, image_scale)
    # utils.show_image_stats("AFTER CORRECTIONS", wl_image)

   
   #    image_area.pyplot(fig)
    coords = streamlit_image_coordinates(wl_image)

    if coords is not None:
        coords = utils.rescale_coordinates(coords, image_scale, True, wl_image_orig)
        image_area.write(coords)
        st.session_state.sightline = utils.append_row(st.session_state.sightline, {'x': coords['x'],
                                                                                   'y': coords['y'],
                                                                                   'SNR': 0.00  })
        print("just after adding a new entry: ", st.session_state.sightline)

with sightlines_area:
    sightlines_area.write("Sightlines")
    sightlines_area.table(st.session_state.sightline)

with spectrum_area:
    spectrum_area.write("Spectra")
    spectrum_area.table(st.session_state.spectra)



