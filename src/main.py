import streamlit as st
# import streamlit.components.v1 as components
# from streamlit_drawable_canvas import st_canvas
from streamlit_image_coordinates import streamlit_image_coordinates
from PIL import Image, ImageDraw, ImageFont

import matplotlib
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import os, sys

from kcwitools import io as kcwi_io
from kcwitools import spec as kcwi_s
from kcwitools import extract_weighted_spectrum as kcwi_ews
from kcwitools import utils as kcwi_u
from kcwitools.image import build_whitelight

import utils as utils
from utils import Sightline

dir_path = os.path.dirname(os.path.realpath(__file__))
bu_path = os.path.abspath(os.path.join(dir_path, "../CGM-learning/code"))
sys.path.append(bu_path)

from bobutils import utils as bu

# some global variables -- we'll deal with that later 
# should these globals be put into session state? 
#   and should we have a button which allows us to load, save and clear session state?
hdr = None
wave = None
flux = None
var = None

# a workaround/hack for st.file-uploader()
var_tmp_filename = "tmp/var.fits"
flux_tmp_filename = "tmp/flux.fits"

spectra = []

flux_color = '#0731F7'
error_color = '#5C5B5B'
mean_color = '#00ff3399'

def init_page():
    st.set_page_config(
        page_title="sightline-explorer",
        page_icon="🧊",
        layout="wide",
        initial_sidebar_state="collapsed",
        menu_items={
            # 'Get Help': 'https://www.extremelycoolapp.com/help',
            # 'Report a bug': "https://www.extremelycoolapp.com/bug",
            'About': "# Sightline Explorer. CGM AND DLA Exploration!"
        }
    )
    st.title("J1429 Sightline Explorer")

def init_graphics():
    matplotlib.use('Agg')

def init_session_state():
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
    if 'rerun_active' not in st.session_state:
        st.session_state.rerun_active = False
    if 'image_controls' not in st.session_state:
        st.session_state.image_controls = {
            'image_scale': 6,
            'brightness': 1.4,
            'contrast': 1.4,
            'sharpness': 1.0,
            'wavelength': 4666,
            'band_width': 5,
        }
    if 'fits_orig' not in st.session_state:
        st.session_state.fits_orig = {
            'hdr': None,
            'wave': None,
            'flux': None,
            'var': None
        }

    print("Lap Counter: ", st.session_state.lap_counter)


@st.cache_data
def load_fits_files(flux_filename,var_filename):
    # load flux and variance fits file
    print("load_fits_files")

    print("Reading flux cube: ", flux_filename)
    # hdr, flux = kcwi_io.open_kcwi_cube(base_path+flux_filename)
    hdr, flux = kcwi_io.open_kcwi_cube(flux_filename)
    
    utils.show_hdr("flux hdr", hdr)

    # NOTE: doctor the header if needed
    print("flux cube read")
    if 'CDELT3' in hdr.keys():
        hdr['CD3_3'] = hdr['CDELT3']

    wave = kcwi_u.build_wave(hdr)
    print("Reading variance cube: ", var_filename)
    _, var = kcwi_io.open_kcwi_cube(var_filename)
    print("variance cube")

    # update session state
    st.session_state.fits_orig['hdr'] = hdr
    st.session_state.fits_orig['wave'] = wave
    st.session_state.fits_orig['flux'] = flux
    st.session_state.fits_orig['var'] = var
    print(f"st.session_state.fits_orig['hdr']:{st.session_state.fits_orig['hdr']}")
    print(f"st.session_state.fits_orig['wave']:{st.session_state.fits_orig['wave']}")
    print(f"st.session_state.fits_orig['flux']:{st.session_state.fits_orig['flux']}")
    print(f"st.session_state.fits_orig['var']:{st.session_state.fits_orig['var']}")



def init_sidebar():
    flux_file = st.sidebar.file_uploader("Select your Flux FITS file", type=['fits', 'fits.gz'])

    if flux_file is not None:
        print("flux file_details: ", flux_file.name, flux_file.type, flux_file.size)
        fp_flux = open(flux_tmp_filename, "wb")
        fp_flux.write(flux_file.getvalue())
        fp_flux.close()
        flux_file.close()
        # print(flux_file)

    variance_file = st.sidebar.file_uploader("Select your Variance FITS file", type=['fits', 'fits.gz'])
    if variance_file is not None:
        print("variance file_details: ", variance_file.name, variance_file.type, variance_file.size)
        fp_var = open(var_tmp_filename, "wb")
        fp_var.write(variance_file.getvalue())
        fp_var.close()
        variance_file.close()

    if flux_file is not None and variance_file is not None:
        print("Loading FITS files")
        load_fits_files(flux_tmp_filename, var_tmp_filename)
        print("FITS files loaded")
        flux_file.close()
        variance_file.close()
        os.remove(flux_tmp_filename)
        os.remove(var_tmp_filename)
    isv = st.session_state.image_controls['image_scale']
    bv = st.session_state.image_controls['brightness']
    cv = st.session_state.image_controls['contrast']
    sv = st.session_state.image_controls['sharpness']
    wv = st.session_state.image_controls['wavelength']
    bwv = st.session_state.image_controls['band_width']

    st.session_state.image_controls['image_scale'] = st.sidebar.slider('Image Scale', min_value=1, max_value=10, value=isv, step=1)
    st.session_state.image_controls['brightness'] = st.sidebar.slider('Brightness', min_value=0.5, max_value=3.0, value=bv, step=0.1)
    st.session_state.image_controls['contrast'] = st.sidebar.slider('Contrast',   min_value=0.5, max_value=3.0, value=cv, step=0.1)
    st.session_state.image_controls['sharpness'] = st.sidebar.slider('Sharpness',  min_value=0.5, max_value=3.0, value=sv, step=0.1)
    # band_width = 5
    wavelength = st.sidebar.slider('Wavelength Center:', min_value=3500, max_value=5500, value=wv)
    band_width  = st.sidebar.slider('Width', min_value=0, max_value=int(5500-3500/2), value=bwv, step=1)
    st.sidebar.write("Wavelength Range:", wavelength - band_width,"-", wavelength + band_width)
    st.session_state.image_controls['wavelength'] = wavelength
    st.session_state.image_controls['band_width']  = band_width

def create_circular_mask(r):
    """
    Creates a circular mask of radius r and normalizes that mask
    """
    n = 2*r + 1  # size of the output array
    center = r  # center of the circular mask

    # create a n x n array filled with the Euclidean distance from the center
    y, x = np.ogrid[-center:n-center, -center:n-center]
    mask = np.sqrt(x**2 + y**2)

    mask = 1 - mask / (r+0.75)
    mask[mask < 0] = 0 # make sure we don't have any negative values in the mask
    mask /= np.sum(mask)  # Divide by the sum of all values

    return mask

def draw_aperature_expander():
    with st.expander("Aperture"):
        ap1,ap2,ap3 = st.columns([1,1,1])

        r = ap1.slider("Radius", min_value=1, max_value=10, value=st.session_state.radius, step=1, key='radius')
        fig, ax = plt.subplots()
        mask = create_circular_mask(r)
        ax.imshow(mask, cmap='gray')
        fig.canvas.draw()
        image_data = np.array(fig.canvas.renderer.buffer_rgba())
        ap2.image(image_data, use_column_width=True)

        ap3.color_picker('Color', value=st.session_state.border_color, key='border_color')
        ap3.slider("Line Width",min_value=1,max_value=5, value=st.session_state.line_width, step=1, key='line_width')

def handle_accept_button_click():
    if st.session_state.sl_current is not None:
        st.session_state.sightlines = utils.append_row(st.session_state.sightlines, st.session_state.sl_current)    
        st.session_state.sl_current = None

def draw_text(ax, text, fontsize=18):
    ax.text(x=0.0, y=0.0, s=text,
            va="center", ha="center", 
            fontsize=fontsize, color="black")

@st.cache_data
def draw_spectrum(index, x, y, wave, flux, var, radius, color):
    spec = kcwi_s.extract_rectangle(x, y, wave, flux, var)
    # spec = kcwi_ews.extract_weighted_spectrum(flux,var,wave,verbose=False,weights='Data',porder=9)    

    # spec = kcwi_s.extract_circle(x, y, wave, flux, var, radius)
    sp_wave=spec.wavelength.value
    sp_flux=spec.flux.value
    sp_error=spec.sig.value

    mosaic_layout = '''LPPPPPPPPPP'''
    fig, ax = plt.subplot_mosaic(mosaic_layout, figsize=(12, 2))

    draw_text(ax['L'], str(index), fontsize=18)
    ax['L'].set_axis_off()

    lw = 0.5
    ax["P"].plot(sp_wave,sp_flux,'-', color=color, lw=lw, label="Flux")
    ax["P"].plot(sp_wave,sp_error,'-', color=error_color, lw=lw, label="Error")

    plt.xlim([3500,5500])
    ax["P"].autoscale(enable=True, axis='y')
    plt.xlabel('Wavelength (A)')
    plt.ylabel('Flux')
    return fig


def show_main_area():
    aperture_area, one, two = st.columns([2, 1, 2])
    image_area, spectrum_area = st.columns([1,2])
    wl_image_display = None

    hdr = st.session_state.fits_orig['hdr']
    # print("hdr: ", hdr)
    flux = st.session_state.fits_orig['flux']

    with image_area:
        image_area.subheader("Image")
        with aperture_area:
            draw_aperature_expander()
        
        
        if hdr is not None and flux is not None:    
            contrast = st.session_state.image_controls['contrast']
            brightness = st.session_state.image_controls['brightness']
            sharpness = st.session_state.image_controls['sharpness']
            wavelength = st.session_state.image_controls['wavelength']
            band_width = st.session_state.image_controls['band_width']
            image_scale = st.session_state.image_controls['image_scale']

            wl_image_original=build_whitelight(hdr, flux, minwave=wavelength - band_width, maxwave=wavelength + band_width)
            # print("wl_image_original: ", wl_image_original)
            _, wl_image_display = utils.make_image_corrections(wl_image_original, contrast, brightness, sharpness, image_scale)
            # print("wl_image_display: ", wl_image_display)


        if wl_image_display is not None:
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

                display_coords = utils.image_to_display(pixel[0], pixel[1], image_scale, wl_image_display)
                print("display coords (raw): ", value)
                print(value, "->", pixel,"->", display_coords)

                if pixel != st.session_state.pixel:  # if we are now on a new pixel
                    st.session_state.pixel = pixel
                    st.session_state.sl_current = Sightline(x=pixel[0], 
                                                            y =pixel[1], 
                                                            disp_x=display_coords[0], 
                                                            disp_y=display_coords[1],                                                         
                                                            radius=st.session_state.radius, 
                                                            color=st.session_state.border_color, 
                                                            label_alignment="la",
                                                            snr=0.)
                    st.session_state.rerun_active = True
                    st.rerun() # this immediately forces a run-run through the loop so that the last thing entered here will be drawn - kinda janky, but works for now

    with image_area:
        image_area.subheader("Sightline Data")
        image_area.dataframe(st.session_state.sightlines)

    with spectrum_area:
        spectrum_area.subheader("Spectra")
        var = st.session_state.fits_orig['var']
        wave = st.session_state.fits_orig['wave']
        
        cols = st.columns((1, 6))
        for index, s in st.session_state.sightlines.iterrows():
            cols[1].pyplot(draw_spectrum(index, s.x, s.y, wave, flux, var, s.radius, s.color))

        if st.session_state.sl_current is not None:
            s = st.session_state.sl_current
            index = len(st.session_state.sightlines)
            cols[1].pyplot(draw_spectrum(index, s.x, s.y, wave, flux, var, s.radius, s.color))


def main():
    print("==========================================")
    print("main.py")
    print("==========================================")

    init_page()
    init_graphics()
    init_session_state()
    init_sidebar()
    show_main_area()
    
    print("==========================================")
    print("end of main.py")
    print("==========================================")
  
if __name__ == "__main__":
    main()   