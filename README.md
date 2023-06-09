# sightline-explorer

## Environment Setup

To run this streamlit app locally, once you've done the necessary ```git clone git@github.com:jerichoBob/sightline-explorer.git```, create a new environment for your streamlit app, activate that environment and then run the app:
```sh
conda create --name streamlit --file streamlit_env_requirements.txt
conda activate streamlit
streamlit run src/main.py
```

**NOTE:** the ``` requirements.txt ``` file was created with ``` pip freeze > requirements.txt ``` and is only used when deploying to the streamlit.app cloud environment (or when you prefer not to use conda).

**NOTE:** This app uses [git-lfs](https://git-lfs.com/) to manage the large (>50MB) FITS data files pushed out with this deploy. Not relevant for you unless you plan on deploying apps with large data files to github.

## Description

A streamlit app to aid in the exploration and understanding the fine structure of the Circumgalactic Medium (CGM) and DLAs using the gravitationally lensed images of background quasars such as J1429. Though this we study the small scale effects of CGM and DLAs gases, and specifically the faint emission lines within the DLAs to better understand the host galaxies. The more lines of sight we have, the better *triangulation* we will have to determine the location of the host galaxies - ray-tracing back to the host galaxies (assuming a correct lensing model for the lensing galaxy).

**sightline-explorer** reads in a FITS flux data cube and an associated variance data cube (both deployed with the app for now) and displays a whitelight (or suitable variant) of the flux cube to the user. 

This is the first thing you should see:
<img src="./assets/sightline-explorer-initial.png" width="500" >

Kinda homely right now, I know. But it will get better. 

When you click on the flux image, the following will happen:

1) A bounding numbered box will be drawn on the flux image representing the area being captured.
2) A 1D spectral extraction of the flux and variance cube will be performed for the sampled area. This will be plotted on the right side of the app. The numbered label assocated with the sample area on the image will correspond to the number label next to the extracted spectra.
   <img src="./assets/sightline-explorer-selected.png" width="500" >
3) TBD - The Signal-to-Noise ratio will be calculated and displayed next to the spectra.
4) The user can use the live SNR data to determine if the size and placement of the sampled area is appropriate. If not, the user can click on a different location of the image and repeat the above steps.
5) Once happy with the spectra and SNR for this location, click the "Accept" button which will lock in the selection and save the relevant parameters (ie x,y location, bounding box size, color, SNR, etc). 
   <img src="./assets/sightline-explorer-accepted.png" width="500" >
6) Clicking on the image again starts the whole process over again, this time with selection 1 and spectra 1. Do this several times and the UI starts to look less homely...
   <img src="./assets/sightline-explorer.png" width="500" >

