# sightline-explorer

A streamlit app to aid in the exploration and understanding of DLAs using the gravitationally lensed images of background quasars such as J1429. Though this we study the small scale effects of DLAs, and specifically the faint emission lines within the DLAs to better understand the host galaxies. The more lines of sight we have, the better *triangulation* we will have to determine the location of the host
galaxies (assuming our lensing model is right for the lensing galaxy) - ray-tracing back to the host galaxies.

## environment setup

I created a new environment for streamlit apps like this

> conda create --name streamlit --file requirements.txt
> conda activate streamlit
> streamlit run src/main.py

And you should see something like this:
![sightline-explorer.png](./assets/sightline-explorer.png)

