# sightline-explorer

Created with the loving support of ChatGPT (thanks, dude!). Here was the initial prompt
```
awesome. I have a streamlit environment created in conda and have that environment activated. I would like to plan for a certain amount of modularity in the future so that the functionality can be extended without too much effort. I would like to keep the ui components separate as much as possible from the data analysis functionality.  

The primary (non-standard) libraries that will be used for kcwi will be rbcodes (https://github.com/rongmon/rbcodes) and kcwitools (https://github.com/pypeit/kcwitools)

The specification is as follows:
The user selects a flux and variance kcwi data cube from a local folder.
One part of the UI will need to show the aperture details. The aperture is used a mask when sampling the image (discussed below). The aperture shape will be a circular weighted mask, normalized so that all of the mask points add up to 1. The ui will allow the user to change the radius of the aperture, the linewidth,style and color of the aperture bounding box. The UI also shows a grayscale image of the mask values

The user will be presented with an flux whitelight image (read from the kcwi flux file). 
The user will then select a point on that image. The cursor will be in the shape of the aperture bounding box, with the selected point at the center of the bounding box. The selected x,y coordinate point will be added to a scrollable list  The act of selecting the point on the image will also extract a 1D spectra by applying the aperture mask to the image across the flux and variance cubes. Those masked flux and variance values will be passed to a new method 'extract_circular_aperture' within kcwitool/spec.py. This new extracted spectrum will displayed and a signal to noise ratio calculation will be performed using the masked versions of the flux to variance data. The UI will display this SNR value along with the plot of the extracted spectrum

I know that's alot.  Does that specification make sense?
```

He said it did, and with that, ***we*** got to coding. :D
