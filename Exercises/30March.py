###From array to grayscale image

#• Use numpy (for example, the repeat function) to create a 5x5 pixel array
#with 0s and 1s. For example you can try to create black and white stripes,
#or a checkerboard pattern. Set the middle pixel to 0.5

#• Multiple the entire image by 255 and store this in another variable. Check
#that the dimensions of the image stay the same.

#• Display both images with matplotlib, What do you see? What do you
#need to do, to make the images appear black and white?

###Color image

#• Now try to create a 5x5 color checkerboard with random colors. This can
#be done in 1 line with the randint function from numpy.random. Display
#the image

#• If you run the code above again, the image will change. You will need to
#set the random seed, for the image to stay the same.

#• Compare how you would describe the colors, to the RGB values of the
#pixels. Are you able to define what kind of values would translate into for
#example red, brown, or yellow?

#• Display the R,G,B channels as images separately. Do the individual values
#tell you enough about the colors of the pixels?
#Resizing an image

#• Load a color image, it can be an image from the dataset or for example a
#photo that’s at least 500x500 pixels. Store the original size in pixels, and
#the original size in bytes, in different variables.

#• Explore the documentation for skimage.transform. Try to resize the
#image to 20% of its size in each dimension, and save the image as a different
#file. Record the new size in bytes, how much does it change?

#• Explore the documentation for skimage.util which has several functions
#for converting the image to different formats. Can you decrease the image
#size in bytes, without changing the image size in pixels? How does this
#affect the image quality?

###Cropping image based on the mask

#• Load an image and its mask, store these in im and mask.
#• Use the numpy package together with mask to find out the minimum/maximum
#coordinates where the lesion is present.

#• Create padded image, which extracts only the lesion from the image, with
#a bit of padding on each side.

###Create some masks

#Create some lesion masks for a few images in the dataset. You can do this
#with a local installation of LabelStudio (labelstud.io) which the TAs can help
#you with. Other software that allows you to create your own outlines is also
#allowed.
