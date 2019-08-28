(Upcoming push for the updated model and model break down notebooks coming soon)
![Image Preview](https://raw.githubusercontent.com/FunFighter/NIHChestXray/master/sample%20img.png)


# NIH Chest X-ray Data Set
These are chest x-rays provided by the National Institutes of Health - Clinical Center at https://nihcc.app.box.com/v/ChestXray-NIHCC

# High level overview
After you go to the local webpage you can go to our upload page where you can submit your own chest x-ray to run into our convnet model.
For the record this was built in three weeks, not 4 years of medical school. Please do not seriously use this for diagnosis.

## Data handling
The data had some funny things like a “412 year old”. Outside of that the data was clean.
Should also be noted that some diseases are very similar on x-rays or related issues.
We prepared the data with One Hot Encoding. 

## Building the Model.
The model is built in Keras and only really using Tensorflow/cuda to handle local system resources (Hey had to justify my computer spending somehow). There is also a Google Colab notebook I built but doesn't have our final model currently . 
We tried many different way of handing multiple output tensors but found that the best way to deal with this multiple classification problem was to use the functional API to build a Binary classification for each diseases. There was a million and one more things I would like to have tried but with the time constraints this was as good as I could tune it.

## Web-app Back end
For the routing we went with Flask.
Outside of being quick to set up and easy. It was great for doing the upload handing.
We used it to adjust the image down to 252x252 so it would fit into our input shape.
Also was great since it didn't put up much of a fight connecting our keras .h5 model.

## Front end
(I will try to get details from the people who worked on this later)
The webpage was build using Bootstrap, and a the visuals were built Tableau.
