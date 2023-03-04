import os
train_dir = '/home/arapkering/Desktop/dogs-vs-cats/train'

"""Note the os inbuilt function os.path.splitext(file) returns two things : the  first string , that comes before the 
file extension and the second string that is the file extension itself . for example , assume the file name is : 
cat.1067.jpg , then  then the os.path.splitext(file) returns two strings : 'cat.1067' and '.jpg' . Training will be 
happening in batches,  that is , each batch  will be having 100 images , 80 for training and 20 for validation The 
model is also trained to  perform binary classification , that is , either 0 or 1. rescale=1./255 scales the pixels 
for the image , making it range between 0 and 1.(note that each image is represented as an matrix of pixels ) by 
shuffling data , we ensure randomness , improving the results .(no need for shuffling the validation data ) 
flow_from_directory() will say that it found 20,000 images in a directory , and these 20000 belong to two classes . 
the ImageDataGenerator will be generating a stream of 100 image tensors at a time only when the first batch is 
consumed tha means  it will only generate when the next() function is called on this generator , the following 
function creates an array of image tensor(each image  is represented as a matrix ) and an axis object (a small figure 
inside the Figure main object , ) it then associates each image with a sub_plot and every subPlot object has a 
function called imshow , that plots the x values against the y in itself. (20,20) tuple means that the whole big 
figure will have a size of 20 inches. Note that teh zip function  is used to iterate over a two arrays 
simultaneously, that is in every loop , the function zip will return a tuple(image ,axis) one item in the images_arr 
and ine item in  the axes array , which is a actually a sub plot object


 Note the plt.tight_layout() is  called on the main plt object , to adjust the position of the sub plots 
    so that none of the subplots overlaps 
    
"""

for fullFileName in os.listdir(train_dir):
    filename, file_extension = os.path.splitext(fullFileName)
    if file_extension.lower() == '.jpg':
        if filename.startswith('c'):
            os.replace(os.path.join(train_dir, fullFileName), os.path.join(train_dir, 'cat', fullFileName))
        else:
            os.replace(os.path.join(train_dir, fullFileName), os.path.join(train_dir, 'dog', fullFileName))
