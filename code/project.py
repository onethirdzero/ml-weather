import sys
import numpy as np
import pandas as pd
from keras import optimizers
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Dropout
import keras.applications.inception_v3 as inception
import final_models.cnn_d as OurNetwork
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
IMSIZE = (196, 196)

if(len(sys.argv)<2):
    print("Please supply image path.")
    print("Exapmle:")
    print("python project.py /PATH/TO/IMAGE")
    print("Exiting.....")
    sys.exit()
    
PHOTO_PATH = sys.argv[1]
# PHOTO_PATH = r"\\devmachine\e$\data\katkam-secret-location\katkam-scaled"
# PHOTO_PATH = r"/Users/jundali/Desktop/katkam-scaled"
# PHOTO_PATH = r"../katkam-scaled"

TRAIN_VAL_RATIO = 0.8
def showImg(filename):
    img = image.load_img(PHOTO_PATH+"/"+filename, target_size=IMSIZE)
    x = image.img_to_array(img)
    plt.imshow(img)
    plt.show()


cleaned_data = pd.read_csv("cleaned_data.csv")
weather_array = list(map(lambda x:x.split(',') ,cleaned_data["Mapped"].values))
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(weather_array)
cleaned_data["Y"] = list(Y)

train_val_mask = np.random.rand(len(cleaned_data)) < TRAIN_VAL_RATIO
val_data = cleaned_data[~train_val_mask]
cleaned_data = cleaned_data[train_val_mask]


# cleaned_data
print(mlb.classes_)

def balance_data (df):
    clear_days_mask = df.apply(lambda x: x["Y"][0]==1,axis = 1)
    clear_days = df[clear_days_mask]
    non_clear_days = df[~clear_days_mask].sample(len(clear_days))
    final_data = clear_days.append(non_clear_days)
    final_data["Y2"] = cleaned_data.apply(lambda x: [x["Y"][0],(x["Y"][0]+1)%2],axis = 1)
    final_data = final_data.sample(frac=1)
    return final_data
# final_data = balance_data(cleaned_data)

def sub_balanced_data(df):
    foggy_days  = cleaned_data[cleaned_data.apply(lambda x: x["Y"][2]==1,axis = 1)]
    rainy_days  = cleaned_data[cleaned_data.apply(lambda x: x["Y"][3]==1,axis = 1)]
    snowy_days  = cleaned_data[cleaned_data.apply(lambda x: x["Y"][4]==1,axis = 1)]
    just_cloudy_days  = cleaned_data[cleaned_data.apply(lambda x: (x["Y"][2]==0) and (x["Y"][3]==0) and (x["Y"][4]==0),axis = 1)]
    samples_per_label = len(snowy_days)
    final_data = foggy_days.sample(samples_per_label).append(rainy_days.sample(samples_per_label))
    final_data = final_data.append(snowy_days.sample(samples_per_label))
    final_data = final_data.append(just_cloudy_days.sample(samples_per_label))
    return final_data
# cleaned_data["Y3"] = cleaned_data.apply(lambda x: x["Y"],axis = 1)
cleaned_data["Y2"] = cleaned_data.apply(lambda x: [x["Y"][0],(x["Y"][0]+1)%2],axis = 1)
cleaned_data["Y3"] = cleaned_data.apply(lambda x: list(x["Y"])[2:5],axis = 1)
final_data = sub_balanced_data(cleaned_data)
train_test_mask = np.random.rand(len(final_data)) < TRAIN_VAL_RATIO
train_data = final_data[train_test_mask]
test_data = final_data[~train_test_mask]


base_model = OurNetwork.network(classes = 2)
# for layer in base_model.layers:
#     layer.trainable = False
x = Flatten(name='sub_flatten')(base_model.get_layer('block3_pool').output)
x = Dense(64, activation='relu', name='sub_fc1')(x)
x = Dropout(0.3)(x)
sub_predictions = Dense(3, activation='sigmoid', name='sub_predictions')(x)
predictions = (base_model.get_layer('predictions').output)
model = Model(inputs=base_model.input, outputs=[predictions, sub_predictions])

model.compile(loss='binary_crossentropy', optimizer="adam", loss_weights={'predictions': 0, 'sub_predictions': 1}, metrics=['accuracy'])
for layer in model.layers:
    layer.trainable = False
print(model.summary())

img_datagen = ImageDataGenerator(width_shift_range=0.2,height_shift_range=0.2)

def my_load_img(img_path,img_datagen,size):
    img = image.load_img(img_path, target_size=size)
    x = image.img_to_array(img)
    x = img_datagen.random_transform(x)
    x = img_datagen.standardize(x)
    return x
def my_img_generator(df,img_datagen,batch_size):
    index = 0
    count = 0
    img_datas=[]
    img_labels=[]
    img_sub_labels =[]
    while 1:
        # create numpy arrays of input data
        item = df.iloc[index]
        if count < batch_size:
            img_datas.append(my_load_img(PHOTO_PATH+"/"+item["Filename"],img_datagen,IMSIZE))
#             [np.array(one_hot_labels),np.array(img_bboxes)]
            img_labels.append(item["Y2"])
            img_sub_labels.append(item["Y3"])
            index=(index+1)%df.shape[0]
            count+=1
        else:
            count=0
            yield (np.array(img_datas),[np.array(img_labels),np.array(img_sub_labels)])

            img_datas = []
            img_labels = []
            img_sub_labels =[]
      

batch_size=64
my_train_generator = my_img_generator(train_data,img_datagen,batch_size)
my_test_generator = my_img_generator(test_data,img_datagen,batch_size)


model.load_weights('final_models/sub_cnn_b.h5',by_name=True)

###############################################################################
# Data training
# model.save_weights('cnn_c.h5') 

# for i in range(0):
#     '''Refresh data'''
#     final_data = sub_balanced_data(cleaned_data)
#     train_test_mask = np.random.rand(len(final_data)) < TRAIN_VAL_RATIO
#     train_data = final_data[train_test_mask]
#     test_data = final_data[~train_test_mask]
#     my_train_generator = my_img_generator(train_data,img_datagen,batch_size)
#     my_test_generator = my_img_generator(test_data,img_datagen,batch_size)
    
#     model.fit_generator(
#             my_train_generator,
#             steps_per_epoch = 5,
#             epochs = 10,
#             validation_data = my_test_generator,
#             verbose = 2,
#             validation_steps = 5)
#     model.save_weights('new_sub_cnn.h5') 
###############################################################################


def batchPredict(imageNames):
    result = []
    for filename in imageNames:
        img_path = PHOTO_PATH+"/"+filename
        img = image.load_img(img_path , target_size=IMSIZE)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = img_datagen.standardize(x)
        preds = model.predict(x)
        result.append([preds[0].reshape(2),preds[1].reshape(3)])
    return result

def get_Y3_Label(preds):
    y3_label = ""
    if(preds[0][0]>0.5):
        return "Clear"
    else:
        if(preds[1][0]>0.5):
            y3_label+="Fog"
        if(preds[1][1]>0.5):
            y3_label+=" Rain"
        if(preds[1][2]>0.5):
            y3_label+=" Snow"
        if(y3_label == ""):
            y3_label="Cloudy"
    return y3_label

def get_predicted_y(preds):
    y = [0, 0, 0, 0,0]
    if(preds[0][0]>0.5):
        y[0] = 1
    else:
        flag = False
        if(preds[1][0]>0.5):
            y[2] = 1
            flag = True
        if(preds[1][1]>0.5):
            y[3]=1
            y[1] = 1
            flag = True
        if(preds[1][2]>0.5):
            y[4]=1
            y[1] = 1
            flag = True
        if (~flag):
            y[1]=1
    return y

sample_data = val_data
print("Predicting ", len(sample_data)," images")
sample_data["Predict_combined"] = batchPredict(sample_data["Filename"].values)
sample_data["Predict_Y2_Label"] = sample_data["Predict_combined"].apply(lambda x: "Clear" if x[0][0]>0.5 else "Non_Clear")
sample_data["Predict_Y3_Label"] = sample_data["Predict_combined"].apply(lambda x: get_Y3_Label(x))
sample_data["Predict_Y"] = sample_data["Predict_combined"].apply(lambda x: get_predicted_y(x))


print("\n\n##############################")
print("Accuracy score: ")
print(accuracy_score(np.stack(sample_data["Y"].values),np.stack(sample_data["Predict_Y"].values)))
print("##############################")
print("Randomly printing 20 samples with their true Y and the predicted Y.")
print(sample_data.sample(20)[["Filename","Weather","Predict_Y2_Label","Predict_Y3_Label"]].to_string(index=True))
