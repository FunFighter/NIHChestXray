from _thread import start_new_thread

#Attach CSV to Images Folder and defline our traiing validation and test
def vaild_gen(batch_fit):
    global valid_generator
    valid_generator = datagen.flow_from_dataframe(
        dataframe=train_df,
        directory="../images/",
        x_col="Image_Index",
        y_col="Finding_Labels",
        subset="validation",
        class_mode="categorical",
        target_size=(15,15),
        batch_size=batch_fit)
    
def train_gen(batch_fit):
    global train_generator
    train_generator=datagen.flow_from_dataframe(
        dataframe=train_df,
        directory="../images/",
        x_col="Image_Index",
        y_col="Finding_Labels",
        subset="training",
        class_mode="categorical",
        target_size=(15,15),
        batch_size=batch_fit)
    
def test_gen(batch_fit):
    global test_generator
    test_generator=test_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory="../images/",
        x_col="Image_Index",
        y_col="Finding_Labels",
        class_mode="categorical",
        target_size=(15,15),
        batch_size=batch_fit)
    
def init_gens(batch_fit):
    start_new_thread(vaild_gen,(batch_fit, ))
    start_new_thread(train_gen,(batch_fit ,))
    start_new_thread(test_gen,(batch_fit, ))
     
    return test_generator , train_generator , valid_generator