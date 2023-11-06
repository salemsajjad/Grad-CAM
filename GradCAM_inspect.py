# PFK CNN imports
from model_utils.data_generators import resize_pad
from model_utils import model_definitions
import numpy as np
import cv2
import os

# =============================================================================================================

# Grad-CAM imports
import warnings
warnings.filterwarnings("ignore")
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
import os
import cv2

# =============================================================================================================

# define the sources
src_dir = r"G:\mash_datasets\New_Dataset\First Dataset\I035-05\validation\sefid_shate"
# image_path  = r"~0_13-48-58-458\DOWN_CAM_2_image.bmp"

# =============================================================================================================

# choose the model
MODEL_BUILDER = model_definitions.MCM01
train_id      = 'N007-RGB'
checkpoint_id = "1402.07.10 10-34-27"
input_shape   = (72,72,3)
output_shape  = 4

model = MODEL_BUILDER(input_shape, output_shape)
model.summary()
model_name  = f"{MODEL_BUILDER.__name__}-{train_id}"
model.load_weights(os.path.join(os.path.join(os.getcwd(), 'model_files', 'weights', model_name, checkpoint_id), 'variables', 'variables'))

# =============================================================================================================

# Inform the code your dataset labels list
# classes_dict = {"not_tarakdar":0, "tarakdar":1}
classes_dict = {"salem":0, "sefid_shate":1, "shekaste_abkhorde":2, "sorakhdar":3}

# =============================================================================================================

# Define the vis_model and print layer names
# select all the layers for which you want to visualize the outputs and store it in a list
outputs = [layer.output for layer in model.layers[1:24]]

# Define a new model that generates the above output
vis_model = Model(model.input, outputs)

layer_names = [layer.name.split("/")[0] for layer in outputs]
print("Layers that will be used for visualization: ")
print(layer_names)
# =============================================================================================================

# Grad-CAM function definitions
def get_CAM(processed_image, actual_label, layer_name='conv2d_5', pred_index=None):
    model_grad = Model([model.inputs],
                       [model.get_layer(layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_output_values, predictions = model_grad(processed_image)

        # watch the conv_output_values
        tape.watch(conv_output_values)

        ## Use binary cross entropy loss
        ## actual_label is 0 if cat, 1 if dog
        # get prediction probability of dog
        # If model does well,
        # pred_prob should be close to 0 if cat, close to 1 if dog
        # pred_prob = predictions[:,1]
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])

        class_channel = predictions[:, pred_index]

        # make sure actual_label is a float, like the rest of the loss calculation
        actual_label = tf.cast(actual_label, dtype=tf.float32)

        # add a tiny value to avoid log of 0
        # smoothing = 0.00001

        # Calculate loss as binary cross entropy
        loss = 0
        # loss = -1 * (actual_label * tf.math.log(pred_prob + smoothing) + (1 - actual_label) * tf.math.log(1 - pred_prob + smoothing))
        # print(f"binary loss: {loss}")

    # get the gradient of the loss with respect to the outputs of the last conv layer
    # grads_values = tape.gradient(loss, conv_output_values)
    grads_values = tape.gradient(class_channel, conv_output_values)
    grads_values = K.mean(grads_values, axis=(0,1,2))
    conv_output_values = np.squeeze(conv_output_values.numpy())
    grads_values = grads_values.numpy()

    # weight the convolution outputs with the computed gradients
    for i in range(model.get_layer(layer_name).output_shape[-1]): # The number of filters of the last convolution
        conv_output_values[:,:,i] *= grads_values[i]
    heatmap = np.mean(conv_output_values, axis=-1)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= heatmap.max()

    del model_grad, conv_output_values, grads_values, loss

    return heatmap

# =============================================================================================================

def show_sample(image_path):

    # Pick image with its label
    image = cv2.imread(image_path)
    image_name = image_path.split("\\")[-1]
    image_label = image_path.split("\\")[-2]
    sample_label = classes_dict[image_label]

    sample_image = resize_pad(image, input_shape[:2])
    if input_shape[-1] == 3:
        sample_image_before_expand = sample_image.copy()/255
        # sample_image_before_expand = cv2.cvtColor(sample_image, cv2.COLOR_BGR2GRAY)/255
        sample_image_processed = np.expand_dims(sample_image_before_expand, axis=0) #axis=(0, 3)
    elif input_shape[-1] == 1:
        sample_image_before_expand = cv2.cvtColor(sample_image, cv2.COLOR_BGR2GRAY)/255
        sample_image_processed = np.expand_dims(sample_image_before_expand, axis=(0, 3))

    activations = vis_model.predict(sample_image_processed)
    # pred_label = np.argmax(model.predict(sample_image_processed), axis=-1)[0]

    output = model.predict(sample_image_processed, verbose=0)[0]
    pred_label = list(classes_dict.values())[np.argmax(output)]

    sample_activation = activations[0][0,:,:,16]

    sample_activation-=sample_activation.mean()
    sample_activation/=sample_activation.std()

    sample_activation *=255
    sample_activation = np.clip(sample_activation, 0, 255).astype(np.uint8)

    heatmap = get_CAM(sample_image_processed, sample_label)
    heatmap = cv2.resize(heatmap, (sample_image_before_expand.shape[0], sample_image_before_expand.shape[1]))
    heatmap = heatmap *255
    heatmap = np.clip(heatmap, 0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
    converted_img = sample_image.copy()/255
    super_imposed_image = cv2.addWeighted(converted_img.astype('float32'), 0.6, heatmap.astype('float32'), 2e-3, 0.0)

    # =================================== Plot the images with matplotlib =======================================

    # f,ax = plt.subplots(2,2, figsize=(15,8))

    # ax[0,0].imshow(sample_image)
    # ax[0,0].set_title(f"True label: {list(classes_dict.keys())[sample_label]} \n Predicted label: {list(classes_dict.keys())[pred_label]}")
    # ax[0,0].axis('off')

    # ax[0,1].imshow(sample_activation)
    # ax[0,1].set_title("Random feature map")
    # ax[0,1].axis('off')

    # ax[1,0].imshow(heatmap)
    # ax[1,0].set_title("Class Activation Map")
    # ax[1,0].axis('off')

    # ax[1,1].imshow(super_imposed_image)
    # ax[1,1].set_title("Activation map superimposed")
    # ax[1,1].axis('off')

    # plt.tight_layout()
    # plt.show()


    # =================================== Show stacked images with openCV =======================================
    # Stack the images horizontally
    resize_factor = 5
    super_imposed_image = (super_imposed_image*255).astype('uint8')
    super_imposed_image_resized = cv2.resize(super_imposed_image, [super_imposed_image.shape[0]*resize_factor, super_imposed_image.shape[1]*resize_factor])
    sample_image_resized = cv2.resize(sample_image, [sample_image.shape[0]*resize_factor, sample_image.shape[1]*resize_factor])
    heatmap_resized = cv2.resize(heatmap, [heatmap.shape[0]*resize_factor, heatmap.shape[1]*resize_factor])

    stacked_horizontal = cv2.hconcat([sample_image_resized, heatmap_resized, super_imposed_image_resized])

    # Define the text to be added
    text = f"True label: {list(classes_dict.keys())[sample_label]}, Predicted label: {list(classes_dict.keys())[pred_label]}"

    # Define the font settings
    font = cv2.FONT_HERSHEY_COMPLEX
    font_scale = 0.5
    font_thickness = 1
    font_color = (255, 255, 255)  # White color in BGR
    position = (10, 20)  # (x, y) coordinates

    # Add the text to the stacked image
    stacked_with_text = cv2.putText(stacked_horizontal, text, position, font, font_scale, font_color, font_thickness, cv2.LINE_AA)

    # Display the stacked image with text
    cv2.imshow(f'{image_name}', stacked_with_text)
    cv2.imwrite(os.path.join(r"model_utils\grad_output", image_name), stacked_with_text)

    # Wait for a key event and close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return activations

# =============================================================================================================

# Choose an image index to show, or leave it as None to get a random image
# activations = show_sample(idx=None)

for parent_dir, _, files in os.walk(src_dir):
    for fname in files:
        if not fname.endswith(".bmp"): continue
        image_path = os.path.join(parent_dir, fname)
        activations = show_sample(image_path)

