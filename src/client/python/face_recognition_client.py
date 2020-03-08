from tensorrtserver.api import *
from util import *
from image_preprocess import *
import json
import cv2
from custom_decorator import *
from model import *

import streamlit as st

def setup_ui():
    # setup sidebar
    url_text = st.sidebar.text_input("input url of the inference server", "trtis-server-container:8000")
    model_name = st.sidebar.selectbox("select the model", ("face_recognition_model",))
    protocol_name = st.sidebar.selectbox("select the protocol", ("http", "grpc"))
    
    # setup file picker
    uploaded_file = st.file_uploader("Choose a image file", type="jpg")
    if uploaded_file is not None:
        # setup environment
        ctx_param = ContextParameter(model_name=model_name)
        ctx_param.server_url = url_text
        ctx_param.http_headers = None
        ctx_param.verbose = True
        ctx_param.protocol = ProtocolType.from_str(protocol_name) # http or grpc
        ctx_param.model_version = -1
        batch_size = 1
        check_health = False
        
        # get parameter from model config
        model_param = get_model_parameter(ctx_param.server_url, ctx_param.protocol, ctx_param.model_name, ctx_param.verbose)
        input_width = model_param.input[0].dims[1]
        input_height = model_param.input[0].dims[0]
        input_size=(input_height, input_width)
        
        # read image
        imgs, orig_imgs = get_byte_data_from_imgs([uploaded_file], size=input_size)
        st.write("image size is " + str(orig_imgs[0].size))
    
        # send infer request
        corr_id = 1000
        datas_dict = {corr_id: imgs}
        with st.spinner('Wait for infer...'):
            result_map = infer(ctx_param, imgs[0].shape, imgs[0].dtype, datas_dict)
        st.success('infer request is completed!')
        
        # draw face boxes
        imgs_with_box = draw_face_box_from_json(orig_imgs, input_size, result_map[corr_id])
        
        # set images
        st.image(imgs_with_box)

@stop_watch
def infer(ctx_param, shape, dtype, datas_dict, batch_size=1):
    """
    infer function
    Arguments:
        ctx_param {ContextParameter} -- ContextParameter
        shape -- data shape (ex. (3, 480, 640))
        datas_dict {dict} -- key is correlation id, value is data list
        batch_size {int} -- batch size
    """
    infer_ctx = InferContext(ctx_param.server_url, 
                             ctx_param.protocol, 
                             ctx_param.model_name, 
                             ctx_param.model_version,
                             http_headers=ctx_param.http_headers, 
                             verbose=ctx_param.verbose)
    
    result_map = {}
    for corr_id, img_list in datas_dict.items():
        # send start request
        send_request(infer_ctx, corr_id, [np.zeros(shape, dtype=dtype)], batch_size=batch_size, start_of_sequence=True)
        
        # send the image data
        for img in img_list:
            send_request(infer_ctx, corr_id, [img], batch_size=batch_size) 
        
        # send end request
        result = send_request(infer_ctx, corr_id, [np.zeros(shape, dtype=dtype)], batch_size=batch_size, end_of_sequence=True)
        
        # postprocess result
        result_map[corr_id] = json.loads(result["OUTPUT"][0][0])

    return result_map

def draw_face_box_from_json(orig_imgs, input_size, result_json):
    imgs_with_box = []
    image_infos = result_json["result"]["image_infos"]
    for image_info, orig_img in zip(image_infos, orig_imgs):
        face_infos = image_info["face_infos"]
        img_array = np.array(orig_img)
        for face_info in face_infos:
            name = face_info["name"]
            box = face_info["box"]
            draw_face_box_with_name(img_array, input_size, box, name)
        imgs_with_box.append(img_array)
    return imgs_with_box

def draw_face_box_with_name(orig_img, input_size, box, name):
    # convert the point
    height_ratio = float(orig_img.shape[0])/float(input_size[0])
    width_ratio  = float(orig_img.shape[1])/float(input_size[1])
    top    = int(float(box["top"]) * height_ratio)
    bottom = int(float(box["bottom"]) * height_ratio)
    left   = int(float(box["left"]) * width_ratio)
    right  = int(float(box["right"]) * width_ratio)
    
    # Draw a box around the face
    cv2.rectangle(orig_img, (left, top), (right, bottom), (0, 0, 255), 2)
    
    # Draw a label with a name below the face
    cv2.rectangle(orig_img, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(orig_img, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

def main():
    setup_ui()

if __name__ == "__main__":
    main()
