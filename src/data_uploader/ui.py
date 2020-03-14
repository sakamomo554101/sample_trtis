import streamlit as st
import io
import csv
import os
import pandas as pd
import requests


def setup_ui():
    url = st.sidebar.text_input("input server url", "trtis-server-container")
    page = st.sidebar.radio("move page", ["main", "list", "reload"])
    if page == "main":
        render_main_page()
    elif page == "list":
        render_list_page()
    elif page == "reload":
        render_reload_page(url=url)
    else:
        st.write("error page")

def render_main_page():
    st.title("Main page")
    face_name = st.text_input("input face name", "")
    uploaded_file = st.file_uploader("Choose a image file", type="jpg")
    if len(face_name) > 0 and uploaded_file is not None:
        # check the duplicate name
        file_name = face_name + ".jpg"
        face_map = read_face_map()
        if face_name in face_map.keys():
            # remove file and face data
            os.remove(get_image_path(face_map[face_name]))
            delete_face_data(face_name)
            
        # save face file
        image_path = get_image_path(file_name)
        face_map[face_name] = file_name
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # register face name
        write_face_data(face_name, file_name)
        
        st.write("save complete")

def render_list_page():
    st.title("List page")
    face_map = read_face_map()
    st.write(
        pd.DataFrame({
            "face name": list(face_map.keys())
        })
    )

def render_reload_page(url):
    base_post_url = "http://" + url + ":8000" + "/api/modelcontrol"
    load_url = base_post_url + "/load/face_recognition_model"
    unload_url = base_post_url + "/unload/face_recognition_model"
    
    # reload model
    response = requests.post(unload_url)
    if response.status_code != 200:
        st.write("unload error : status code is " + str(response.status_code))
        return
    response = requests.post(load_url)
    if response.status_code != 200:
        st.write("load error : status code is " + str(response.status_code))
    st.write("reload complete")

def read_face_map(csv_path="dataset/face/face.csv"):
    with open(csv_path) as csv_file:
        reader = csv.reader(csv_file)
        face_map = {}
        for face in reader:
            face_map[face[1]] = face[0] # Map key is face_name. Map value is image path of face data.
        return face_map

def write_face_data(face_name, file_name, csv_path="dataset/face/face.csv"):
    with open(csv_path, "a") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([file_name, face_name])

def delete_face_data(face_name, csv_path="dataset/face/face.csv"):
    df = pd.read_csv(csv_path)
    df = df.drop(df[df[0] == face_name])
    df.to_csv(csv_path)

def get_image_path(file_name):
    # TODO : 外からディレクトリパスが設定できる様にする（CustomInstanceのface modelの対応も同時に必要）
    dir_path = "dataset/face/image"
    return os.path.join(dir_path, file_name)

def main():
    setup_ui()

if __name__ == "__main__":
    main()
