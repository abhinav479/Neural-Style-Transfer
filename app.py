import streamlit as st
import os
from nst import NeuralStyleTransfer
from PIL import Image

def main():
    st.title("Neural Style Transfer")
    content_folder = 'Images'
    style_folder = 'Style'
    output_folder = 'Output'
    
    st.header("Upload your images")
    
    uploaded_content_files = st.file_uploader("Choose content images", accept_multiple_files=True)
    uploaded_style_files = st.file_uploader("Choose style images", accept_multiple_files=True)
    
    if st.button("Stylize"):
        if uploaded_content_files and uploaded_style_files:
            st.write("Stylizing images... Please wait.")
            nst = NeuralStyleTransfer()
            
            for content_file in uploaded_content_files:
                content_path = os.path.join(content_folder, content_file.name)
                with open(content_path, "wb") as f:
                    f.write(content_file.getbuffer())
                
                for style_file in uploaded_style_files:
                    style_path = os.path.join(style_folder, style_file.name)
                    with open(style_path, "wb") as f:
                        f.write(style_file.getbuffer())
                    
                    output_path = os.path.join(output_folder, f"{content_file.name.split('.')[0]}_{style_file.name.split('.')[0]}.png")
                    st_progress_bar = st.progress(0)
                    losses = nst.run(content_path, style_path, output_path, epochs=2, steps_per_epoch=100, st_progress_bar=st_progress_bar)
                    st.image(output_path, caption=f"Stylized Image: {content_file.name} with {style_file.name}")

if __name__ == "__main__":
    main()
